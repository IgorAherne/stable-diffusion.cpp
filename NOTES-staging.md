# The weights' staging to the card and the PCIe link's power state (2026-10-25)

An Opus agent's investigation of "the diffusion weights' staging takes 1.33 s per picture on the GTX 1070 pod (4,005 MB,
~3 GB/s)". No engine code changed; the measurements are the RTX 3060's (Windows 11, PCIe 4.0 x16), the pod's are to do.

## What the pod's 1.33 s is
- The pod's runs pin the diffusion weights: `run_sd.py` (and `suite.py` through it) sets `SD_PIN_PARAMS=diffusion`
  unless `--no-pin`, and the hub's `--pin auto` pins too there (its RAM is read from the cgroup limit, 32 GiB >= 22 GiB).
  Page-locked weights never go through the staging ring (`model_manager_staging.cpp` skips page-locked sources): each
  tensor is one `cudaMemcpyAsync` + sync from page-locked RAM, a direct DMA. So 4,005 MB / 1.33 s = 3.0 GB/s is the
  link's own rate at that moment, not a pageable-memory copy: a page-locked bounce buffer (option a: that is the ring,
  which exists since spellcast-perf-9fcb5c2) or `cudaHostRegister` (option b) cannot make it faster.
- 3.0-3.4 GB/s is what a x16 link moves at PCIe gen 1, the generation an idle NVIDIA card drops its link to.

## Measured on the RTX 3060 (`pcie_probe.py`, `pcie_wake.py`, `pcie_linger.py` in the hub's `code/devtools/qwen-speed`:
torch, 4 GiB in 64 MB pieces)
| case | rate |
|---|---|
| page-locked, cold card (idle 12-15 s: P8, link gen 1) | 11.2-12.3 GB/s over 4.29 GB (0.35-0.39 s); the first pieces 3.3-3.4 GB/s, then the link goes to gen 4 |
| page-locked, warm (right after compute: P2, gen 4) | 26.8 GB/s (0.160 s), from the first piece |
| page-locked, beside a running compute kernel | 26.8 GB/s |
| pageable, warm | 19.2 GB/s |
| after compute, idle 0 / 0.5 / 1 s, then the copy | 26.8 GB/s |
| idle 2 / 3 s | 22.8 GB/s |
| idle 5 / 8 s | 11.6 GB/s (cold again) |
| cold, a spin kernel of 2 / 5 / 10 / 20 / 50 / 100 ms first | copy 11.3 / 12.1 / 13.6 / 18.0 / 26.9 / 26.9 GB/s, but spin + copy 0.40-0.45 s against 0.385 s without: a cold card runs the spin at idle clocks, the wake-up takes ~0.25 s whatever triggers it |
| cold, the spin beside the copy | 10.7-11.2 GB/s: no help (on Windows the copy itself wakes the link) |

In sd-cli on this PC the diffusion staging (page-locked) takes 0.46-0.47 s: the weights are read into page-locked RAM
(3.7-6.0 s in these runs: "loading tensors completed") right before it, the card idles meanwhile and the copy starts cold; the floor is 0.16 s.

## Conclusion
- Option (c): the copy path stays as it is. The lever is the link's power state, not the copy mechanism.
- On Windows (the product) the cold start costs at most ~0.3 s for 4 GB (0.46 -> 0.16 s here), and only when the card
  has been idle for more than ~3 s before the staging (a picture right after the prompt's encoding starts warm).
- On the pod the whole copy runs at the gen-1 rate, so there the copy evidently does not wake the link (to verify; the
  host's maximum link may also be narrower than x16). If `pinned, warm` / `beside compute` reach ~12 GB/s there, a
  keep-awake kernel during the staging would take the 1.33 s to ~0.35 s on that pod.
- A way to take the Windows cold start too: start GPU work when the model manager begins reading the weights (the
  seconds of host work above), so the link is up when the copy starts. It needs a CUDA hook (a ggml-cuda proc address)
  and costs a few hundred ms of card power; not done.

## To run on the pod
In `code/devtools/qwen-speed` (synced to the pod with the hub's code): `CUDA_VISIBLE_DEVICES=0 python pcie_probe.py --gib 4 --idle 15`, `python pcie_wake.py --idle 12`,
`python pcie_linger.py --cool 12` (the scripts use torch; they print the rates and nvidia-smi's link generation and
width during each copy), and `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv`.

2026-10-29: answered on the next pod (instance 58301245, one GTX 1070): `nvidia-smi --query-gpu=pcie.link.gen.current,
pcie.link.width.current` reads gen 3, width 2 DURING a drawing step - the host gives the card two lanes (~2 GB/s), so
its staging of 4,005 MB takes 2.6 s whatever the power state; a user's x16 slot moves it in 0.3-0.4 s. Pod staging
times are the pod's, not the product's.

# Weights kept on the card between runs (2026-10-29, `SD_KEEP_STAGED_PARAMS`)

With the offload placement (`--params-backend *=cpu`) every runner's weights were staged to the card at `runner_start`
and evicted at `runner_end` (`GGMLRunner::runner_end` -> `evict_compute_backend_params`): the diffusion weights for
every picture (0.3 s from page-locked RAM, 1-2 s from the file's mapping on a 16 GB PC, a disk read once the pages were
dropped), the text encoder's for every new prompt. `SD_KEEP_STAGED_PARAMS=1` (read once, `ModelManager::
keeps_staged_params`) makes `runner_end` only unpin them (`release_compute_backend_params`): the staged blocks stay on
the card, the next run of that model finds them (`stage_tensors_to_compute_backend` skips staged tensors), and
another runner short of memory evicts them in `ensure_compute_backend_capacity` - idle weights least recently used
first (`last_use_epoch`: the text encoder before the diffusion model), the other runners' retained caches (the prefix
cache) last; without the variable the order is upstream's (caches first, then weights) and nothing is kept, so sd-cli
and the official behaviour are unchanged. A LoRA change (`set_loras`) frees the kept blocks: they carry the old LoRA.
The graph planner (`GGMLRunner::compute`, `fits_monolithic`) asks the same eviction for the whole graph before it
decides to cut the graph into segments: measured without that, the diffusion model's first step on the 8 GB
emulation with an example picture ran in 34 segments (its weights streamed per layer, 29 s instead of 11) because
the idle text encoder still counted as resident in the pure fit check.
Measured through the hub on the RTX 3060 12 GB with the Turbo (text only, 8 steps): after the first picture no
"staged compute params" line appears any more; the decode (direct convolutions, so that it fits beside the 4 GB kept)
evicts the text encoder's blocks one by one until it fits (3.4 of 4.4 GB that day, one block stayed); a re-roll
12.3 -> 11.8 s with the old direct conv kernel (1.8 s against the im2col decode's 1.7). An 8 GB card's decode would
evict the diffusion weights every picture, so the hub's launcher asks for it only where the budget holds them beside
the decode (`--keep-params auto`: 12 GB cards).
