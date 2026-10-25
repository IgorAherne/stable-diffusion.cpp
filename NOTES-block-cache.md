# First-Block Cache for Qwen-Image 2.1 (branch spellcast-block-cache)

The cache cuts the cost of a Qwen-Image 2.1 picture without changing the step count or the sampler. On some steps the
transformer's middle blocks are not computed; their residual from the last fully computed step is reused instead.
Block 0 is always computed fresh, and the last quarter of the blocks (the tail) is recomputed on top of the cached
residual. It belongs to the FBCache family of para-attn / WaveSpeed, with two changes found by measurement: a fixed
schedule instead of a threshold on block 0's change, and the tail refresh. Code: `src/model/diffusion/qwen_image_2_1.hpp`
(the comment block above `BlockCacheConfig`). User docs: `docs/qwen_image_2.1.md`, "First-Block Cache".

## Product form

- Model arg: `qwen_image_2_1_first_block_cache=<fraction>[:<tail blocks>]`.
- Environment variable: `SD_QWEN_FIRST_BLOCK_CACHE=<fraction>[:<tail blocks>]`. It is read at the start of every
  sampling run and overrides the model arg for that run; `0` turns the cache off for that run.
- `fraction` is the share of the N sampling steps that reuse the cache. `0` or no value means off: each step is then
  today's single graph, pixel-identical.
- `tail` is the number of last blocks recomputed on a cached step. The default is round(num_layers / 4) = 8. `0` is
  allowed but is research only: the pictures come out softer.
- Fields are separated by `:` because `,` separates `--model-args`; the variable also accepts `,`.

### The schedule

The schedule is fixed and depends only on N and the fraction (`first_block_cache_schedule`):
- The first W = max(2, round(0.32 N)) steps and the last step are always computed fully.
- K = round(fraction x N) of the slots W+1 .. N-1 reuse the cache, capped at the number of slots.
- When the cached steps are at most one more than the computed slots, they are placed one at a time, evenly.
- Otherwise they go in runs separated by single computed steps. The runs are as even as possible, so at most 3 long
  while K <= 0.75 x the slots. The longer runs go nearest the middle of the slots, which is the flattest part of block
  0's change curve; on a tie the later run gets it.
- The INFO line prints the schedule once per run: `first-block cache: K of N steps reuse the cached blocks 1..31-T
  (tail T recomputed): <list>`.

| N | fraction | W | K | cached steps |
|---|---|---|---|---|
| 25 | 0.5 | 8 | 13 | 9,10,11,13,14,15,17,18,19,20,22,23,24 |
| 20 | 0.5 | 6 | 10 | 7,8,10,11,12,14,15,16,18,19 |
| 30 | 0.5 | 10 | 15 | 11,12,13,15,16,17,19,20,21,23,24,25,27,28,29 |
| 25 | 0.4 | 8 | 10 | 9,11,13,14,16,17,19,20,22,24 |
| 25 | 0.3 | 8 | 8 | 10,12,14,16,18,20,22,24 |
| 25 | 0.2 | 8 | 5 | 10,13,17,20,23 |

A Python port of the function was checked for N = 1..80 and fractions 0..1 in steps of 0.01. In every case it gave K
distinct steps inside the slots, sorted, and runs of at most 3 whenever K <= 0.75 x the slots.

For N = 25 at 0.5 this schedule differs by one step from the 13-skip list measured on the GTX 1070
(9,10,11,13,14,15,17,18,19,21,22,23,24). That list has its run of four at the end (21-24); the rule puts it in the
middle (17-20).

### The graphs

There are two graphs per step instead of one. H is the head (1 in the product form) and T is the tail.
- **A**: the prelude and blocks 0..H-1.
  - It persists `hf`, the output of block H-1, over the whole sequence this step's joint holds.
  - It persists r1 = h1 - h0, block 0's residual over the drawn picture's tokens.
  - It returns the indicator's two sums, reduced on the card: sum|r1 - r1_prev| and sum|r1_prev|. They cost about
    1 ms and are only logged in the product form (VERBOSE line per step).
- **B**, a fully computed step: blocks H..31 on `hf`, then the head.
  - It persists D, the residual of blocks H..31-T.
  - With T = 0, D covers the drawn picture's tokens.
  - With T > 0, D covers the whole sequence the step holds: in REUSE that is the picture's tokens only; in the STORE or
    uncached step the prompt and reference-image tokens come too. Their share of D is the same on every step, because
    they never attend to the drawn picture and take the zero timestep.
  - A + B compute exactly what the single graph does (pixel-identical, see below).
- **C**, a cached step: blocks 32-T..31 on hf + D, then the head (T = 0: the head on hf + D).
  - In a REUSE step, a D made by the STORE step contributes its last tokens (the picture's).

Keys and memory:
- State and tensors are kept per condition, by the engine's prefix id. With CFG, the positive and the negative
  condition keep their own; both follow the same schedule.
- Everything is reset at the start of every sampling run (a re-roll, each picture of a batch).
- The tensors (`hf`, r1, D; F32, 64 MiB each at 1024x1024) are runner cache entries that are not retained. They stay
  on the card from step to step and go at the run's end.
- `hf` is dropped after the step's second graph, and the old D before graph B. At most four of these tensors exist at
  once (during A: hf, r1, r1_prev, D).
- With the prefix cache, the STORE step (the first) is always computed fully. If memory runs out, the cache is turned
  off for the run, together with the prefix cache, and the step is retried as one graph.

### Research knobs

These are kept as they were. They are read at load and apply only to runs without the product form:
- `qwen_image_2_1_block_cache=<threshold>[:<first full>:<last full>:<max skips>[:<tail>[:<head>]]]` and
  `SD_QWEN_BLOCK_CACHE` (the same value): the threshold policy on block 0's accumulated change.
- `SD_QWEN_BLOCK_CACHE_STEPS=9,10,12`: skip exactly these steps.
- `SD_QWEN_BLOCK_CACHE_TAIL` and `SD_QWEN_BLOCK_CACHE_HEAD`: override the matching fields.

## Measurements

### GTX 1070, from the pod

Three prompts (chest, potion, knight), 25 steps. The three ratios are the picture's value divided by the cache-off
picture's.

| variant | Laplacian variance | high-frequency energy | luma std | PSNR | sampling |
|---|---|---|---|---|---|
| 12 steps skipped, no tail | 0.84 | 0.93 | 0.97 | 40-43 dB | 75-81 s |
| the same 12, tail 4 | 0.91 | 0.96 | 0.98 | 40-43 dB | 83.5 s |
| the same 12, tail 6 | 0.94 | 0.96 | 0.99 | | 87.7 s |
| the same 12, tail 8 | 0.995 | 0.99 | 1.00 | 40-43 dB | 91.7 s |
| tail 8, 13 skips (9,10,11,13,14,15,17,18,19,21,22,23,24) | 0.99 | 0.99 | 1.00 | 38-41 dB | 85.5-87.6 s |
| tail 8, 14 skips, warm-up 6 (7,8,10,11,12,14,15,16,18,19,20,22,23,24) | 0.99 | 0.99 | 1.00 | 36.5-40 dB | 84 s |
| head 16 (half-computed steps) | 0.66 sharpness | | | | |

Mean brightness was unchanged in every variant. The fine detail, contrast and gloss are made in the last blocks. A
tail refresh of the last quarter restores them; computing more blocks at the head does not.

### RTX 3060, threshold policy, no tail (the first design)

Six-prompt suite: `code/devtools/qwen-speed/suite.py`, flash attention, 1024x1024, 25 steps, CFG 1. The reference is
the cache off.

"Visible" means RGB premultiplied by alpha. The raw RGB metrics also count the colour under the transparent background
(alpha 0), which nobody sees; the knight's raw numbers are dominated by it. "px >1 %" is the share of pixels in which
some channel moved by more than 2.55 levels.

| setting (threshold[:first full:last full:max skips]) | skipped steps (chest, knight, potion, fox, knight_back, ork_right) | steps 2-25: s (speed-up) | visible PSNR dB mean (min) | visible LPIPS mean (max) | visible mean dev % mean (max) | visible px >1 % mean (max) | raw RGB PSNR mean (min) | raw RGB mean dev 0-255 / % | alpha mean dev % | raw px >1 % rgb / alpha |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | 1,1,1,1,2,1 | 37.6 (1.05x) | 54.5 (50.5) | 0.0003 (0.0007) | 0.05 (0.08) | 0.8 (2.2) | 50.2 (46.1) | 0.31 / 0.12 | 0.02 | 2.8 / 0.4 |
| 0.06 | 5,5,5,5,5,5 | 31.5 (1.25x) | 47.1 (41.8) | 0.0014 (0.0022) | 0.11 (0.17) | 3.7 (6.1) | 44.4 (41.2) | 0.61 / 0.24 | 0.05 | 8.5 / 1.0 |
| 0.08 | 8,7,7,8,8,8 | 27.2 (1.45x) | 44.0 (37.7) | 0.0032 (0.0063) | 0.18 (0.28) | 6.4 (11.3) | 41.2 (37.0) | 0.93 / 0.37 | 0.08 | 14.4 / 1.5 |
| 0.10 | 12,12,12,12,13,12 | 20.1 (1.96x) | 32.7 (25.9) | 0.0230 (0.0464) | 0.65 (1.19) | 18.1 (23.6) | 30.8 (25.0) | 2.52 / 0.99 | 0.34 | 29.0 / 2.4 |
| 0.12 | 13,13,13,13,13,13 | 18.8 (2.09x) | 30.6 (18.6) | 0.0438 (0.1450) | 1.19 (3.86) | 22.6 (32.4) | 27.8 (18.0) | 4.82 / 1.89 | 0.58 | 39.7 / 3.3 |
| 0.16 | 14,14,14,14,15,14 | 17.3 (2.28x) | 30.4 (18.7) | 0.0458 (0.1450) | 1.22 (3.85) | 23.1 (32.2) | 27.5 (18.1) | 5.25 / 2.06 | 0.63 | 43.1 / 4.4 |
| 0.20 | 16,16,16,16,16,16 | 14.2 (2.77x) | 28.3 (19.2) | 0.0566 (0.1394) | 1.44 (3.65) | 24.9 (34.3) | 25.2 (18.5) | 7.07 / 2.77 | 0.77 | 48.1 / 5.5 |
| 0.32 | 17,17,17,17,17,17 | 12.7 (3.11x) | 26.6 (16.6) | 0.0768 (0.1981) | 1.92 (5.30) | 26.0 (34.9) | 23.6 (16.3) | 9.32 / 3.66 | 1.31 | 51.8 / 7.6 |
| 0.12:8:1:3 | 10,10,10,10,10,10 | 23.7 (1.66x) | 39.9 (33.4) | 0.0069 (0.0100) | 0.28 (0.45) | 10.5 (15.6) | 36.1 (26.9) | 2.25 / 0.88 | 0.13 | 27.1 / 2.2 |
| 0.16:8:1:3 | 12,11,11,11,12,11 | 21.4 (1.84x) | 38.0 (32.2) | 0.0105 (0.0167) | 0.36 (0.54) | 13.8 (18.5) | 34.1 (25.3) | 2.90 / 1.14 | 0.19 | 34.0 / 3.1 |
| 0.24:8:1:3 | 12,12,12,12,12,12 | 20.4 (1.94x) | 36.5 (29.7) | 0.0146 (0.0215) | 0.45 (0.75) | 16.2 (22.4) | 31.7 (20.3) | 4.76 / 1.87 | 0.24 | 39.1 / 3.7 |
| 0.12:6:1:3 | 11,11,11,11,11,11 | 22.0 (1.79x) | 38.4 (32.3) | 0.0083 (0.0124) | 0.32 (0.52) | 12.0 (17.6) | 34.6 (26.6) | 2.41 / 0.94 | 0.16 | 28.6 / 2.4 |
| 0.20:6:1:3 | 13,13,13,13,13,13 | 18.9 (2.09x) | 35.6 (29.0) | 0.0162 (0.0253) | 0.50 (0.84) | 17.9 (24.9) | 31.1 (20.4) | 4.81 / 1.89 | 0.27 | 39.9 / 4.0 |
| 0.30:10:1:3 | 11,11,11,11,11,11 | 22.0 (1.79x) | 38.2 (32.7) | 0.0109 (0.0164) | 0.36 (0.52) | 13.9 (18.1) | 32.7 (18.1) | 5.59 / 2.19 | 0.18 | 38.5 / 2.9 |
| control: SageAttention vs flash, cache off | 0,0,0,0,0,0 | 27.9 (1.41x) | 37.8 (30.6) | 0.0087 (0.0174) | 0.31 (0.57) | 8.2 (13.8) | 35.2 (30.1) | 1.18 / 0.46 | 0.17 | 12.3 / 1.1 |

Reference (cache off): steps 2-25 take 39.4 s on average (35.2-35.7 s text-only, 47.5-47.7 s with an example picture).

Findings:
- **Early skips move the composition.** With only 2 warm-up steps, thresholds of 0.10 and above skip steps 3-9, and the
  composition moves; the fox, a bistable seed, changes most.
- **A warm-up of 6-8 fixes that.** With it, 10-12 skips stay in the class of the SageAttention switch the user had
  already accepted (the control row).
- **The indicator is nearly a schedule.** Block 0's change per step was nearly the same for all six prompts (within
  about 10 %). It is U-shaped: 0.16-0.17 at step 2, 0.09-0.10 at steps 3-9, 0.037-0.041 at steps 15-18, 0.13 at step
  25. So the threshold policy produced almost the same skips on every prompt. This is why the product form uses a
  fixed schedule.

### Cost of a skipped step (RTX 3060, chest 1024x1024, steady REUSE steps)

| step | graph A | graph B or C | total |
|---|---|---|---|
| computed fully (A + B) | 54 ms | 1,393-1,397 ms | 1,447-1,451 ms, the same as the single graph |
| cached, tail 0 | 54 ms | 3 ms | 57 ms |
| cached, tail 4 | 54-55 ms | 179-180 ms | ~235 ms |
| cached, tail 8 (product default) | 53-54 ms | 363-366 ms | ~418 ms (29 % of a full step) |
| cached, head 16 | 726-727 ms | 3 ms | ~730 ms |

Compute buffer: 773-783 MB for graphs A and B, against 709-742 MB for the single graph. Graph A keeps h0 alive through
block 0, and with a tail D stays alive through the tail (805 MB).

### The product form on the RTX 3060 (chest, seed 4242, flash attention, 1024x1024)

- Cache off, 2 and 25 steps: pixel-identical to the unchanged engine (a43013e).
- `SD_QWEN_FIRST_BLOCK_CACHE=0.5`, 25 steps:
  - The INFO line read `first-block cache: 13 of 25 steps reuse the cached blocks 1..23 (tail 8 recomputed):
    9,10,11,13,14,15,17,18,19,20,22,23,24`, and the per-step log skipped exactly those steps.
  - Sampling took 25.0 s against 37.6 s with the cache off. Steps 2-25 took 21.6 s against 34.5 s (1.60x).
  - A cached step costs 54 + 364 ms; a full step 54 + 1,397-1,401 ms.
  - Against the cache-off picture: visible PSNR 41.2 dB, visible mean deviation 0.29 %, raw RGB PSNR 39.8 dB.
  - Sampling-phase VRAM (card total minus the idle desktop): 5,241 MiB, against 4,993 MiB for the cache-off chest
    measured in an earlier session (+248 MiB, the same as without a tail). The compute buffer peaked at 805 MB, against
    783 MB without a tail.

### Verified on the RTX 3060

Pixel-identical means equal decoded pixels; the PNG files differ only in the embedded version text.
- Cache off is pixel-identical to the unchanged engine (a43013e):
  - chest at 2 and 25 steps;
  - knight_back with an example picture, seed 5;
  - with the prefix cache off.
- A + B with nothing skipped is pixel-identical to the single graph (25 steps).
- `TAIL=31` and `HEAD=32` with a skipped step (the cached range is empty, so D = 0) are pixel-identical to the cache
  off.
- The step-list skip and the threshold skip of the same step are pixel-identical. The 15b9528 and 63dbe11 code paths
  give the same pixels.
- Batch `-b 2`: picture 2 is pixel-identical to its seed drawn alone, so the state is reset per run.
- CFG 4 with a negative prompt runs, and the two conditions decide separately.
- SageAttention with `prefix_cache_keep=on` runs.
- VRAM, chest, sampling phase, card total: 6,228 MiB with the cache off, 6,476 MiB with the threshold cache on, tail 0
  (+248 MiB).

## Not verified

- A gcc/Linux build. The code is plain C++17, but it was compiled with MSVC 2022 only.
- The GTX 1070 numbers above are the pod's, not this machine's.
- Segmented (8 GB) execution with the cache on; two-evaluation samplers (Heun, DPM2), where the step number repeats
  within a step; img2img; LoRA.

## Traps

- `,` separates `--model-args`, so the cache's fields use `:`.
- The suite's "steady s/it" ignores cached steps: they print as it/s and its regex reads s/it only. Compare "sampling
  completed" or, better, steps 2-25.
- Step 1 includes uploading the weights from the memory-mapped file: 10-50 s on this PC depending on the page cache.
  Compare steps 2-25.
- On Windows, a `:` in an output file name creates an NTFS stream, not a file.
- The hub sets `SD_QWEN_FIRST_BLOCK_CACHE` in-process before each picture. The engine reads it with `getenv` at the
  run's start, which sees Python's `os.environ` changes only when both use the same C runtime (the UCRT, as built with
  /MD). Check this in the hub.
- ggml's CMake makes the `ggml` target depend on its backends (`add_dependencies(ggml ggml-cuda)`), so building
  `stable-diffusion` rebuilds `ggml-cuda` whenever ggml's working tree changed.
