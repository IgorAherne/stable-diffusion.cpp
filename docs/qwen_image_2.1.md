# How to Use

Qwen Image 2.1 supports text-to-image generation and image editing, using Qwen3-VL-8B as the text encoder and its own VAE.

## Download weights

- Download Qwen Image 2.1
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae
- Download Qwen3-VL-8B-Instruct
    - safetensors (BF16 or INT8 convrot): https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders
    - gguf: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
    - For image editing with a GGUF text encoder, also download `mmproj-Qwen3VL-8B-Instruct-F16.gguf` from the same repository and pass it with `--llm_vision`.

Use `qwen_image_2.1_vae_bf16.safetensors` with this model. The earlier Qwen Image and Wan 2.2 VAE weights are not interchangeable with the Qwen Image 2.1 VAE weights.

## Examples

Run the following commands from the build directory. Use image dimensions divisible by 32. The resolution-dependent flow schedule is selected automatically.

### Text to image

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors --vae ..\models\vae\qwen_image_2.1_vae_bf16.safetensors --llm ..\models\text_encoders\Qwen3VL-8B-Instruct-Q4_K_M.gguf -p "a lovely cat holding a sign says 'qwen2.1.cpp'" --cfg-scale 6.0 --sampling-method euler -v --offload-to-cpu --fa -o qwen_image_2.1.png
```

<img alt="Qwen Image 2.1 example" src="../assets/qwen/qwen_image_2.1.png" />

To use GGUF diffusion weights, set `--diffusion-model` to the path of a file such as `qwen_image_2.1-Q4_K.gguf`.

### Image editing

Pass the reference image with `-r` and describe the edit in `-p`. Vision weights are required; the example below loads them separately with `--llm_vision`.

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors --vae ..\models\vae\qwen_image_2.1_vae_bf16.safetensors --llm ..\models\text_encoders\Qwen3VL-8B-Instruct-Q4_K_M.gguf --llm_vision ..\models\text_encoders\Qwen3VL-8B-Instruct-mmproj-BF16.gguf -r ..\assets\qwen\qwen_image_2.1.png -p "change 'qwen2.1.cpp' to 'sd.cpp'" --cfg-scale 6.0 --sampling-method euler -v --offload-to-cpu --fa -o qwen_image_2.1_edit.png
```

For multiple reference images, repeat `-r` in the desired order, for example `-r first.png -r second.png`.

### Prefix cache

By default, the first denoising call for each fixed condition saves the text and reference-image keys and values from every transformer layer. Later calls only compute the target-image tokens. Positive and negative conditions use separate caches.

`qwen_image_2_1_prefix_cache_keep` in `--model-args` sets whether a cache outlives its sampling run, so that the next image with the same condition (another seed in a new request, or the next image of `--batch-count`) skips the prefix pass in its first step too. Caches are matched by the exact values of the conditioning, the reference latents, the token layout, the target latent's shape and the attention settings. A kept cache stays in device memory between images and is released when the VAE needs that memory, when a new prompt or new reference images are encoded, when a later run does not use it, and when models or LoRAs change.

- `auto` (default): keep the caches only with Flash Attention and without Sage Attention. Flash Attention converts the keys to FP16 in both the cache-writing and the cache-reading graphs, so a kept cache gives bit-identical images.
- `on`: keep the caches with every attention. With Sage Attention, the cache-writing graph also passes the generated image's keys in FP16, like the cache-reading graphs do, because Sage Attention quantizes keys from the type it receives. This keeps an image identical whether its cache was kept or rebuilt, but images differ from `auto` and `off` by rounding.
- `off`: release the caches when sampling ends.

With Qwen Image 2.1, the latents of the reference images are also kept for the next request with the same reference images (unless VAE tiling or circular padding is enabled).

Set `qwen_image_2_1_prefix_cache_type` in `--model-args` to `auto` or a type name using the same parser and case-sensitive names as `--type`:

- `auto` (default): use FP16 only when Flash Attention is enabled, Sage Attention is disabled, the attention scale is unchanged, and every attention operation in the cache-writing or cache-reading graph selects Flash Attention after backend support checks. If an operation falls back, rebuild the prefix in FP32 before executing and keep FP32 for the rest of that sampling run.
- `f32`: always store FP32 keys and values.
- `f16`: always store FP16 keys and values, including with ordinary attention or custom attention scaling. This saves cache memory but can introduce additional rounding error.
- Other types, such as `bf16`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_K`, `q6_K`, `iq4_nl`, and `iq4_xs`: use the requested storage type if the ggml build provides runtime conversion to and from FP32. Quantization is lossy and must be selected explicitly; `auto` never selects a quantized type.

Cache data is packed into contiguous rows of `hidden_size` elements before conversion, so 256-element quantization blocks work with the model's 128-element attention heads without padding. The type's block size must divide `hidden_size`. Unknown types, types lacking runtime conversion (for example `q8_1` and several IQ formats), and incompatible block sizes are ignored with a warning, leaving the previous setting or the default `auto` unchanged.

For example, use `--model-args qwen_image_2_1_prefix_cache_type=q8_0` to enable 8-bit cache storage. Cached keys and values are converted back to the attention input type before concatenating with the current target tokens. This reduces persistent cache memory; attention working buffers still use floating-point values, and conversion adds work on each step. Backends without the required conversion operations use the existing CPU fallback.

For the default 32-layer model, a prefix of 4096 tokens takes approximately the following memory per condition, excluding weights, working buffers, and allocation overhead:

| Cache type | Memory |
| --- | ---: |
| `f32` | 4 GiB |
| `f16` | 2 GiB |
| `q8_0` | 1.0625 GiB |
| `q4_0` | 0.5625 GiB |

The runner accounts for the cache when checking the memory budget. If a cached execution runs out of memory, it releases the prefix caches, disables caching for the rest of that sampling run, and retries the full sequence once. Per-step conditioning extensions currently use the full-sequence path.

Disable this optimization with `--model-args qwen_image_2_1_prefix_cache=false`. It reuses step-independent activations; numerical results can still differ slightly because the matrix sizes change.

### First-block cache

`qwen_image_2_1_block_cache=<threshold>[:<first full steps>:<last full steps>:<max skips in a row>[:<tail blocks>[:<head blocks>]]]` in `--model-args` (or the environment variable `SD_QWEN_BLOCK_CACHE` with the same value; the model arg wins) skips transformer blocks 1..31 on steps where the input barely changed (the "First-Block Cache"). Each step computes the embeddings and block 0 first and compares block 0's residual over the image tokens with the previous step's: `sum|r - r_prev| / sum|r_prev|`. While the sum of these changes since the last fully computed step stays below the threshold, the step reuses the residual of blocks 1..31 from the last fully computed step instead of computing it. The step count and the sampler are unchanged; a skipped step advances the sampler with the approximated output. Defaults: 2 first and 1 last steps always computed, at most 3 skipped steps in a row; `0` or no value turns the cache off. The fields are separated by `:` because `,` separates the model args.

Two optional fields move the cached part (both default to today's behaviour): `<head blocks>` H (default 1) computes blocks 0..H-1 on every step before the decision, and `<tail blocks>` T (default 0) recomputes the last T blocks on a skipped step on top of the cached residual, which then covers blocks H..31-T. The environment variables `SD_QWEN_BLOCK_CACHE_HEAD` and `SD_QWEN_BLOCK_CACHE_TAIL` override these fields. For experiments, `SD_QWEN_BLOCK_CACHE_STEPS=9,10,12` skips exactly the listed steps (1-based, as the log prints them) instead of deciding by the threshold; it turns the cache on by itself, and a step that fills the prefix cache or has no cached residual yet is computed fully.

Without the cache, each step is the same single graph as before, so images are unchanged. With it, images change: the deviation grows with the number of skipped steps, and skipping early steps changes the composition more than skipping later ones (raise the first full steps for more skipped steps at the same quality). The state and the cached tensors (three F32 tensors of `hidden_size` x image tokens per condition, 64 MiB each at 1024x1024) are per sampling run and per condition (with CFG, the positive and negative conditions decide separately) and are released when sampling ends. The log reports the skipped steps and the per-step change once per run (`-v` prints each decision).

### Alpha channel

This model supports alpha channel output. As the model determines whether to output a regular image or with transparency through the prompt, according to [official recommendation](https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba), use the following prompt format for better results:

> `This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`

Since transparency is decided by the prompt rather than by the input or an explicit switch, the same format applies equally to editing, whether or not the reference image itself has an alpha channel. Note that alpha is kept only in `.png` and `.webp` outputs; saving as `.jpg` drops the transparency.

Here are some examples ran with Q6_K quantization:
| Input | Prompt | Output |
| --- | --- | --- |
| ![Qwen Image 2.1 alpha input example 1](../assets/qwen/qwen-image-2.1-alpha-in1.png) | This is an RGBA image with transparency. Replace the text "BLOOM" with "Qwen Image 2.1", keeping the same font of the original text. The image has alpha channel and the background is transparent. | ![Qwen Image 2.1 alpha output example 1](../assets/qwen/qwen-image-2.1-alpha-out1.png) |
| ![Qwen Image 2.1 alpha input example 2](../assets/logo.png) | This is an RGBA image with transparency. Remove the background of the image, keeping only the text and cat. The image has alpha channel and the background is transparent. | ![Qwen Image 2.1 alpha output example 2](../assets/qwen/qwen-image-2.1-alpha-out2.png) |

### Other features

Other features of the model could be found on the [model card from QwenLM/Qwen-Image-2.1 repo](https://github.com/QwenLM/Qwen-Image-2.1), including 2 finetuned prompt rewriting Qwen3.5-9B model.
