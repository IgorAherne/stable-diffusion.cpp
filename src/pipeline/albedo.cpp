// The albedo add-on for Qwen Image 2.1 (sd_albedo): Marigold V2's recipe retrained on this model. A picture's latent
// (the VAE encoder's mean, normalised) goes through one transformer pass at a fixed timestep with the add-on's LoRA
// and prompt embedding, without noise; latent - prediction, decoded by the add-on's own VAE decoder, is the linear
// albedo as 2 * albedo - 1.

#include "generation.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "core/backend_fit.h"
#include "diffusion_engine.h"
#include "model/adapter/lora.hpp"
#include "model/diffusion/model.hpp"
#include "model/vae/vae.hpp"
#include "model_loader.h"
#include "name_conversion.h"

struct AlbedoAddon {
    std::shared_ptr<ModelManager> manager;

    std::string lora_path;
    std::shared_ptr<LoraModel> lora;
    std::shared_ptr<MultiLoraAdapter> adapter;
    size_t lora_bytes = 0;

    std::string condition_path;
    sd::Tensor<float> context;  // [context_dim, L, 1]

    std::string decoder_path;
    std::shared_ptr<VAE> decoder;
    std::vector<ggml_tensor*> decoder_params;
    size_t decoder_bytes = 0;

    void release_lora() {
        adapter.reset();
        lora.reset();
        lora_path.clear();
        lora_bytes = 0;
    }

    void release_decoder() {
        if (decoder != nullptr) {
            decoder->runner_end();
            // (a decoder tensor left registered would keep pointing into the freed runner, as ~LoraModel)
            GGML_ASSERT(manager->unregister_param_tensors(decoder_params));
        }
        decoder.reset();
        decoder_params.clear();
        decoder_path.clear();
        decoder_bytes = 0;
    }

    ~AlbedoAddon() {
        release_decoder();
        release_lora();
    }
};

namespace sd::pipeline {

    namespace {

        constexpr double MiB = 1024.0 * 1024.0;

        // An add-on file is no runner's source: refresh_model_sources must not take its catalog revision for a model
        // change, which would rebuild the runners and drop what they keep for the next picture.
        bool add_addon_file(StableDiffusionGGML* sd, const std::string& path, const char* prefix, ModelLoader::FileId* id) {
            const bool in_sync = sd->runner_state_.catalog_revision == sd->model_manager->loader().revision();
            if (!sd->model_manager->add_isolated_file(path, prefix, id)) {
                LOG_ERROR("albedo: cannot read '%s'", path.c_str());
                return false;
            }
            if (in_sync) {
                sd->runner_state_.catalog_revision = sd->model_manager->loader().revision();
            }
            return true;
        }

        // The down/up pairs whose model weight exists with matching shapes (a fused weight such as [gate | up] takes
        // one pair per part: ".weight", ".weight.1"); *all_fit: every tensor of the LoRA belongs to such a pair.
        size_t check_lora_pairs(StableDiffusionGGML* sd, LoraModel& lora, bool* all_fit) {
            struct Part {
                ggml_tensor* down = nullptr;
                ggml_tensor* up   = nullptr;
            };
            std::map<std::string, std::map<int, Part>> weights;  // model weight -> part index -> pair
            std::vector<std::string> problems;
            for (const auto& [name, tensor] : lora.lora_tensors) {
                std::string key = starts_with(name, "lora.") ? name.substr(5) : std::string();
                bool down       = false;
                bool up         = false;
                if (ends_with(key, ".lora_down")) {
                    down = true;
                    key.resize(key.size() - strlen(".lora_down"));
                } else if (ends_with(key, ".lora_up")) {
                    up = true;
                    key.resize(key.size() - strlen(".lora_up"));
                } else if (ends_with(key, ".alpha")) {
                    key.resize(key.size() - strlen(".alpha"));
                } else {
                    problems.push_back(name + " (not a LoRA pair's tensor)");
                    continue;
                }
                int part        = 0;
                const auto dot  = key.find_last_of('.');
                const auto tail = dot == std::string::npos ? std::string() : key.substr(dot + 1);
                if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) {
                    part = std::atoi(tail.c_str());
                    key.resize(dot);
                }
                auto& entry = weights[key][part];
                if (down) {
                    entry.down = tensor;
                } else if (up) {
                    entry.up = tensor;
                }
            }
            std::map<std::string, ggml_tensor*> model_tensors;
            sd->diffusion_model->get_param_tensors(model_tensors);
            size_t pairs = 0;
            for (const auto& [key, parts] : weights) {
                auto weight = model_tensors.find(key);
                if (weight == model_tensors.end()) {
                    problems.push_back(key + " (no such weight in the diffusion model)");
                    continue;
                }
                int64_t out_rows = 0;
                int expected     = 0;
                bool ok          = true;
                for (const auto& [part, pair] : parts) {
                    ok = ok && part == expected++ && pair.down != nullptr && pair.up != nullptr &&
                         pair.down->ne[0] == weight->second->ne[0] && pair.up->ne[0] == pair.down->ne[1];
                    if (ok) {
                        out_rows += pair.up->ne[1];
                    }
                }
                if (!ok || out_rows != weight->second->ne[1]) {
                    problems.push_back(key + " (the pair's shapes do not fit the weight)");
                    continue;
                }
                pairs += parts.size();
            }
            for (size_t i = 0; i < problems.size() && i < 8; ++i) {
                LOG_ERROR("albedo LoRA: %s", problems[i].c_str());
            }
            *all_fit = problems.empty();
            return pairs;
        }

        bool load_lora(StableDiffusionGGML* sd, AlbedoAddon& addon, const std::string& path, float multiplier) {
            if (addon.lora != nullptr && addon.lora_path == path) {
                addon.lora->multiplier = multiplier;
                return true;
            }
            addon.release_lora();
            int64_t t0 = ggml_time_ms();
            ModelManager::LoraSpec spec;
            spec.path       = path;
            spec.multiplier = multiplier;
            spec.required   = true;
            if (!add_addon_file(sd, path, "lora.", &spec.file_id)) {
                return false;
            }
            spec.file_revision = sd->model_manager->loader().file_revision(spec.file_id);
            auto lora          = sd->load_lora_model(spec, SDBackendModule::DIFFUSION, [](const std::string& name) {
                return is_diffusion_model_name(name);
            });
            if (lora == nullptr || lora->lora_tensors.empty()) {
                LOG_ERROR("albedo: '%s' holds no LoRA for the diffusion model", path.c_str());
                return false;
            }
            lora->preprocess_lora_tensors(sd->model_manager->tensor_names());
            size_t total = 0;
            for (const auto& [name, tensor] : lora->lora_tensors) {
                total += ends_with(name, ".lora_down") ? 1 : 0;
            }
            bool all_fit         = false;
            const size_t matched = check_lora_pairs(sd, *lora, &all_fit);
            if (!all_fit || matched == 0 || matched != total) {
                LOG_ERROR("albedo LoRA '%s': %zu of %zu pairs match the diffusion model's weights", path.c_str(), matched, total);
                return false;
            }
            size_t bytes = 0;
            for (const auto& [name, tensor] : lora->lora_tensors) {
                bytes += ggml_nbytes(tensor);
            }
            LOG_INFO("albedo LoRA: %zu of %zu pairs match the diffusion model's weights (%.1f MB, applied at runtime), loaded in %.2fs",
                     matched, total, bytes / MiB, (ggml_time_ms() - t0) / 1000.0);
            addon.lora       = lora;
            addon.adapter    = std::make_shared<MultiLoraAdapter>(std::vector<std::shared_ptr<LoraModel>>{lora});
            addon.lora_path  = path;
            addon.lora_bytes = bytes;
            return true;
        }

        bool load_condition(StableDiffusionGGML* sd, AlbedoAddon& addon, const std::string& path) {
            if (!addon.context.empty() && addon.condition_path == path) {
                return true;
            }
            addon.context = {};
            addon.condition_path.clear();
            ModelLoader loader;
            if (!loader.init_from_file(path)) {
                LOG_ERROR("albedo: cannot read the prompt embedding '%s'", path.c_str());
                return false;
            }
            const auto& tensors = loader.get_tensor_storage_map();
            auto found          = tensors.find("prompt_embeds");
            const int64_t width = 4096;  // Qwen Image 2.1's context_dim (Qwen3-VL-8B's hidden size)
            if (found == tensors.end() || found->second.ne[0] != width || found->second.ne[1] <= 0 ||
                found->second.nelements() != found->second.ne[0] * found->second.ne[1]) {
                LOG_ERROR("albedo: '%s' needs a tensor \"prompt_embeds\" [L, %" PRId64 "]", path.c_str(), width);
                return false;
            }
            const int64_t tokens = found->second.ne[1];
            std::vector<float> data;
            if (!loader.load_float_tensor("prompt_embeds", data, sd->n_threads)) {
                LOG_ERROR("albedo: cannot load \"prompt_embeds\" from '%s'", path.c_str());
                return false;
            }
            addon.context        = sd::Tensor<float>({width, tokens, 1}, std::move(data));
            addon.condition_path = path;
            LOG_INFO("albedo prompt: %" PRId64 " tokens from '%s'", tokens, path.c_str());
            return true;
        }

        bool load_decoder(StableDiffusionGGML* sd, AlbedoAddon& addon, const std::string& path) {
            if (addon.decoder != nullptr && addon.decoder_path == path) {
                return true;
            }
            addon.release_decoder();
            ModelLoader::FileId id = 0;
            if (!add_addon_file(sd, path, "vae.", &id)) {
                return false;
            }
            const auto tensors = sd->model_manager->loader().file_tensors(id, sd->version);
            sd::model_builders::Context ctx{sd->config_->params, sd->version, tensors, sd->backend_manager, sd->model_manager};
            auto decoder = sd::model_builders::build_vae_decoder(ctx);
            if (decoder == nullptr) {
                LOG_ERROR("albedo: no decode-only VAE for %s", model_version_to_str[sd->version]);
                return false;
            }
            decoder->set_max_graph_vram_bytes(sd->max_graph_vram_bytes_for_module(SDBackendModule::VAE));
            std::map<std::string, ggml_tensor*> params;
            decoder->get_param_tensors(params);
            std::vector<ggml_tensor*> registered;
            size_t bytes   = 0;
            size_t missing = 0;
            for (const auto& [name, tensor] : params) {
                if (tensors.find(name) == tensors.end() && missing++ < 4) {
                    LOG_ERROR("albedo decoder '%s' lacks %s", path.c_str(), name.c_str());
                }
                registered.push_back(tensor);
                bytes += ggml_nbytes(tensor);
            }
            if (missing != 0) {
                LOG_ERROR("albedo decoder '%s': %zu of %zu tensors missing", path.c_str(), missing, params.size());
                return false;
            }
            const auto mode = sd->backend_manager.params_backend_is_disk(SDBackendModule::VAE) ? ModelManager::ResidencyMode::Disk
                                                                                               : ModelManager::ResidencyMode::ParamBackend;
            if (!sd->model_manager->register_param_tensors(ModelComponent::AddonVAE, std::move(params), mode,
                                                           sd->backend_for(SDBackendModule::VAE),
                                                           sd->params_backend_for(SDBackendModule::VAE),
                                                           nullptr, false, false, nullptr, id, sd->version)) {
                LOG_ERROR("albedo: cannot register the decoder '%s'", path.c_str());
                return false;
            }
            addon.decoder        = decoder;
            addon.decoder_params = std::move(registered);
            addon.decoder_path   = path;
            addon.decoder_bytes  = bytes;
            if (!sd->model_manager->validate_registered_tensors()) {
                addon.release_decoder();
                return false;
            }
            LOG_INFO("albedo decoder: %zu tensors (%.1f MB) from '%s'", addon.decoder_params.size(), bytes / MiB, path.c_str());
            return true;
        }

        // While a VAE works, the residency manager may drop what the idle diffusion runner keeps for the next picture
        // where the VAE cannot fit next to it, as during a picture's own decode (see RetainedDiffusionCacheOffer).
        class DiffusionCacheOffer {
            std::shared_ptr<DiffusionModelRunner> runner_;

        public:
            explicit DiffusionCacheOffer(const std::shared_ptr<DiffusionModelRunner>& runner)
                : runner_(runner) {
                runner_->offer_retained_cache(true);
            }
            ~DiffusionCacheOffer() { runner_->offer_retained_cache(false); }
            DiffusionCacheOffer(const DiffusionCacheOffer&)            = delete;
            DiffusionCacheOffer& operator=(const DiffusionCacheOffer&) = delete;
        };

        // The request's circular padding and the diffusion model's weight adapter, put back when the call ends.
        class AlbedoScope {
            StableDiffusionGGML* sd_;
            bool circular_x_;
            bool circular_y_;
            std::shared_ptr<WeightAdapter> adapter_;

        public:
            explicit AlbedoScope(StableDiffusionGGML* sd)
                : sd_(sd), circular_x_(sd->circular_x), circular_y_(sd->circular_y), adapter_(sd->diffusion_model->get_weight_adapter()) {
                sd_->apply_circular_axes(false, false);
                sd_->first_stage_model->set_circular_axes(false, false);
            }
            // The transformer pass is over: its buffers go before the decode.
            void end_transformer() {
                sd_->diffusion_model->runner_end();
                sd_->diffusion_model->set_weight_adapter(adapter_);
            }
            ~AlbedoScope() {
                end_transformer();
                sd_->apply_circular_axes(circular_x_, circular_y_);
                sd_->first_stage_model->set_circular_axes(circular_x_, circular_y_);
            }
            AlbedoScope(const AlbedoScope&)            = delete;
            AlbedoScope& operator=(const AlbedoScope&) = delete;
        };

        sd::Tensor<float> encode(StableDiffusionGGML* sd, const sd::Tensor<float>& rgba) {
            DiffusionCacheOffer offer(sd->diffusion_model);
            sd_tiling_params_t tiling = {false, false, 0, 0, 0.5f, 0.f, 0.f, nullptr};
            auto vae                  = sd->first_stage_model;
            auto latents              = vae->encode(sd->n_threads, rgba, tiling);
            if (latents.empty() && vae->last_compute_status() == GGML_STATUS_ALLOC_FAILED) {
                LOG_WARN("albedo: the VAE encode did not fit in one piece; encoding in tiles");
                tiling.enabled = true;
                latents        = vae->encode(sd->n_threads, rgba, tiling);
            }
            if (latents.empty()) {
                return {};
            }
            return vae->vae_to_diffusion_latents(vae->vae_output_to_latents(latents, sd->rng));
        }

        sd::Tensor<float> decode(StableDiffusionGGML* sd, VAE& decoder, const sd::Tensor<float>& x) {
            DiffusionCacheOffer offer(sd->diffusion_model);
            auto latents              = decoder.diffusion_to_vae_latents(x);
            sd_tiling_params_t tiling = {false, false, 0, 0, 0.5f, 0.f, 0.f, nullptr};
            int tile_size_w           = static_cast<int>(latents.shape()[0]);
            int tile_size_h           = static_cast<int>(latents.shape()[1]);
            for (;;) {
                float overlap = 0.f;
                if (tiling.enabled && !decoder.get_tile_sizes(tile_size_w, tile_size_h, overlap, tiling, latents.shape()[0], latents.shape()[1])) {
                    return {};
                }
                auto decoded = decoder.decode(sd->n_threads, latents, tiling);
                if (!decoded.empty() ||
                    !sd::backend_fit::prepare_vae_decode_retry_tiling(tiling, false, decoder.last_compute_status(),
                                                                      tile_size_w, tile_size_h, decoder.get_scale_factor())) {
                    return decoded;
                }
            }
        }

    }  // namespace

    sd_image_t* albedo(StableDiffusionGGML* sd, const sd_albedo_params_t* params) {
        const int64_t t0 = ggml_time_ms();
        if (sd->version != VERSION_QWEN_IMAGE_2_1 || sd->diffusion_model == nullptr || sd->first_stage_model == nullptr ||
            (sd->uses_tae() && !sd->tae_preview_only())) {
            LOG_ERROR("albedo: needs a Qwen Image 2.1 context with its VAE (loaded: %s)", model_version_to_str[sd->version]);
            return nullptr;
        }
        const sd_image_t& image = params->image;
        if (image.data == nullptr || image.channel != 3 || image.width == 0 || image.height == 0 ||
            image.width % 32 != 0 || image.height % 32 != 0) {
            LOG_ERROR("albedo: needs an RGB picture with sides multiples of 32 (got %ux%u, %u channels)",
                      image.width, image.height, image.channel);
            return nullptr;
        }
        if (params->lora_path == nullptr || params->condition_path == nullptr || params->decoder_path == nullptr) {
            LOG_ERROR("albedo: the LoRA, prompt embedding and decoder paths are all needed");
            return nullptr;
        }
        if (!std::isfinite(params->timestep) || params->timestep < 0.f || params->timestep > 1.f ||
            !std::isfinite(params->lora_multiplier) || !(params->output_gamma > 0.f) || !std::isfinite(params->output_gamma)) {
            LOG_ERROR("albedo: invalid timestep %g, LoRA multiplier %g or output gamma %g",
                      params->timestep, params->lora_multiplier, params->output_gamma);
            return nullptr;
        }
        if (sd->apply_lora_immediately && !sd->conditioning_loras_.empty()) {
            LOG_WARN("albedo: the diffusion weights carry the last picture's merged LoRAs; the add-on was trained without them");
        }

        if (sd->albedo_addon_ == nullptr) {
            sd->albedo_addon_          = std::make_shared<AlbedoAddon>();
            sd->albedo_addon_->manager = sd->model_manager;
        }
        AlbedoAddon& addon = *sd->albedo_addon_;
        if (!load_lora(sd, addon, params->lora_path, params->lora_multiplier) ||
            !load_condition(sd, addon, params->condition_path) ||
            !load_decoder(sd, addon, params->decoder_path)) {
            return nullptr;
        }
        const int64_t t_load = ggml_time_ms();

        const int64_t W = image.width;
        const int64_t H = image.height;
        // The Qwen Image 2.1 VAE reads RGBA: the picture goes in opaque.
        sd::Tensor<float> rgba({W, H, 4, 1});
        float* px = rgba.data();
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                const uint8_t* src = image.data + (y * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    px[(c * H + y) * W + x] = src[c] / 255.f;
                }
                px[(3 * H + y) * W + x] = 1.f;
            }
        }

        AlbedoScope scope(sd);
        auto latent = encode(sd, rgba);
        if (latent.empty()) {
            LOG_ERROR("albedo: VAE encode failed");
            return nullptr;
        }
        const int64_t t_encode = ggml_time_ms();

        // The runner takes flow timesteps on the [0, 1000] scale (diffusers' timestep times its time_factor).
        auto timesteps = sd::Tensor<float>::from_vector({params->timestep * 1000.f});
        QwenImage21DiffusionExtra extra;
        extra.isolated = true;
        DiffusionParams diffusion_params;
        diffusion_params.x         = &latent;
        diffusion_params.timesteps = &timesteps;
        diffusion_params.context   = &addon.context;
        diffusion_params.extra     = extra;
        addon.lora->applied_lora_tensors.clear();
        addon.lora->skipped_incompatible_lora_tensors.clear();
        sd->diffusion_model->set_weight_adapter(addon.adapter);
        auto prediction = sd->diffusion_model->compute(sd->n_threads, diffusion_params);
        scope.end_transformer();
        if (prediction.empty() || prediction.shape() != latent.shape()) {
            LOG_ERROR("albedo: the transformer pass failed");
            return nullptr;
        }
        // Every LoRA tensor must have gone into the graph (a layer outside the adapter's paths would drop its pair).
        if (!addon.lora->skipped_incompatible_lora_tensors.empty() ||
            addon.lora->applied_lora_tensors.size() != addon.lora->lora_tensors.size()) {
            LOG_ERROR("albedo: the transformer used %zu of the LoRA's %zu tensors (%zu skipped)",
                      addon.lora->applied_lora_tensors.size(), addon.lora->lora_tensors.size(),
                      addon.lora->skipped_incompatible_lora_tensors.size());
            return nullptr;
        }
        const int64_t t_dit = ggml_time_ms();

        auto decoded = decode(sd, *addon.decoder, latent - prediction);
        if (decoded.empty() || decoded.shape()[0] != W || decoded.shape()[1] != H || decoded.shape()[2] < 3) {
            LOG_ERROR("albedo: decode failed");
            return nullptr;
        }
        const int64_t t_decode = ggml_time_ms();

        // decode gives (x + 1) / 2 clamped to [0, 1]: the linear albedo
        uint8_t* pixels = static_cast<uint8_t*>(malloc(static_cast<size_t>(W * H * 3)));
        auto* result    = static_cast<sd_image_t*>(malloc(sizeof(sd_image_t)));
        if (pixels == nullptr || result == nullptr) {
            free(pixels);
            free(result);
            return nullptr;
        }
        const float inverse_gamma = 1.f / params->output_gamma;
        const float* albedo       = decoded.data();
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const float a               = std::min(1.f, std::max(0.f, albedo[(c * H + y) * W + x]));
                    pixels[(y * W + x) * 3 + c] = static_cast<uint8_t>(std::pow(a, inverse_gamma) * 255.f + 0.5f);
                }
            }
        }
        *result = {static_cast<uint32_t>(W), static_cast<uint32_t>(H), 3, pixels};
        LOG_INFO("albedo %" PRId64 "x%" PRId64 " in %.2fs (add-on files %.2fs, encode %.2fs, transformer %.2fs, decode %.2fs); add-on weights LoRA %.1f MB + decoder %.1f MB",
                 W, H, (t_decode - t0) / 1000.0, (t_load - t0) / 1000.0, (t_encode - t_load) / 1000.0,
                 (t_dit - t_encode) / 1000.0, (t_decode - t_dit) / 1000.0, addon.lora_bytes / MiB, addon.decoder_bytes / MiB);
        return result;
    }

}  // namespace sd::pipeline
