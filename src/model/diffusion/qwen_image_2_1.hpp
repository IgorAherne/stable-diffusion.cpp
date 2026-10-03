#ifndef __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
#define __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__

#include <typeinfo>

#include "model/diffusion/qwen_image.hpp"

namespace Qwen {

    struct QwenImage21Config {
        int64_t in_channels       = 64;
        int64_t out_channels      = 64;
        int64_t hidden_size       = 4096;
        int64_t context_dim       = 4096;
        int64_t head_dim          = 128;
        int64_t intermediate_size = 12288;
        int num_layers            = 32;
        bool fused_mlp            = false;
        std::vector<int> axes_dim = {16, 56, 56};

        static QwenImage21Config detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            QwenImage21Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = weights.find(prefix + "." + suffix);
                return it == weights.end() ? nullptr : &it->second;
            };
            if (auto w = find("img_in.weight")) {
                config.in_channels = w->ne[0];
                config.hidden_size = w->ne[1];
            }
            if (auto w = find("proj_out.weight")) {
                config.out_channels = w->ne[1];
            }
            if (auto w = find("txt_in.in_layer.weight")) {
                config.context_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.attn.norm_q.weight")) {
                config.head_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.img_mlp.gate_up.weight")) {
                config.intermediate_size = w->ne[1] / 2;
                config.fused_mlp         = true;
            } else if (auto w = find("transformer_blocks.0.img_mlp.proj.weight")) {
                config.intermediate_size = w->ne[1];
            }
            int layers                     = 0;
            const std::string block_prefix = prefix + ".transformer_blocks.";
            for (const auto& [name, _] : weights) {
                if (starts_with(name, block_prefix)) {
                    layers = std::max(layers, atoi(name.substr(block_prefix.size()).c_str()) + 1);
                }
            }
            if (layers > 0) {
                config.num_layers = layers;
                LOG_VERBOSE("qwen_image_2_1: layers = %d, hidden_size = %" PRId64 ", context_dim = %" PRId64,
                            layers, config.hidden_size, config.context_dim);
            }
            return config;
        }
    };

    struct QwenImage21Segment {
        int64_t start;
        int64_t end;
        int64_t context_start;
        int image_index;
    };

    struct QwenImage21Layout {
        std::vector<QwenImage21Segment> segments;
        std::vector<std::vector<float>> positions;
        int64_t prefix_length = 0;
        Rope::PositionLayout rope_layout;

        static QwenImage21Layout build(int64_t text_length,
                                       const sd::Tensor<int32_t>& image_slots,
                                       const std::vector<std::pair<int64_t, int64_t>>& image_shapes) {
            if (image_shapes.empty() || (!image_slots.empty() && image_slots.numel() != text_length)) {
                throw std::runtime_error("Qwen Image 2.1: invalid image token layout");
            }
            QwenImage21Layout layout;
            int64_t position  = 0;
            int next_image    = 0;
            auto append_image = [&](int index, int64_t context_start) {
                auto [height, width] = image_shapes[index];
                int64_t start        = static_cast<int64_t>(layout.positions.size());
                layout.segments.push_back({start, start + height * width, context_start, index});
                layout.rope_layout.append_image(static_cast<int>(height), static_cast<int>(width));
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        layout.positions.push_back({static_cast<float>(position),
                                                    static_cast<float>(h - (height - height / 2)),
                                                    static_cast<float>(w - (width - width / 2))});
                    }
                }
                position += std::max(height, width);
            };
            for (int64_t i = 0; i < text_length;) {
                int tag       = image_slots.empty() ? 0 : image_slots[i];
                int64_t begin = i++;
                while (i < text_length && (image_slots.empty() ? 0 : image_slots[i]) == tag) {
                    ++i;
                }
                if (tag != 0) {
                    if (tag != next_image + 1 || next_image + 1 >= static_cast<int>(image_shapes.size()) ||
                        (i - begin) * 4 != image_shapes[next_image].first * image_shapes[next_image].second) {
                        throw std::runtime_error("Qwen Image 2.1: vision slots and reference latents must have matching sizes");
                    }
                    append_image(next_image++, begin);
                } else {
                    int64_t start = static_cast<int64_t>(layout.positions.size());
                    layout.segments.push_back({start, start + i - begin, begin, -1});
                    layout.rope_layout.append_tokens(i - begin);
                    for (int64_t j = begin; j < i; ++j, ++position) {
                        float p = static_cast<float>(position);
                        layout.positions.push_back({p, p, p});
                    }
                }
            }
            if (next_image + 1 != static_cast<int>(image_shapes.size())) {
                throw std::runtime_error("Qwen Image 2.1: missing reference image slots");
            }
            layout.prefix_length = static_cast<int64_t>(layout.positions.size());
            append_image(next_image, text_length);
            return layout;
        }
    };

    struct QwenImage21ModulationRows {
        ggml_tensor* target = nullptr;  // [hidden] factor for the image being drawn
        ggml_tensor* prefix = nullptr;  // [hidden] factor for the prefix tokens (text, example pictures); null in REUSE
    };

    struct QwenImage21PrefixCache {
        enum class Mode {
            NONE,
            STORE,
            REUSE
        };
        Mode mode = Mode::NONE;
        std::string name;
        std::string cut_group;
        int64_t prefix_length = 0;
        ggml_type type        = GGML_TYPE_F32;
        bool* flash_attn_used = nullptr;
    };

    // Rope::apply_rope's permute of the table (pe [2, 2, d_head/2, L] -> [2, d_head/2, L, 2]), done once per graph and
    // shared by every layer's q and k instead of once per apply_rope call.
    __STATIC_INLINE__ ggml_tensor* qwen_image_2_1_rope_table(ggml_context* ctx, ggml_tensor* pe) {
        return ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
    }

    // The rotary embedding the blocks apply to q and k (qwen_image_2_1_apply_rope): pe as the model builds it
    // ([2, 2, d_head/2, L]) for the fused ggml_rope_table op, or, where the backend does not have that op, the
    // decomposition's table (qwen_image_2_1_rope_table(pe), permuted once per graph).
    struct QwenImage21Rope {
        ggml_tensor* pe    = nullptr;
        ggml_tensor* table = nullptr;  // set when !fused
        bool fused         = false;
    };

    // Whether the backend runs ggml_rope_table for q/k shaped [d_head, n_head, L, 1] (F32, rows contiguous) and pe.
    __STATIC_INLINE__ bool qwen_image_2_1_rope_table_supported(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* pe, int64_t d_head, int64_t n_head) {
#ifndef SD_USE_UPSTREAM_GGML
        if (backend == nullptr || pe == nullptr || pe->type != GGML_TYPE_F32 || !ggml_is_contiguous(pe) ||
            pe->ne[0] != 2 || pe->ne[1] != 2 || pe->ne[2] * 2 != d_head) {
            return false;
        }
        // a probe node on an unallocated tensor of q's shape: supports_op only looks at types, shapes and strides
        auto probe_x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, pe->ne[3], 1);
        auto probe   = ggml_rope_table(ctx, probe_x, pe, GGML_TYPE_F32);
        return ggml_backend_supports_op(backend, probe);
#else
        return false;
#endif
    }

    // Rope::apply_rope (interleaved), same results. Fused: one ggml_rope_table op (no permute copies, repeats, muls
    // and add; each product and the sum rounded to F32 on its own, so bit-identical), which can also give the result
    // as F16 (type; the values ggml_cast would give). Otherwise the same operations as Rope::apply_rope, with the table
    // from qwen_image_2_1_rope_table, and always F32: callers check the result's type.
    // x: [d_head, n_head, L, N] -> [d_head, L, n_head * N]
    __STATIC_INLINE__ ggml_tensor* qwen_image_2_1_apply_rope(ggml_context* ctx, ggml_tensor* x, const QwenImage21Rope& rope, ggml_type type = GGML_TYPE_F32) {
        int64_t d_head = x->ne[0];
        int64_t n_head = x->ne[1];
        int64_t L      = x->ne[2];
        int64_t N      = x->ne[3];
#ifndef SD_USE_UPSTREAM_GGML
        if (rope.fused && x->type == GGML_TYPE_F32 && x->nb[0] == ggml_type_size(GGML_TYPE_F32) && rope.pe->ne[3] == L) {
            auto x_out = ggml_rope_table(ctx, x, rope.pe, type);        // [N, n_head, L, d_head]
            return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);  // [N*n_head, L, d_head]
        }
#endif
        // (the table is built per graph when the fused op is off; a q/k the op cannot take builds its own)
        ggml_tensor* pe_table = rope.table != nullptr ? rope.table : qwen_image_2_1_rope_table(ctx, rope.pe);
        if (N == 1 && ggml_is_contiguous(x)) {
            // One copy instead of two: element (j, l, h, c) = x[2j + c, h, l], the same [d_head/2, L, n_head, 2]
            // tensor apply_rope builds with permute(0, 2, 1, 3), reshape and permute(3, 0, 1, 2).
            x = ggml_reshape_4d(ctx, x, 2, d_head / 2, n_head, L);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 2, 1));  // [2, n_head, L, d_head/2]
        } else {
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));       // [N, n_head, L, d_head]
            x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);  // [N * n_head, L, d_head/2, 2]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));       // [2, N * n_head, L, d_head/2]
        }

        int64_t offset = x->nb[2] * x->ne[2];
        auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);  // [N * n_head, L, d_head/2]
        auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);  // [N * n_head, L, d_head/2]
        x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0            = ggml_repeat(ctx, x_0, temp_x);  // [N * n_head, L, d_head/2, 2]
        x_1            = ggml_repeat(ctx, x_1, temp_x);  // [N * n_head, L, d_head/2, 2]

        offset    = pe_table->nb[2] * pe_table->ne[2];
        auto pe_0 = ggml_view_3d(ctx, pe_table, pe_table->ne[0], pe_table->ne[1], pe_table->ne[2], pe_table->nb[1], pe_table->nb[2], offset * 0);  // [L, d_head/2, 2]
        auto pe_1 = ggml_view_3d(ctx, pe_table, pe_table->ne[0], pe_table->ne[1], pe_table->ne[2], pe_table->nb[1], pe_table->nb[2], offset * 1);  // [L, d_head/2, 2]

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));  // [N * n_head, L, d_head/2, 2]
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);                                // [N*n_head, L, d_head]
    }

    class QwenImage21ZeroCenterRMSNorm : public RMSNorm {
    public:
        using RMSNorm::RMSNorm;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto weight = params["weight"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, weight, prefix + "weight");
            }
            weight = ggml_scale_bias(ctx->ggml_ctx, weight, 1.f, 1.f);
            return ggml_mul(ctx->ggml_ctx, ggml_rms_norm(ctx->ggml_ctx, x, eps), weight);
        }
    };

    class QwenImage21TextProjection : public GGMLBlock {
    public:
        QwenImage21TextProjection(const QwenImage21Config& config) {
            blocks["text_norm"] = std::make_shared<QwenImage21ZeroCenterRMSNorm>(config.context_dim, 1e-6f);
            blocks["in_layer"]  = std::make_shared<Linear>(config.context_dim, config.hidden_size, false);
            blocks["out_layer"] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<QwenImage21ZeroCenterRMSNorm>(blocks["text_norm"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Linear>(blocks["in_layer"])->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            return std::dynamic_pointer_cast<Linear>(blocks["out_layer"])->forward(ctx, x);
        }
    };

    class QwenImage21Attention : public QwenImageAttention {
    public:
        QwenImage21Attention(const QwenImage21Config& config)
            : QwenImageAttention(config.hidden_size, config.head_dim, config.hidden_size / config.head_dim, 0, 0, false, false) {
            for (const auto* name : {"add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q", "norm_added_k", "to_add_out"}) {
                blocks.erase(name);
            }
        }

        // The factor forward(..., defer_out_scale = true) leaves for the caller to apply (to_out's deferred output
        // scale, see Linear::deferred_output_scale); 1 when it cannot be deferred.
        float deferred_output_scale(GGMLRunnerContext* ctx) {
            return std::dynamic_pointer_cast<Linear>(blocks["to_out.0"])->deferred_output_scale(ctx);
        }

        // blocks[norm_name] (RMSNorm) then qwen_image_2_1_apply_rope. Where the fused rope runs, one
        // ggml_rms_norm_mul_rope_table op instead (the backend's RMS_NORM + MUL arithmetic, bit-identical), unless a
        // weight adapter (LoRA) is active or the weight is not a plain F32 [d_head] tensor on the runtime backend's
        // device (or in host memory). type: F16 asks the fused op for an F16 result (F32 when it cannot run).
        ggml_tensor* norm_rope(GGMLRunnerContext* ctx, ggml_tensor* x, const char* norm_name, const QwenImage21Rope& rope, ggml_type type = GGML_TYPE_F32) {
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks[norm_name]);
#ifndef SD_USE_UPSTREAM_GGML
            ggml_tensor* w = norm->unpatched_weight();
            auto on_runtime_device = [&](ggml_tensor* t) {
                return t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer) ||
                       ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == ggml_backend_get_device(ctx->backend);
            };
            // (exactly RMSNorm: a subclass may compute something else in forward)
            if (rope.fused && typeid(*norm) == typeid(RMSNorm) && ctx->weight_adapter == nullptr && w != nullptr && w->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0] && on_runtime_device(w) &&
                x->type == GGML_TYPE_F32 && x->nb[0] == ggml_type_size(GGML_TYPE_F32) && rope.pe->ne[3] == x->ne[2]) {
                auto out = ggml_rms_norm_mul_rope_table(ctx->ggml_ctx, x, w, norm->get_eps(), rope.pe, type);
                if (ggml_backend_supports_op(ctx->backend, out)) {
                    return ggml_reshape_3d(ctx->ggml_ctx, out, x->ne[0], x->ne[2], x->ne[1] * x->ne[3]);  // [N*n_head, L, d_head]
                }
            }
#endif
            return qwen_image_2_1_apply_rope(ctx->ggml_ctx, norm->forward(ctx, x), rope, type);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const QwenImage21Rope& rope, const std::vector<QwenImage21Segment>& segments, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, bool defer_out_scale = false) {
            int64_t heads = x->ne[0] / dim_head;
            auto project  = [&](const char* name) {
                auto h = std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x);
                return ggml_reshape_4d(ctx->ggml_ctx, h, dim_head, heads, x->ne[1], x->ne[2]);
            };
            auto q = project("to_q");
            auto k = project("to_k");
            auto v = project("to_v");
            q      = norm_rope(ctx, q, "norm_q", rope);
            // Prefix-cache REUSE with flash attention: K straight from the RoPE op in F16, the values the attention's
            // F32 -> F16 cast would give, so the concat with the prefix runs in F16 and that cast goes away
            // (cast(concat(a, b)) == concat(cast(a), cast(b)) element by element). Not with an attention scale
            // override (it would scale the F16 values) or sage attention (it takes F32 K); if flash attention does not
            // end up taking K, the F32 path below is built instead.
            auto k_proj        = k;
            const bool k_f16   = cache.mode == QwenImage21PrefixCache::Mode::REUSE && rope.fused && ctx->flash_attn_enabled &&
                                 !ctx->sage_attn_enabled && !(ctx->attn_scale > 0.f);
            k                  = norm_rope(ctx, k_proj, "norm_k", rope, k_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
            if (cache.mode == QwenImage21PrefixCache::Mode::STORE) {
                // Preserve query-first attention evaluation while writing each layer's
                // prefix before its full-sequence K/V can accumulate across layers.
                ctx->expand_graph(q);
                auto persist = [&](ggml_tensor* tensor, int axis, const char* name) {
                    auto part = ggml_ext_slice(ctx->ggml_ctx, tensor, axis, 0, cache.prefix_length);
                    // Pack the contiguous data into wider rows so quantization blocks
                    // can exceed head_dim without padding or changing element order.
                    part      = ggml_reshape_2d(ctx->ggml_ctx, part, x->ne[0], cache.prefix_length);
                    auto copy = ggml_cast(ctx->ggml_ctx, part, cache.type);
                    // Keep the copy in this layer's segment so graph cuts do not
                    // retain or recompute the full-sequence K/V in the final segment.
                    sd::ggml_graph_cut::mark_graph_cut(copy, cache.cut_group, name);
                    ctx->persist_cache_tensor(cache.name + "." + name, copy);
                };
                persist(k, 1, "k");
                persist(v, 2, "v");
            }
            auto attend = [&](ggml_tensor* aq, ggml_tensor* ak, ggml_tensor* av, ggml_tensor* mask, bool* used = nullptr) {
                bool used_flash_attn = false;
                // strided_f32_v: on CUDA, flash attention converts the permuted F32 V view to F16 itself.
                auto out             = ggml_ext_attention_ext(ctx, aq, ak, av, heads, mask, true, ctx->flash_attn_enabled, 1.f, &used_flash_attn, true);
                if (used != nullptr) {
                    *used = used_flash_attn;
                    if (!used_flash_attn) {
                        return out;  // the caller retries; its result decides
                    }
                }
                if (cache.flash_attn_used != nullptr) {
                    *cache.flash_attn_used &= used_flash_attn;
                }
                return out;
            };
            ggml_tensor* result = nullptr;
            if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                auto prefix_k_cache = ctx->load_cache_tensor(cache.name + ".k");
                auto prefix_v_cache = ctx->load_cache_tensor(cache.name + ".v");
                GGML_ASSERT(prefix_k_cache != nullptr && prefix_v_cache != nullptr);
                // the prefix's values as the F32 path gets them (cache type -> F32), then -> F16 like its cast
                auto prefix_f16 = [&](ggml_tensor* prefix) {
                    if (prefix->type != GGML_TYPE_F16) {
                        if (prefix->type != GGML_TYPE_F32) {
                            prefix = ggml_cast(ctx->ggml_ctx, prefix, GGML_TYPE_F32);
                        }
                        prefix = ggml_cast(ctx->ggml_ctx, prefix, GGML_TYPE_F16);
                    }
                    return prefix;
                };
                if (k->type == GGML_TYPE_F16) {
                    auto prefix_k = ggml_reshape_4d(ctx->ggml_ctx, prefix_f16(prefix_k_cache), dim_head, cache.prefix_length, heads, k->ne[3]);
                    auto k_all    = ggml_concat(ctx->ggml_ctx, prefix_k, k, 1);
                    // V in F16 too: the image tokens cast before the concat, which then moves half the bytes, and flash
                    // attention takes the F16 V as it is instead of converting the F32 concat (the same
                    // ggml_cuda_cast<half> values: cast(concat(a, b)) == concat(cast(a), cast(b)) element by element).
                    auto prefix_v = ggml_reshape_4d(ctx->ggml_ctx, prefix_f16(prefix_v_cache), dim_head, heads, cache.prefix_length, v->ne[3]);
                    auto v_all    = ggml_concat(ctx->ggml_ctx, prefix_v, ggml_cast(ctx->ggml_ctx, v, GGML_TYPE_F16), 2);
                    bool used_fa  = false;
                    if (ggml_backend_supports_op(ctx->backend, k_all) && ggml_backend_supports_op(ctx->backend, v_all)) {
                        result = attend(q, k_all, v_all, nullptr, &used_fa);
                    }
                    if (!used_fa) {
                        result = nullptr;
                        k      = norm_rope(ctx, k_proj, "norm_k", rope);  // F32
                    }
                }
                if (result == nullptr) {
                    auto prefix_v = prefix_v_cache;
                    if (prefix_v->type != v->type) {
                        prefix_v = ggml_cast(ctx->ggml_ctx, prefix_v, v->type);
                    }
                    prefix_v      = ggml_reshape_4d(ctx->ggml_ctx, prefix_v, dim_head, heads, cache.prefix_length, v->ne[3]);
                    v             = ggml_concat(ctx->ggml_ctx, prefix_v, v, 2);
                    auto prefix_k = prefix_k_cache;
                    if (prefix_k->type != k->type) {
                        prefix_k = ggml_cast(ctx->ggml_ctx, prefix_k, k->type);
                    }
                    prefix_k = ggml_reshape_4d(ctx->ggml_ctx, prefix_k, dim_head, cache.prefix_length, heads, k->ne[3]);
                    k        = ggml_concat(ctx->ggml_ctx, prefix_k, k, 1);
                    result   = attend(q, k, v, nullptr);
                }
            } else {
                for (size_t i = 0; i < segments.size(); ++i) {
                    const auto& segment = segments[i];
                    auto sq             = ggml_ext_slice(ctx->ggml_ctx, q, 1, segment.start, segment.end);
                    auto sk             = ggml_ext_slice(ctx->ggml_ctx, k, 1, 0, segment.end);
                    auto sv             = ggml_ext_slice(ctx->ggml_ctx, v, 2, 0, segment.end);
                    auto out            = attend(sq, sk, sv, masks[i]);
                    result              = result == nullptr ? out : ggml_concat(ctx->ggml_ctx, result, out, 1);
                }
            }
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
                to_out->set_force_prec_f32(true);
            }
            return defer_out_scale ? to_out->forward_deferring_output_scale(ctx, result) : to_out->forward(ctx, result);
        }
    };

    class QwenImage21TransformerBlock : public GGMLBlock {
    public:
        QwenImage21TransformerBlock(const QwenImage21Config& config) {
            blocks["img_norm1"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["img_norm2"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["attn"]      = std::make_shared<QwenImage21Attention>(config);
            if (config.fused_mlp) {
                blocks["img_mlp.gate_up"] = std::make_shared<Linear>(config.hidden_size, 2 * config.intermediate_size, false);
            } else {
                blocks["img_mlp.proj"]       = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
                blocks["img_mlp.gate_layer"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
            }
            blocks["img_mlp.out"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, false);
        }

        // The factors one of the four modulation chunks ([hidden, 2]: row 0 for the image being drawn, row 1, the
        // zero timestep, for the prefix tokens) multiplies the tokens by: 1 + scale, or tanh(gate). They are the same
        // for every layer, so the model builds them once per graph instead of once per layer. factor: an exact power
        // of two the gated branch's output still has to be multiplied by (folded into the row).
        static QwenImage21ModulationRows modulation_rows(GGMLRunnerContext* ctx, ggml_tensor* params, bool with_prefix, bool gate, float factor = 1.f) {
            GGML_ASSERT(params->nb[0] == ggml_type_size(params->type) && params->ne[1] == 2);
            GGML_ASSERT(gate || factor == 1.f);
            auto row = [&](int64_t index) {
                // A 1-D view of the row ([hidden], contiguous): no copy, and ggml_mul broadcasts it exactly like
                // the old [hidden, 1] contiguous chunk.
                auto r = ggml_view_1d(ctx->ggml_ctx, params, params->ne[0], index * params->nb[1]);
                r      = gate ? ggml_tanh(ctx->ggml_ctx, r) : ggml_scale_bias(ctx->ggml_ctx, r, 1.f, 1.f);
                if (factor != 1.f) {
                    r = ggml_scale(ctx->ggml_ctx, r, factor);
                }
                ctx->expand_graph(r);
                return r;
            };
            QwenImage21ModulationRows rows;
            rows.target = row(0);
            if (with_prefix) {
                rows.prefix = row(1);
            }
            return rows;
        }

        static ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, const QwenImage21ModulationRows& rows, int64_t prefix_length) {
            if (prefix_length == 0) {
                // All tokens are the image being drawn (prefix-cache REUSE): multiply x itself, no slice copy.
                return ggml_mul(ctx, x, rows.target);
            }
            GGML_ASSERT(rows.prefix != nullptr);
            // Views, not copies: ggml_mul reads the strided parts and writes new contiguous tensors.
            auto target = ggml_mul(ctx, ggml_ext_slice(ctx, x, 1, prefix_length, x->ne[1], false), rows.target);
            auto prefix = ggml_mul(ctx, ggml_ext_slice(ctx, x, 1, 0, prefix_length, false), rows.prefix);
            return ggml_concat(ctx, prefix, target, 1);
        }

        float attn_deferred_output_scale(GGMLRunnerContext* ctx) {
            return std::dynamic_pointer_cast<QwenImage21Attention>(blocks["attn"])->deferred_output_scale(ctx);
        }

        // attn_out_scale_folded: modulation[1] already includes the attention's deferred output scale
        // (attn_deferred_output_scale), so the attention leaves it out.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const std::vector<QwenImage21ModulationRows>& modulation, const QwenImage21Rope& rope, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, bool attn_out_scale_folded = false) {
            const int64_t prefix_length = cache.mode == QwenImage21PrefixCache::Mode::REUSE ? 0 : layout.prefix_length;
            auto h                      = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"])->forward(ctx, x);
            h                           = modulate(ctx->ggml_ctx, h, modulation[0], prefix_length);
            h                           = std::dynamic_pointer_cast<QwenImage21Attention>(blocks["attn"])->forward(ctx, h, rope, layout.segments, masks, cache, attn_out_scale_folded);
            x                           = ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[1], prefix_length));
            h                           = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"])->forward(ctx, x);
            h                           = modulate(ctx->ggml_ctx, h, modulation[2], prefix_length);
            auto fused = blocks.find("img_mlp.gate_up");
            if (fused != blocks.end()) {
                auto gate_up = std::dynamic_pointer_cast<Linear>(fused->second)->forward(ctx, h);
                // gate_up = [gate | up] along ne0. ggml_swiglu (not swapped) computes silu(first half) * second half
                // straight from gate_up, the same per-element silu(gate) * up as the chunk + silu + mul below
                // (on CUDA even the same unary_gated_op_kernel<op_silu>), without copying both halves out first.
                auto swiglu = ggml_swiglu(ctx->ggml_ctx, gate_up);
                if (ctx->backend != nullptr && ggml_backend_supports_op(ctx->backend, swiglu)) {
                    h = swiglu;
                } else {
                    auto parts = ggml_ext_chunk(ctx->ggml_ctx, gate_up, 2, 0);
                    h          = ggml_mul(ctx->ggml_ctx, parts[1], ggml_silu(ctx->ggml_ctx, parts[0]));
                }
            } else {
                auto gate = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.gate_layer"])->forward(ctx, h);
                h         = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.proj"])->forward(ctx, h);
                h         = ggml_mul(ctx->ggml_ctx, h, ggml_silu(ctx->ggml_ctx, gate));
            }
            h = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.out"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[3], prefix_length));
        }
    };

    class QwenImage21Model : public GGMLBlock {
        QwenImage21Config config;

    public:
        QwenImage21Model(const QwenImage21Config& config)
            : config(config) {
            blocks["time_text_embed.timestep_embedder"] = std::make_shared<TimestepEmbedding>(256, config.hidden_size, 0, 0, false);
            blocks["txt_in"]                            = std::make_shared<QwenImage21TextProjection>(config);
            blocks["img_in"]                            = std::make_shared<Linear>(config.in_channels, config.hidden_size, false);
            blocks["modulation.1"]                      = std::make_shared<Linear>(config.hidden_size, 4 * config.hidden_size, false);
            blocks["norm_out.linear"]                   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
            blocks["norm_out.norm"]                     = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["proj_out"]                          = std::make_shared<Linear>(config.hidden_size, config.out_channels, false);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<QwenImage21TransformerBlock>(config);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache) {
            auto time = ggml_concat(ctx->ggml_ctx, timestep, ggml_ext_zeros_like(ctx->ggml_ctx, timestep), 0);
            // Runtime flow timesteps already use the [0, 1000] scale.
            time               = ggml_ext_timestep_embedding(ctx->ggml_ctx, time, 256, 10000, 1.f);
            time               = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_text_embed.timestep_embedder"])->forward(ctx, time);
            time               = ggml_silu(ctx->ggml_ctx, time);
            auto modulation    = std::dynamic_pointer_cast<Linear>(blocks["modulation.1"])->forward(ctx, time);
            auto mod           = ggml_ext_chunk(ctx->ggml_ctx, modulation, 4, 0);
            // Chunks 0 and 2 scale the normed tokens (1 + scale), 1 and 3 gate the residual branches (tanh).
            const bool with_prefix = cache.mode != QwenImage21PrefixCache::Mode::REUSE && layout.prefix_length > 0;
            // Each attention's to_out scales its input by 1/32 and its output back by 32 (against NaN with k-quants).
            // When every layer can defer that x32 (no LoRA, no linear_scale override, plain weights), fold it into the
            // attention gate rows instead: (32 h) tanh(g) and h (32 tanh(g)) are the same real product (x32 is exact),
            // so the rounded results are identical, and one 64 MiB scale per layer goes away.
            float attn_out_scale = 0.f;
            for (int i = 0; i < config.num_layers; ++i) {
                auto block    = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                const float f = block->attn_deferred_output_scale(ctx);
                attn_out_scale = (attn_out_scale == 0.f || attn_out_scale == f) ? f : 1.f;
            }
            const bool attn_out_scale_folded = attn_out_scale != 0.f && attn_out_scale != 1.f;
            std::vector<QwenImage21ModulationRows> mod_rows;
            for (int i = 0; i < 4; ++i) {
                mod_rows.push_back(QwenImage21TransformerBlock::modulation_rows(ctx, mod[i], with_prefix, i % 2 == 1,
                                                                                i == 1 && attn_out_scale_folded ? attn_out_scale : 1.f));
            }
            // The blocks' rotary embedding: ggml_rope_table where the backend has it, else the decomposition with
            // the table permuted once per graph.
            QwenImage21Rope rope;
            rope.pe    = pe;
            rope.fused = qwen_image_2_1_rope_table_supported(ctx->ggml_ctx, ctx->backend, pe, config.head_dim, config.hidden_size / config.head_dim);
            if (!rope.fused) {
                rope.table = qwen_image_2_1_rope_table(ctx->ggml_ctx, pe);
                ctx->expand_graph(rope.table);
            }
            auto img_in        = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            ggml_tensor* joint = nullptr;
            if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                joint = img_in->forward(ctx, DiT::patchify(ctx->ggml_ctx, x, 1, 1));
            } else {
                auto text = std::dynamic_pointer_cast<QwenImage21TextProjection>(blocks["txt_in"])->forward(ctx, context);
                for (const auto& segment : layout.segments) {
                    ggml_tensor* h;
                    if (segment.image_index < 0) {
                        h = ggml_ext_slice(ctx->ggml_ctx, text, 1, segment.context_start,
                                           segment.context_start + segment.end - segment.start);
                    } else {
                        auto image = segment.image_index == static_cast<int>(refs.size()) ? x : refs[segment.image_index];
                        h          = img_in->forward(ctx, DiT::patchify(ctx->ggml_ctx, image, 1, 1));
                    }
                    joint = joint == nullptr ? h : ggml_concat(ctx->ggml_ctx, joint, h, 1);
                }
            }
            sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.prelude", "joint");
            for (int i = 0; i < config.num_layers; ++i) {
                const std::string layer = "transformer_blocks." + std::to_string(i);
                auto layer_cache        = cache;
                layer_cache.name        = cache.name + "." + std::to_string(i);
                layer_cache.cut_group   = "qwen_image_2_1." + layer;
                auto block              = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks[layer]);
                joint                   = block->forward(ctx, joint, mod_rows, rope, layout, masks, layer_cache, attn_out_scale_folded);
                sd::ggml_graph_cut::mark_graph_cut(joint, layer_cache.cut_group, "joint");
            }
            if (cache.mode != QwenImage21PrefixCache::Mode::REUSE) {
                joint = ggml_ext_slice(ctx->ggml_ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            }
            auto scale = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_ext_chunk(ctx->ggml_ctx, time, 2, 1)[0]);
            joint      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out.norm"])->forward(ctx, joint);
            joint      = ggml_mul(ctx->ggml_ctx, joint, ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f));
            joint      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, joint);
            return DiT::unpatchify_and_crop(ctx->ggml_ctx, joint, x->ne[1], x->ne[0], 1, 1);
        }
    };

    struct QwenImage21Runner : public DiffusionModelRunner {
        QwenImage21Config config;
        QwenImage21Model model;
        std::vector<float> pe_data;
        std::vector<sd::Tensor<float>> mask_data;
        ggml_type prefix_cache_type = GGML_TYPE_COUNT;
        bool prefix_cache_enabled   = true;
        bool prefix_cache_disabled  = false;
        bool prefix_cache_auto_f32  = false;

        static bool supports_prefix_cache_type(ggml_type type) {
            if (type == GGML_TYPE_F32) {
                return true;
            }
            const auto* traits = ggml_get_type_traits(type);
            if (traits->from_float_ref == nullptr || traits->to_float == nullptr) {
                return false;
            }
            auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu == nullptr) {
                return false;
            }
            auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                ggml_init({3 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
            if (ctx == nullptr) {
                return false;
            }
            // Some reference quantizers have no runtime copy support. Query the
            // device through the registry so dynamically loaded CPU backends work.
            auto source  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ggml_blck_size(type));
            auto encoded = ggml_cast(ctx.get(), source, type);
            auto decoded = ggml_cast(ctx.get(), encoded, GGML_TYPE_F32);
            return ggml_backend_dev_supports_op(cpu, encoded) && ggml_backend_dev_supports_op(cpu, decoded);
        }

        QwenImage21Runner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr, const char* model_args = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(QwenImage21Config::detect_from_weights(weights, prefix)),
              model(config) {
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                if (key == "qwen_image_2_1_prefix_cache" && !parse_strict_bool(value, prefix_cache_enabled)) {
                    LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                } else if (key == "qwen_image_2_1_prefix_cache_type") {
                    if (value == "auto") {
                        prefix_cache_type = GGML_TYPE_COUNT;
                        continue;
                    }
                    const auto type = sd_type_to_ggml_type(str_to_sd_type(value.c_str()));
                    if (type == GGML_TYPE_COUNT) {
                        LOG_WARN("ignoring unknown Qwen Image 2.1 cache type '%s'", value.c_str());
                    } else if (!supports_prefix_cache_type(type)) {
                        LOG_WARN("ignoring Qwen Image 2.1 cache type '%s': runtime conversion to and from F32 is unavailable", value.c_str());
                    } else if (config.hidden_size % ggml_blck_size(type) != 0) {
                        LOG_WARN("ignoring Qwen Image 2.1 cache type '%s': block size %" PRId64 " does not divide hidden size %" PRId64,
                                 value.c_str(), ggml_blck_size(type), config.hidden_size);
                    } else {
                        prefix_cache_type = type;
                    }
                }
            }
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "qwen_image_2_1"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        bool has_prefix_cache(const QwenImage21PrefixCache& cache) {
            for (int i = 0; i < config.num_layers; ++i) {
                const auto name = cache.name + "." + std::to_string(i);
                auto k          = get_cache_tensor_by_name(name + ".k");
                auto v          = get_cache_tensor_by_name(name + ".v");
                if (k == nullptr || v == nullptr || k->type != cache.type || v->type != cache.type ||
                    k->ne[0] != config.hidden_size || k->ne[1] != cache.prefix_length || k->ne[2] != 1 || k->ne[3] != 1 ||
                    v->ne[0] != config.hidden_size || v->ne[1] != cache.prefix_length || v->ne[2] != 1 || v->ne[3] != 1) {
                    return false;
                }
            }
            return true;
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            const auto& x       = tensor_or_empty(inputs.x);
            const auto& context = tensor_or_empty(inputs.context);
            if (x.empty() || context.empty() || context.dim() < 2 || context.shape()[0] != config.context_dim ||
                tensor_or_empty(inputs.timesteps).numel() != 1 ||
                x.dim() != 4 || x.shape()[3] != 1 || x.shape()[2] != config.in_channels) {
                LOG_ERROR("Qwen Image 2.1 requires an image latent and text conditioning with batch size 1");
                return {};
            }
            static const std::vector<sd::Tensor<float>> empty_refs;
            const auto& refs = inputs.ref_latents && inputs.ref_image_params.pass_to_dit ? *inputs.ref_latents : empty_refs;
            std::vector<std::pair<int64_t, int64_t>> shapes;
            for (const auto& ref : refs) {
                if (ref.dim() != 4 || ref.shape()[2] != config.in_channels || ref.shape()[3] != 1) {
                    LOG_ERROR("Qwen Image 2.1: invalid reference latent shape");
                    return {};
                }
                shapes.emplace_back(ref.shape()[1], ref.shape()[0]);
            }
            shapes.emplace_back(x.shape()[1], x.shape()[0]);
            const auto* extra = std::get_if<QwenImage21DiffusionExtra>(&inputs.extra);
            QwenImage21Layout layout;
            try {
                layout = QwenImage21Layout::build(context.shape()[1], tensor_or_empty(extra ? extra->image_slots : nullptr), shapes);
            } catch (const std::exception& error) {
                LOG_ERROR("%s", error.what());
                return {};
            }
            if (!runner_started()) {
                prefix_cache_disabled = false;
                prefix_cache_auto_f32 = false;
            }
            QwenImage21PrefixCache cache;
            if (prefix_cache_enabled && !prefix_cache_disabled && extra != nullptr && extra->prefix_id != 0 && layout.prefix_length > 0) {
                cache.name = "qwen_image_2_1.prefix." + std::to_string(extra->prefix_id) +
                             ".circular." + std::to_string(circular_x_enabled) + std::to_string(circular_y_enabled);
                cache.prefix_length = layout.prefix_length;
                if (prefix_cache_type != GGML_TYPE_COUNT) {
                    cache.type = prefix_cache_type;
                } else if (!prefix_cache_auto_f32 && flash_attn_enabled && !sage_attn_enabled &&
                           (attn_scale <= 0.f || attn_scale == 1.f)) {
                    cache.type = GGML_TYPE_F16;
                }
                cache.mode = has_prefix_cache(cache) ? QwenImage21PrefixCache::Mode::REUSE : QwenImage21PrefixCache::Mode::STORE;
            }
            bool flash_attn_used = true;
            auto run             = [&](const QwenImage21PrefixCache& active_cache) {
                flash_attn_used    = true;
                auto checked_cache = active_cache;
                if (prefix_cache_type == GGML_TYPE_COUNT && active_cache.type == GGML_TYPE_F16) {
                    checked_cache.flash_attn_used = &flash_attn_used;
                }
                const bool cached         = active_cache.mode == QwenImage21PrefixCache::Mode::REUSE;
                const auto first_position = layout.positions.begin() + (cached ? layout.prefix_length : 0);
                Rope::Embedding embedding;
                embedding.ids.assign(first_position, layout.positions.end());
                const size_t offset             = cached ? static_cast<size_t>(layout.prefix_length) : 0;
                embedding.positions.token_count = embedding.ids.size();
                for (auto region : layout.rope_layout.images) {
                    if (region.begin >= offset) {
                        region.begin -= offset;
                        embedding.positions.images.push_back(region);
                    }
                }
                embedding.values = Rope::embed_nd(embedding.ids, 1, 10000.f, config.axes_dim, embedding.layout, &embedding.frequencies);
                pe_data          = finish_rope_pe(std::move(embedding));
                mask_data.clear();
                if (!cached) {
                    for (const auto& segment : layout.segments) {
                        sd::Tensor<float> mask;
                        if (segment.image_index < 0) {
                            mask = sd::Tensor<float>::zeros({segment.end, segment.end - segment.start});
                            for (int64_t q = segment.start; q < segment.end; ++q) {
                                for (int64_t k = q + 1; k < segment.end; ++k) {
                                    mask[k + segment.end * (q - segment.start)] = -INFINITY;
                                }
                            }
                        }
                        mask_data.push_back(std::move(mask));
                    }
                }
                auto build = [&]() {
                    auto graph = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE * 2);
                    auto pe    = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2,
                                                                layout.positions.size() - (cached ? layout.prefix_length : 0));
                    set_backend_tensor_data(pe, pe_data.data());
                    std::vector<ggml_tensor*> masks, ref_inputs;
                    for (const auto& mask : mask_data) {
                        masks.push_back(mask.empty() ? nullptr : make_input(mask));
                    }
                    if (!cached) {
                        for (const auto& ref : refs) {
                            ref_inputs.push_back(make_input(ref));
                        }
                    }
                    auto ctx = get_context(graph);
                    auto out = model.forward(&ctx, make_input(x), make_input(*inputs.timesteps), cached ? nullptr : make_input(context),
                                                         ref_inputs, pe, layout, masks, checked_cache);
                    if (!flash_attn_used) {
                        return static_cast<ggml_cgraph*>(nullptr);
                    }
                    ggml_build_forward_expand(graph, out);
                    return graph;
                };
                return restore_trailing_singleton_dims(GGMLRunner::compute(build, n_threads, false), x.dim());
            };
            auto result = run(cache);
            if (result.empty() && !flash_attn_used) {
                // Casting an F16 cache back to F32 cannot recover its original values.
                // Recompute the prefix before executing a graph that falls back from FA.
                free_cache_ctx_and_buffer();
                prefix_cache_auto_f32 = true;
                cache.type            = GGML_TYPE_F32;
                cache.mode            = QwenImage21PrefixCache::Mode::STORE;
                LOG_DEBUG("Qwen Image 2.1: Flash Attention unavailable; using F32 prefix caching for this sampling run");
                result = run(cache);
            }
            if (result.empty() && last_compute_status() == GGML_STATUS_ALLOC_FAILED &&
                (cache.mode != QwenImage21PrefixCache::Mode::NONE || !cache_.empty())) {
                // The failed graph has ended before persistent inputs are released.
                free_cache_ctx_and_buffer();
                prefix_cache_disabled = true;
                LOG_WARN("Qwen Image 2.1: insufficient memory for prefix caching; retrying without it for this sampling run");
                return run(QwenImage21PrefixCache{});
            }
            if (!result.empty() && cache.mode == QwenImage21PrefixCache::Mode::STORE) {
                if (!has_prefix_cache(cache)) {
                    free_cache_ctx_and_buffer();
                    prefix_cache_disabled = true;
                    LOG_WARN("Qwen Image 2.1: incomplete prefix cache; disabling it for this sampling run");
                } else {
                    LOG_DEBUG("Qwen Image 2.1: cached prefix %" PRIu64 " (%" PRId64 " tokens, %s)", extra->prefix_id, layout.prefix_length, ggml_type_name(cache.type));
                }
            }
            return result;
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
