#ifndef __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
#define __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
        // STORE and uncached graphs: the picture being drawn (the last segment) attends to K in F16, as the REUSE graphs
        // give it to the attention, so a step that reuses a kept prefix computes what these do (see
        // QwenImage21Runner::PrefixKeep).
        bool f16_target_keys = false;
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
        return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);                               // [N*n_head, L, d_head]
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
            ggml_tensor* w         = norm->unpatched_weight();
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
                if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                    // q, k and v's matmuls one after the other in the graph (instead of each next to its own RoPE
                    // and concat): the CUDA backend then quantizes their shared input once for all three. Same
                    // values. Only in REUSE: keeping all three projections alive at once raises the graph's peak
                    // memory by one projection, which the STORE graph (the whole sequence) sets; REUSE stays below.
                    ctx->expand_graph(h);
                }
                return ggml_reshape_4d(ctx->ggml_ctx, h, dim_head, heads, x->ne[1], x->ne[2]);
            };
            auto q = project("to_q");
            auto k = project("to_k");
            auto v = project("to_v");
            q      = norm_rope(ctx, q, "norm_q", rope);
            // Prefix-cache REUSE with flash or sage attention: K straight from the RoPE op in F16, the values the
            // attention's F32 -> F16 cast would give, so the concat with the prefix runs in F16 and that cast goes away
            // (cast(concat(a, b)) == concat(cast(a), cast(b)) element by element); the sage op takes the F16 K as it is
            // too (it quantizes K to INT8 from it). Not with an attention scale override (it would scale the F16
            // values); if neither attention ends up taking K, the F32 path below is built instead.
            auto k_proj      = k;
            const bool k_f16 = cache.mode == QwenImage21PrefixCache::Mode::REUSE && rope.fused &&
                               (ctx->flash_attn_enabled || ctx->sage_attn_enabled) && !(ctx->attn_scale > 0.f);
            k = norm_rope(ctx, k_proj, "norm_k", rope, k_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
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
                auto out = ggml_ext_attention_ext(ctx, aq, ak, av, heads, mask, true, ctx->flash_attn_enabled, 1.f, &used_flash_attn, true);
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
                    if (used_fa) {
                        // The concats into the graph now, each followed by a node keeping its RoPE / cast input
                        // allocated past it: the CUDA backend then writes K's RoPE and V's cast straight into the
                        // concats' outputs (whose memory could otherwise be their just-freed inputs) instead of
                        // copying them there. Same values; only the order of independent nodes changes.
                        ctx->expand_graph(k_all);
                        ctx->keep_allocated(k_proj);
                        ctx->expand_graph(v_all);
                        ctx->keep_allocated(v);
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
                // f16_target_keys: the same F32 -> F16 values the REUSE graph's K has (its cache, its RoPE op's output)
                auto keys = [&](ggml_tensor* sk, size_t i) {
                    const bool f16 = cache.f16_target_keys && i + 1 == segments.size() && rope.fused &&
                                     !(ctx->attn_scale > 0.f) && sk->type == GGML_TYPE_F32;
                    return f16 ? ggml_cast(ctx->ggml_ctx, sk, GGML_TYPE_F16) : sk;
                };
                for (size_t i = 0; i < segments.size(); ++i) {
                    const auto& segment = segments[i];
                    auto sq             = ggml_ext_slice(ctx->ggml_ctx, q, 1, segment.start, segment.end);
                    auto sk             = keys(ggml_ext_slice(ctx->ggml_ctx, k, 1, 0, segment.end), i);
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
            auto fused                  = blocks.find("img_mlp.gate_up");
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

        // What the blocks of one step share, built once per graph: the timestep's embedding, the four modulation rows,
        // the rotary embedding, and whether the attention gate rows carry to_out's deferred x32.
        struct StepInputs {
            ggml_tensor* time = nullptr;  // [hidden, 2]: silu of the embedded timestep, and of t = 0 (the prefix's)
            std::vector<QwenImage21ModulationRows> mod_rows;
            QwenImage21Rope rope;
            bool attn_out_scale_folded = false;
        };

        ggml_tensor* time_embedding(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto time = ggml_concat(ctx->ggml_ctx, timestep, ggml_ext_zeros_like(ctx->ggml_ctx, timestep), 0);
            // Runtime flow timesteps already use the [0, 1000] scale.
            time = ggml_ext_timestep_embedding(ctx->ggml_ctx, time, 256, 10000, 1.f);
            time = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_text_embed.timestep_embedder"])->forward(ctx, time);
            return ggml_silu(ctx->ggml_ctx, time);
        }

        StepInputs step_inputs(GGMLRunnerContext* ctx, ggml_tensor* timestep, ggml_tensor* pe, const QwenImage21Layout& layout, const QwenImage21PrefixCache& cache) {
            StepInputs step;
            step.time       = time_embedding(ctx, timestep);
            auto modulation = std::dynamic_pointer_cast<Linear>(blocks["modulation.1"])->forward(ctx, step.time);
            auto mod        = ggml_ext_chunk(ctx->ggml_ctx, modulation, 4, 0);
            // Chunks 0 and 2 scale the normed tokens (1 + scale), 1 and 3 gate the residual branches (tanh).
            const bool with_prefix = cache.mode != QwenImage21PrefixCache::Mode::REUSE && layout.prefix_length > 0;
            // Each attention's to_out scales its input by 1/32 and its output back by 32 (against NaN with k-quants).
            // When every layer can defer that x32 (no LoRA, no linear_scale override, plain weights), fold it into the
            // attention gate rows instead: (32 h) tanh(g) and h (32 tanh(g)) are the same real product (x32 is exact),
            // so the rounded results are identical, and one 64 MiB scale per layer goes away.
            float attn_out_scale = 0.f;
            for (int i = 0; i < config.num_layers; ++i) {
                auto block     = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                const float f  = block->attn_deferred_output_scale(ctx);
                attn_out_scale = (attn_out_scale == 0.f || attn_out_scale == f) ? f : 1.f;
            }
            step.attn_out_scale_folded = attn_out_scale != 0.f && attn_out_scale != 1.f;
            for (int i = 0; i < 4; ++i) {
                step.mod_rows.push_back(QwenImage21TransformerBlock::modulation_rows(ctx, mod[i], with_prefix, i % 2 == 1,
                                                                                     i == 1 && step.attn_out_scale_folded ? attn_out_scale : 1.f));
            }
            // The blocks' rotary embedding: ggml_rope_table where the backend has it, else the decomposition with
            // the table permuted once per graph.
            step.rope.pe    = pe;
            step.rope.fused = qwen_image_2_1_rope_table_supported(ctx->ggml_ctx, ctx->backend, pe, config.head_dim, config.hidden_size / config.head_dim);
            if (!step.rope.fused) {
                step.rope.table = qwen_image_2_1_rope_table(ctx->ggml_ctx, pe);
                ctx->expand_graph(step.rope.table);
            }
            return step;
        }

        // The tokens the blocks take: the prefix (text, example pictures) and the picture being drawn, or in REUSE
        // only the picture (the prefix's keys and values come from the cache).
        ggml_tensor* embed(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, const QwenImage21Layout& layout, const QwenImage21PrefixCache& cache) {
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
            return joint;
        }

        // Transformer blocks [begin, end) on joint, each a graph cut of its own.
        ggml_tensor* run_blocks(GGMLRunnerContext* ctx, ggml_tensor* joint, const StepInputs& step, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const std::string layer = "transformer_blocks." + std::to_string(i);
                auto layer_cache        = cache;
                layer_cache.name        = cache.name + "." + std::to_string(i);
                layer_cache.cut_group   = "qwen_image_2_1." + layer;
                auto block              = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks[layer]);
                joint                   = block->forward(ctx, joint, step.mod_rows, step.rope, layout, masks, layer_cache, step.attn_out_scale_folded);
                sd::ggml_graph_cut::mark_graph_cut(joint, layer_cache.cut_group, "joint");
            }
            return joint;
        }

        // The last block's output, the drawn picture's tokens only (outside REUSE the prefix tokens come first), to the
        // model's output: norm_out, its modulation, proj_out, unpatchify.
        ggml_tensor* head(GGMLRunnerContext* ctx, ggml_tensor* joint, ggml_tensor* time, ggml_tensor* x) {
            auto scale = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_ext_chunk(ctx->ggml_ctx, time, 2, 1)[0]);
            joint      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out.norm"])->forward(ctx, joint);
            joint      = ggml_mul(ctx->ggml_ctx, joint, ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f));
            joint      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, joint);
            return DiT::unpatchify_and_crop(ctx->ggml_ctx, joint, x->ne[1], x->ne[0], 1, 1);
        }

        // One graph for the whole step (the model without the first-block cache).
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache) {
            auto step  = step_inputs(ctx, timestep, pe, layout, cache);
            auto joint = embed(ctx, x, context, refs, layout, cache);
            joint      = run_blocks(ctx, joint, step, layout, masks, cache, 0, config.num_layers);
            if (cache.mode != QwenImage21PrefixCache::Mode::REUSE) {
                joint = ggml_ext_slice(ctx->ggml_ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            }
            return head(ctx, joint, step.time, x);
        }

        int num_layers() const { return config.num_layers; }

        // The drawn picture's tokens of a block's output, as forward hands them to the head.
        ggml_tensor* drawn_tokens(GGMLRunnerContext* ctx, ggml_tensor* joint, const QwenImage21Layout& layout, const QwenImage21PrefixCache& cache) {
            if (cache.mode != QwenImage21PrefixCache::Mode::REUSE) {
                joint = ggml_ext_slice(ctx->ggml_ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            }
            return joint;
        }

        // The first-block cache's graph A: the prelude and blocks [0, head). after_block0(h0, h1) runs once block 0
        // (input h0, output h1) is built, before the next blocks; returns the output of block head - 1. All over the
        // whole sequence joint holds (see embed).
        ggml_tensor* forward_first_blocks(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, int head, const std::function<void(ggml_tensor*, ggml_tensor*)>& after_block0) {
            auto step = step_inputs(ctx, timestep, pe, layout, cache);
            auto h0   = embed(ctx, x, context, refs, layout, cache);
            auto h1   = run_blocks(ctx, h0, step, layout, masks, cache, 0, 1);
            after_block0(h0, h1);
            return run_blocks(ctx, h1, step, layout, masks, cache, 1, head);
        }

        // Graph B without a tail: blocks [begin, last] on graph A's output (the same sequence). Returns the drawn
        // picture's tokens after the last block (sliced as forward slices them) and sets *time for the head.
        ggml_tensor* forward_remaining_blocks(GGMLRunnerContext* ctx, ggml_tensor* h, ggml_tensor* timestep, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, int begin, ggml_tensor** time) {
            auto step  = step_inputs(ctx, timestep, pe, layout, cache);
            auto joint = run_blocks(ctx, h, step, layout, masks, cache, begin, config.num_layers);
            *time      = step.time;
            return drawn_tokens(ctx, joint, layout, cache);
        }

        // Graph C without a tail: the head on the drawn picture's tokens (graph A's output + the cached residual).
        ggml_tensor* forward_head(GGMLRunnerContext* ctx, ggml_tensor* tokens, ggml_tensor* timestep, ggml_tensor* x) {
            return head(ctx, tokens, time_embedding(ctx, timestep), x);
        }
    };

    struct QwenImage21Runner : public DiffusionModelRunner {
        QwenImage21Config config;
        QwenImage21Model model;
        // Rotary tables already computed, by everything they were computed from (see compute): one per prompt layout
        // and prefix-cache mode (cond / uncond x STORE / REUSE), reused round-robin.
        struct RopeTable {
            std::vector<double> key;
            std::vector<float> values;
        };
        std::array<RopeTable, 4> rope_tables;
        size_t next_rope_table = 0;
        std::vector<sd::Tensor<float>> mask_data;
        ggml_type prefix_cache_type = GGML_TYPE_COUNT;
        bool prefix_cache_enabled   = true;
        bool prefix_cache_disabled  = false;
        bool prefix_cache_auto_f32  = false;

        // Whether the prefixes outlive the sampling run (model arg qwen_image_2_1_prefix_cache_keep=auto|on|off). Every
        // re-roll (a new generate call) and every variation (batch_count) starts a new run, which computed the prefix
        // again in its first step: RTX 3060, one 1024x1024 example picture (4,155 prefix tokens), 4.4 s instead of
        // 1.9 s under SageAttention, 5.4 s instead of 2.6 s under flash attention. The kept K/V are exactly the ones a
        // STORE step computes, but that step's own attention takes the keys in F32 and a REUSE step in F16: the same
        // values for flash attention (it casts them to F16), not for SageAttention (it quantizes them to INT8 from
        // what it gets). So, measured: under flash attention a reused prefix gives the picture bit for bit; under
        // SageAttention it differs (55 dB PSNR at 25 steps with the example picture; 35.5 dB and a lock's shape
        // changed, text only), and a variation would no longer be the picture its seed gives on its own.
        // AUTO: kept under flash attention without SageAttention only (the pictures stay what they were).
        // ON: kept everywhere; under SageAttention the STORE step then gives the drawn picture's attention the keys in
        // F16 as well (f16_target_keys), so a picture is what its seed gives with or without a kept prefix (measured
        // bit for bit), and differs from AUTO's by that rounding (53-55 dB PSNR with the example picture).
        // OFF: never kept.
        enum class PrefixKeep {
            AUTO,
            ON,
            OFF
        };
        PrefixKeep prefix_cache_keep = PrefixKeep::AUTO;

        bool keep_prefixes() const {
            return prefix_cache_keep == PrefixKeep::ON ||
                   (prefix_cache_keep == PrefixKeep::AUTO && flash_attn_enabled && !sage_attn_enabled);
        }

        // The prefixes whose K/V the cache holds, with copies of everything those were computed from (when kept):
        // matched by content, as the tensors behind the engine's prefix ids are new for every request. The K/V do not
        // depend on the seed or the timestep.
        struct RetainedPrefix {
            std::string name;
            sd::Tensor<float> context;
            sd::Tensor<int32_t> image_slots;
            std::vector<sd::Tensor<float>> refs;
            std::vector<int64_t> x_shape;  // the STORE graph's shapes, so the cached values are the ones it computes
            std::array<float, 2> scales;   // linear_scale, attn_scale
            std::array<bool, 4> flags;     // circular x / y, flash and sage attention
            bool used = false;             // by the current sampling run
        };
        std::vector<RetainedPrefix> retained_prefixes;
        std::map<uint64_t, std::string> run_prefixes;  // this run's prefix ids -> RetainedPrefix::name
        uint64_t next_prefix_serial = 1;

        // ---- The First-Block Cache ----
        // The product form (model arg qwen_image_2_1_first_block_cache=<fraction>[:<tail>], environment variable
        // SD_QWEN_FIRST_BLOCK_CACHE read at the start of every sampling run): a fixed share of the steps reuses the
        // cached residual of the middle blocks, on a schedule fixed by the step count (first_block_cache_schedule), with
        // block 0 computed fresh and the last quarter of the blocks (the tail) recomputed on top of the cached residual:
        // the fine detail, contrast and gloss come from the last blocks (GTX 1070 measurements, NOTES-block-cache.md).
        // It runs the machinery below with the schedule as its step list, head 1 and that tail.
        // The research knobs (model arg qwen_image_2_1_block_cache, environment variables SD_QWEN_BLOCK_CACHE,
        // SD_QWEN_BLOCK_CACHE_STEPS / _TAIL / _HEAD, read at load; used only by runs without the product form): the
        // "First-Block Cache" of para-attn / WaveSpeed, a relative of TeaCache: on steps where the model's input
        // barely changed since the last step, the middle blocks would add almost the same residual as they did on the
        // last fully computed step, so that residual is reused instead of computed. With H blocks computed fresh at the
        // start of every step (head, default 1) and T blocks recomputed at the end of a skipped step (tail, default 0),
        // the cached residual covers blocks [H, 32 - T). Each step is then two graphs instead of one (without the cache:
        // the single graph of forward, unchanged):
        //   A  the prelude and blocks 0..H-1 on the step's latent and timestep. Persists hf (block H-1's output, the
        //      whole sequence the step's joint holds: in STORE / uncached mode the prefix tokens first, in REUSE only the
        //      drawn picture's) and r1 = h1 - h0, block 0's residual over the drawn picture's tokens (whatever H is).
        //      Its output is the indicator's two sums, sum|r1 - r1_prev| and sum|r1_prev| (r1_prev: the previous
        //      step's r1), reduced on the card: only two floats come back.
        //   B  (a fully computed step) hf -> blocks H..31 -> the head; persists D, the residual of blocks H..31-T: over
        //      the drawn picture's tokens when T = 0, over the whole sequence the step holds when T > 0 (the prefix
        //      tokens' part is the same on every step: they never attend to the drawn picture and take the zero
        //      timestep). A + B compute what the single graph computes.
        //   C  (a skipped step) T = 0: the head on hf + D; T > 0: blocks 32-T..31 on hf + D, then the head. D is the
        //      last fully computed step's.
        // The policy, on the host between A and B/C: ind = sum|r1 - r1_prev| / sum|r1_prev| is added up since the
        // last fully computed step; the step is skipped (C) while that sum stays below the threshold, unless it is one
        // of the first / last steps that are always computed, the cap of skipped steps in a row is reached, or there is
        // nothing to compare with or no D yet; otherwise B, which caches the new D and resets the sum. An explicit list
        // of steps to skip (SD_QWEN_BLOCK_CACHE_STEPS, for experiments) replaces that policy: exactly the listed steps
        // are skipped (the indicator is still computed and logged). A STORE step (the one that fills the prefix cache)
        // is always computed fully. The sampler's steps and schedule do not change: a skipped step advances the
        // sampler with the approximated output.
        // Keys: one state and one set of tensors (hf, r1, D: F32, hidden x tokens, ~64 MiB each at 1024x1024) per
        // condition, by the engine's prefix id (a run with CFG has two, each with its own indicator and decisions). The
        // state starts again with every sampling run (a re-roll, each picture of a batch); the tensors are cache
        // entries the runner does not retain, so they live on the card from step to step and go at the run's end
        // (hf only from A to the step's second graph; at most four of them at once, during A: hf, r1, r1_prev, D).
        // Without the engine's step numbers or prefix ids (generation extensions), the step is the single graph.
        struct BlockCacheConfig {
            float threshold = 0.f;  // 0: off
            int first_full  = 2;    // steps always computed fully at the start
            int last_full   = 1;    // ... and at the end
            int max_skips   = 3;    // skipped steps in a row at most
            int tail        = 0;    // blocks recomputed at the end of a skipped step
            int head        = 1;    // blocks computed fresh at the start of every step
            std::vector<int> steps; // the steps to skip, replacing the policy above (SD_QWEN_BLOCK_CACHE_STEPS)
            bool enabled() const { return threshold > 0.f || !steps.empty(); }
        };
        BlockCacheConfig block_cache;  // the research setting, read at load
        BlockCacheConfig run_cache;    // this sampling run's: the product form's schedule, or the research setting
        bool run_first_block_cache = false;

        struct FirstBlockCacheConfig {
            float fraction = 0.f;  // the share of the sampling steps that reuse the cache; 0: off
            int tail       = -1;   // the last blocks recomputed on those steps; -1: a quarter of the layers
        };
        FirstBlockCacheConfig first_block_cache;  // the model arg's, the default of every run

        // "<fraction>[:<tail blocks>]" (',' also separates them, except inside --model-args)
        static bool parse_first_block_cache(const std::string& text, int num_layers, FirstBlockCacheConfig& config) {
            const auto split = text.find_first_of(":,");
            FirstBlockCacheConfig parsed;
            if (!parse_strict_float(text.substr(0, split), parsed.fraction) || !(parsed.fraction >= 0.f && parsed.fraction <= 1.f)) {
                return false;
            }
            if (split != std::string::npos &&
                (!parse_strict_int(text.substr(split + 1), parsed.tail) || parsed.tail < 0 || parsed.tail > num_layers - 1)) {
                return false;
            }
            config = parsed;
            return true;
        }

        // The steps of an N-step run that reuse the cache (1-based, ascending). The first W = max(2, round(0.32 N))
        // steps and the last step are always computed; K = round(fraction N) of the slots between (W + 1 .. N - 1) reuse
        // the cache, spread as evenly as they go: one at a time while there are at most one more of them than computed
        // slots, else in runs separated by single computed steps, the runs as even as possible (at most 3 long while
        // K <= 3/4 of the slots) and the longer ones nearest the middle of the slots, where a step changes the least
        // (the later one on a tie). N = 25, fraction 0.5: W = 8, K = 13: 9-11, 13-15, 17-20, 22-24.
        static std::vector<int> first_block_cache_schedule(float fraction, int steps) {
            std::vector<int> cached;
            if (!(fraction > 0.f) || steps <= 0) {
                return cached;
            }
            const int warmup = std::max(2, static_cast<int>(std::floor(0.32 * steps + 0.5)));
            const int slots  = steps - 1 - warmup;
            const int count  = std::min(slots, static_cast<int>(std::floor(static_cast<double>(fraction) * steps + 0.5)));
            if (slots <= 0 || count <= 0) {
                return cached;
            }
            const int first    = warmup + 1;
            const int computed = slots - count;
            if (count <= computed + 1) {
                for (int i = 0; i < count; ++i) {
                    cached.push_back(first + static_cast<int>(std::floor((i + 0.5) * slots / count)));
                }
                return cached;
            }
            const int runs = computed + 1;
            std::vector<int> length(runs, count / runs);
            std::vector<int> order(runs);
            for (int i = 0; i < runs; ++i) {
                order[i] = i;
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                const double da = std::abs(a - (runs - 1) / 2.0), db = std::abs(b - (runs - 1) / 2.0);
                return da != db ? da < db : a > b;
            });
            for (int i = 0; i < count % runs; ++i) {
                ++length[order[i]];
            }
            int step = first;
            for (int run = 0; run < runs; ++run, ++step) {
                for (int i = 0; i < length[run]; ++i) {
                    cached.push_back(step++);
                }
            }
            return cached;
        }

        enum class BlockCacheGraph {
            NONE,              // the whole step in one graph (forward)
            FIRST_BLOCK,       // A
            REMAINING_BLOCKS,  // B
            HEAD               // C
        };
        struct BlockCacheState {
            float accumulated = 0.f;  // the indicator added up since the last fully computed step
            int skips_in_row  = 0;
            std::vector<int> skipped_steps;
            std::string indicators;  // for the run's log line
            bool reported = false;
        };
        std::map<uint64_t, BlockCacheState> block_cache_states;  // this run's, by prefix id
        bool block_cache_disabled = false;                        // for this run (out of memory)
        bool block_cache_unavailable_logged = false;
        static constexpr const char* BLOCK_CACHE_ENTRY = "qwen_image_2_1.block_cache.";

        static std::string block_cache_entry(uint64_t prefix_id, const char* tensor) {
            return BLOCK_CACHE_ENTRY + std::to_string(prefix_id) + "." + tensor;
        }

        // "<threshold>[:<first full steps>:<last full steps>:<max skips in a row>[:<tail>[:<head>]]]" (',' also separates
        // the fields, except inside --model-args, where it separates the args)
        static bool parse_block_cache(const std::string& text, BlockCacheConfig& config) {
            std::vector<std::string> fields(1);
            for (char c : text) {
                if (c == ':' || c == ',') {
                    fields.emplace_back();
                } else {
                    fields.back() += c;
                }
            }
            if (fields.size() > 6) {
                return false;
            }
            BlockCacheConfig parsed;
            if (!parse_strict_float(fields[0], parsed.threshold) || !(parsed.threshold >= 0.f) || !std::isfinite(parsed.threshold)) {
                return false;
            }
            int* counts[] = {&parsed.first_full, &parsed.last_full, &parsed.max_skips, &parsed.tail, &parsed.head};
            for (size_t i = 1; i < fields.size(); ++i) {
                if (!parse_strict_int(fields[i], *counts[i - 1]) || *counts[i - 1] < 0) {
                    return false;
                }
            }
            parsed.steps = config.steps;
            config       = parsed;
            return true;
        }

        // Step numbers: "9,10,12" (' ', ':' and ';' separate them too)
        static bool parse_step_list(const std::string& text, std::vector<int>& steps) {
            std::vector<int> parsed;
            std::string field;
            auto take = [&]() {
                if (!trim(field).empty()) {
                    int step = 0;
                    if (!parse_strict_int(field, step) || step < 1) {
                        return false;
                    }
                    parsed.push_back(step);
                }
                field.clear();
                return true;
            };
            for (char c : text) {
                if (c == ',' || c == ':' || c == ';' || c == ' ') {
                    if (!take()) {
                        return false;
                    }
                } else {
                    field += c;
                }
            }
            if (!take()) {
                return false;
            }
            steps = parsed;
            return true;
        }

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
            bool block_cache_set = false;
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                if (key == "qwen_image_2_1_first_block_cache") {
                    if (!parse_first_block_cache(value, config.num_layers, first_block_cache)) {
                        LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                    }
                } else if (key == "qwen_image_2_1_block_cache") {
                    block_cache_set = parse_block_cache(value, block_cache);
                    if (!block_cache_set) {
                        LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                    }
                } else if (key == "qwen_image_2_1_prefix_cache" && !parse_strict_bool(value, prefix_cache_enabled)) {
                    LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                } else if (key == "qwen_image_2_1_prefix_cache_keep") {
                    bool keep = false;
                    if (value == "auto") {
                        prefix_cache_keep = PrefixKeep::AUTO;
                    } else if (parse_strict_bool(value, keep)) {
                        prefix_cache_keep = keep ? PrefixKeep::ON : PrefixKeep::OFF;
                    } else {
                        LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                    }
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
            const char* block_cache_env = std::getenv("SD_QWEN_BLOCK_CACHE");
            if (!block_cache_set && block_cache_env != nullptr && block_cache_env[0] != '\0' &&
                !parse_block_cache(block_cache_env, block_cache)) {
                LOG_WARN("ignoring invalid SD_QWEN_BLOCK_CACHE '%s'", block_cache_env);
            }
            // The experiments' variables (they override the matching fields of the setting above)
            auto env_count = [](const char* name, int& value) {
                const char* text = std::getenv(name);
                if (text != nullptr && text[0] != '\0') {
                    int parsed = 0;
                    if (parse_strict_int(text, parsed) && parsed >= 0) {
                        value = parsed;
                    } else {
                        LOG_WARN("ignoring invalid %s '%s'", name, text);
                    }
                }
            };
            env_count("SD_QWEN_BLOCK_CACHE_TAIL", block_cache.tail);
            env_count("SD_QWEN_BLOCK_CACHE_HEAD", block_cache.head);
            const char* steps_env = std::getenv("SD_QWEN_BLOCK_CACHE_STEPS");
            if (steps_env != nullptr && steps_env[0] != '\0' && !parse_step_list(steps_env, block_cache.steps)) {
                LOG_WARN("ignoring invalid SD_QWEN_BLOCK_CACHE_STEPS '%s'", steps_env);
            }
            if (block_cache.head < 1 || block_cache.head + block_cache.tail > config.num_layers) {
                LOG_WARN("Qwen Image 2.1: first-block cache head %d / tail %d do not fit %d layers; using head 1, tail 0",
                         block_cache.head, block_cache.tail, config.num_layers);
                block_cache.head = 1;
                block_cache.tail = 0;
            }
            if (first_block_cache.fraction > 0.f) {
                LOG_INFO("Qwen Image 2.1: First-Block Cache by default for %.0f %% of the steps (SD_QWEN_FIRST_BLOCK_CACHE sets it per run)",
                         first_block_cache.fraction * 100.0);
            }
            if (!block_cache.steps.empty()) {
                std::string list;
                for (int step : block_cache.steps) {
                    list += (list.empty() ? "" : " ") + std::to_string(step);
                }
                LOG_INFO("Qwen Image 2.1: first-block cache on, skipping exactly steps %s (head %d, tail %d blocks)",
                         list.c_str(), block_cache.head, block_cache.tail);
            } else if (block_cache.threshold > 0.f) {
                LOG_INFO("Qwen Image 2.1: first-block cache on (threshold %.4g; the first %d and the last %d steps computed fully; at most %d skipped steps in a row; head %d, tail %d blocks)",
                         block_cache.threshold, block_cache.first_full, block_cache.last_full, block_cache.max_skips, block_cache.head, block_cache.tail);
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

        bool owns_prefix_entry(const std::string& name) const {
            return std::any_of(retained_prefixes.begin(), retained_prefixes.end(), [&](const RetainedPrefix& prefix) {
                return starts_with(name, prefix.name + ".");
            });
        }

        bool retain_cache_entry(const std::string& name) const override {
            return keep_prefixes() && owns_prefix_entry(name);
        }

        // Frees the K/V of the prefixes the current run has not used.
        void drop_unused_prefixes() {
            auto unused = std::remove_if(retained_prefixes.begin(), retained_prefixes.end(),
                                         [](const RetainedPrefix& prefix) { return !prefix.used; });
            if (unused == retained_prefixes.end()) {
                return;
            }
            retained_prefixes.erase(unused, retained_prefixes.end());
            // (the first-block cache's entries are this run's: they stay until its end)
            cache_.retain([this](const std::string& name) { return owns_prefix_entry(name) || starts_with(name, BLOCK_CACHE_ENTRY); });
            sync_runtime_residency();
        }

        // The cache name for this run's prefix id: the prefix computed earlier from the same inputs (bit for bit, the
        // same graph shapes and settings), else a new one, for which the prefixes this run does not use make room
        // (with CFG, a changed prompt then drops the unchanged negative prompt's prefix too). first_use: the id's
        // first compute in this run.
        std::string resolve_prefix(uint64_t prefix_id,
                                   const sd::Tensor<float>& context,
                                   const sd::Tensor<int32_t>& image_slots,
                                   const std::vector<sd::Tensor<float>>& refs,
                                   const sd::Tensor<float>& x,
                                   bool* first_use) {
            auto run   = run_prefixes.find(prefix_id);
            *first_use = run == run_prefixes.end();
            if (!*first_use) {
                return run->second;
            }
            auto same = [](const auto& a, const auto& b) {
                return a.shape() == b.shape() && a.numel() == b.numel() &&
                       (a.numel() == 0 || std::memcmp(a.data(), b.data(), static_cast<size_t>(a.numel()) * sizeof(*a.data())) == 0);
            };
            const std::array<float, 2> scales = {linear_scale, attn_scale};
            const std::array<bool, 4> flags   = {circular_x_enabled, circular_y_enabled, flash_attn_enabled, sage_attn_enabled};

            auto entry = std::find_if(retained_prefixes.begin(), retained_prefixes.end(), [&](const RetainedPrefix& prefix) {
                if (prefix.x_shape != x.shape() || prefix.scales != scales || prefix.flags != flags ||
                    prefix.refs.size() != refs.size() || !same(prefix.context, context) || !same(prefix.image_slots, image_slots)) {
                    return false;
                }
                for (size_t i = 0; i < refs.size(); ++i) {
                    if (!same(prefix.refs[i], refs[i])) {
                        return false;
                    }
                }
                return true;
            });
            if (entry == retained_prefixes.end()) {
                drop_unused_prefixes();
                RetainedPrefix prefix;
                prefix.name = "qwen_image_2_1.prefix." + std::to_string(next_prefix_serial++);
                if (keep_prefixes()) {
                    prefix.context     = context;
                    prefix.image_slots = image_slots;
                    prefix.refs        = refs;
                    prefix.x_shape     = x.shape();
                    prefix.scales      = scales;
                    prefix.flags       = flags;
                }
                retained_prefixes.push_back(std::move(prefix));
                entry = std::prev(retained_prefixes.end());
            }
            entry->used             = true;
            run_prefixes[prefix_id] = entry->name;
            return entry->name;
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
                // A new sampling run (sample() ends the runner after each): its prefix ids start again, and the
                // prefixes the previous run did not use go.
                prefix_cache_disabled = false;
                prefix_cache_auto_f32 = false;
                run_prefixes.clear();
                if (keep_prefixes()) {
                    drop_unused_prefixes();
                } else {
                    retained_prefixes.clear();
                }
                for (auto& prefix : retained_prefixes) {
                    prefix.used = false;
                }
                // The first-block cache starts again too (its tensors went at the last run's end; this is a safeguard,
                // as its keys, the prefix ids, start again). This run's setting: the product form's (the environment
                // variable, read now, else the model arg) with its schedule for this run's step count, else the research
                // setting.
                block_cache_states.clear();
                block_cache_disabled           = false;
                block_cache_unavailable_logged = false;
                run_cache                      = block_cache;
                run_first_block_cache          = false;
                FirstBlockCacheConfig product  = first_block_cache;
                const char* product_env        = std::getenv("SD_QWEN_FIRST_BLOCK_CACHE");
                if (product_env != nullptr && product_env[0] != '\0' && !parse_first_block_cache(product_env, config.num_layers, product)) {
                    LOG_WARN("ignoring invalid SD_QWEN_FIRST_BLOCK_CACHE '%s'", product_env);
                }
                const int run_steps = extra != nullptr ? extra->steps : 0;
                const auto schedule = first_block_cache_schedule(product.fraction, run_steps);
                if (!schedule.empty()) {
                    run_cache             = BlockCacheConfig();
                    run_cache.steps       = schedule;
                    run_cache.tail        = product.tail >= 0 ? product.tail : static_cast<int>(std::floor(config.num_layers * 0.25 + 0.5));
                    run_first_block_cache = true;
                    std::string list;
                    for (int step : schedule) {
                        list += (list.empty() ? "" : ",") + std::to_string(step);
                    }
                    LOG_INFO("Qwen Image 2.1: first-block cache: %zu of %d steps reuse the cached blocks 1..%d (tail %d recomputed): %s",
                             schedule.size(), run_steps, config.num_layers - 1 - run_cache.tail, run_cache.tail, list.c_str());
                } else if (product.fraction > 0.f) {
                    LOG_INFO("Qwen Image 2.1: first-block cache: no step of a %d-step run reuses the cache", run_steps);
                }
                if (run_cache.enabled()) {
                    cache_.retain([](const std::string& name) { return !starts_with(name, BLOCK_CACHE_ENTRY); });
                    sync_runtime_residency();
                }
            }
            QwenImage21PrefixCache cache;
            cache.f16_target_keys = prefix_cache_keep == PrefixKeep::ON && sage_attn_enabled;
            bool first_use        = false;
            if (prefix_cache_enabled && !prefix_cache_disabled && extra != nullptr && extra->prefix_id != 0 && layout.prefix_length > 0) {
                cache.name          = resolve_prefix(extra->prefix_id, context, tensor_or_empty(extra->image_slots), refs, x, &first_use);
                cache.prefix_length = layout.prefix_length;
                if (prefix_cache_type != GGML_TYPE_COUNT) {
                    cache.type = prefix_cache_type;
                } else if (!prefix_cache_auto_f32 && (flash_attn_enabled || sage_attn_enabled) &&
                           (attn_scale <= 0.f || attn_scale == 1.f)) {
                    // F16, as flash attention casts K and V to F16 and the sage op reads V in F16 and quantizes K from
                    // it (half the memory of F32). A graph whose attention takes neither recomputes in F32 (below).
                    cache.type = GGML_TYPE_F16;
                }
                cache.mode = has_prefix_cache(cache) ? QwenImage21PrefixCache::Mode::REUSE : QwenImage21PrefixCache::Mode::STORE;
                if (first_use && cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                    LOG_VERBOSE("Qwen Image 2.1: reusing the prefix of an earlier picture (%" PRId64 " tokens, %s)",
                                layout.prefix_length, ggml_type_name(cache.type));
                }
            } else {
                drop_unused_prefixes();
            }
            // The first-block cache for this call (see BlockCacheConfig): it needs the step numbers and the condition's id.
            uint64_t block_id = 0;
            if (run_cache.enabled() && !block_cache_disabled) {
                if (extra != nullptr && extra->prefix_id != 0 && extra->step > 0 && extra->steps > 0) {
                    block_id = extra->prefix_id;
                } else if (!block_cache_unavailable_logged) {
                    LOG_INFO("Qwen Image 2.1: first-block cache off for this sampling run (the sampler gave no step numbers or condition ids)");
                    block_cache_unavailable_logged = true;
                }
            }
            const int64_t target_length = static_cast<int64_t>(layout.positions.size()) - layout.prefix_length;
            bool block_compare          = false;  // graph A: r1_prev is cached, compare with it
            bool flash_attn_used        = true;
            auto run                    = [&](const QwenImage21PrefixCache& active_cache, BlockCacheGraph stage) {
                flash_attn_used    = true;
                auto checked_cache = active_cache;
                if (prefix_cache_type == GGML_TYPE_COUNT && active_cache.type == GGML_TYPE_F16) {
                    checked_cache.flash_attn_used = &flash_attn_used;
                }
                const bool cached         = active_cache.mode == QwenImage21PrefixCache::Mode::REUSE;
                const auto first_position = layout.positions.begin() + (cached ? layout.prefix_length : 0);
                const size_t offset       = cached ? static_cast<size_t>(layout.prefix_length) : 0;
                std::vector<Rope::SpatialRegion> regions;
                for (auto region : layout.rope_layout.images) {
                    if (region.begin >= offset) {
                        region.begin -= offset;
                        regions.push_back(region);
                    }
                }
                // The rotary table depends only on the token positions, the image regions and the circular flags,
                // which stay the same for every step of a sampling run (and for every picture of the same prompt
                // layout): compute it once per such key (~7 ms of CPU work per forward, during which the GPU sat
                // idle) instead of once per forward. Same values: the same computation on the same inputs.
                // (a double holds every float, size and flag here exactly)
                std::vector<double> key = {static_cast<double>(circular_x_enabled), static_cast<double>(circular_y_enabled),
                                           static_cast<double>(regions.size()), static_cast<double>(layout.positions.end() - first_position)};
                for (const auto& region : regions) {
                    key.insert(key.end(), {static_cast<double>(region.begin), static_cast<double>(region.count), region.height_period,
                                           region.width_period, static_cast<double>(region.height_axis), static_cast<double>(region.width_axis)});
                }
                for (auto it = first_position; it != layout.positions.end(); ++it) {
                    key.push_back(static_cast<double>(it->size()));
                    key.insert(key.end(), it->begin(), it->end());
                }
                RopeTable* table = nullptr;
                for (auto& candidate : rope_tables) {
                    if (!candidate.values.empty() && candidate.key == key) {
                        table = &candidate;
                        break;
                    }
                }
                if (table == nullptr) {
                    table           = &rope_tables[next_rope_table];
                    next_rope_table = (next_rope_table + 1) % rope_tables.size();
                    table->key.clear();
                    Rope::Embedding embedding;
                    embedding.ids.assign(first_position, layout.positions.end());
                    embedding.positions.token_count = embedding.ids.size();
                    embedding.positions.images      = regions;
                    embedding.values                = Rope::embed_nd(embedding.ids, 1, 10000.f, config.axes_dim, embedding.layout, &embedding.frequencies);
                    table->values                   = finish_rope_pe(std::move(embedding));
                    table->key                      = std::move(key);
                }
                const std::vector<float>& pe_data = table->values;
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
                    // the drawn picture's tokens of a tensor over the whole sequence (they come last; a view)
                    auto target = [&](ggml_tensor* t) {
                        return t->ne[1] == target_length ? t : ggml_ext_slice(ctx.ggml_ctx, t, 1, t->ne[1] - target_length, t->ne[1], false);
                    };
                    ggml_tensor* out = nullptr;
                    switch (stage) {
                        case BlockCacheGraph::NONE:
                            out = model.forward(&ctx, make_input(x), make_input(*inputs.timesteps), cached ? nullptr : make_input(context),
                                                ref_inputs, pe, layout, masks, checked_cache);
                            break;
                        case BlockCacheGraph::FIRST_BLOCK: {
                            ggml_tensor* r1 = nullptr;
                            auto hf         = model.forward_first_blocks(&ctx, make_input(x), make_input(*inputs.timesteps), cached ? nullptr : make_input(context),
                                                                         ref_inputs, pe, layout, masks, checked_cache, run_cache.head,
                                                                         [&](ggml_tensor* h0, ggml_tensor* h1) {
                                                                             r1 = ggml_sub(ctx.ggml_ctx, target(h1), target(h0));
                                                                             ctx.persist_cache_tensor(block_cache_entry(block_id, "r1"), r1);
                                                                         });
                            ctx.persist_cache_tensor(block_cache_entry(block_id, "hf"), hf);
                            // (the committed r1 of the previous step; the one staged above replaces it at the graph's end)
                            auto r1_prev = block_compare ? ctx.load_cache_tensor(block_cache_entry(block_id, "r1")) : nullptr;
                            if (r1_prev != nullptr) {
                                auto change = ggml_sum(ctx.ggml_ctx, ggml_abs(ctx.ggml_ctx, ggml_sub(ctx.ggml_ctx, r1, r1_prev)));
                                auto size   = ggml_sum(ctx.ggml_ctx, ggml_abs(ctx.ggml_ctx, r1_prev));
                                out         = ggml_concat(ctx.ggml_ctx, change, size, 0);
                            } else {
                                out = ggml_sum(ctx.ggml_ctx, ggml_abs(ctx.ggml_ctx, r1));
                            }
                            break;
                        }
                        case BlockCacheGraph::REMAINING_BLOCKS: {
                            auto hf = ctx.load_cache_tensor(block_cache_entry(block_id, "hf"));
                            GGML_ASSERT(hf != nullptr);
                            if (run_cache.tail == 0) {
                                ggml_tensor* time = nullptr;
                                auto last         = model.forward_remaining_blocks(&ctx, hf, make_input(*inputs.timesteps), pe, layout, masks, checked_cache,
                                                                                   run_cache.head, &time);
                                // D before the head, so that the head's output stays the graph's last node
                                ctx.persist_cache_tensor(block_cache_entry(block_id, "d"), ggml_sub(ctx.ggml_ctx, last, target(hf)));
                                out = model.head(&ctx, last, time, make_input(x));
                            } else {
                                const int tail_begin = model.num_layers() - run_cache.tail;
                                auto step            = model.step_inputs(&ctx, make_input(*inputs.timesteps), pe, layout, checked_cache);
                                auto mid             = model.run_blocks(&ctx, hf, step, layout, masks, checked_cache, run_cache.head, tail_begin);
                                ctx.persist_cache_tensor(block_cache_entry(block_id, "d"), ggml_sub(ctx.ggml_ctx, mid, hf));
                                auto last = model.run_blocks(&ctx, mid, step, layout, masks, checked_cache, tail_begin, model.num_layers());
                                out       = model.head(&ctx, model.drawn_tokens(&ctx, last, layout, checked_cache), step.time, make_input(x));
                            }
                            break;
                        }
                        case BlockCacheGraph::HEAD: {
                            auto hf = ctx.load_cache_tensor(block_cache_entry(block_id, "hf"));
                            auto d  = ctx.load_cache_tensor(block_cache_entry(block_id, "d"));
                            GGML_ASSERT(hf != nullptr && d != nullptr);
                            if (run_cache.tail == 0) {
                                out = model.forward_head(&ctx, ggml_add(ctx.ggml_ctx, target(hf), d), make_input(*inputs.timesteps), make_input(x));
                            } else {
                                // (a REUSE step holds the drawn picture's tokens only; a D cached by a STORE step, the
                                // whole sequence, gives them its last ones)
                                auto step = model.step_inputs(&ctx, make_input(*inputs.timesteps), pe, layout, checked_cache);
                                auto mid  = ggml_add(ctx.ggml_ctx, hf, d->ne[1] == hf->ne[1] ? d : target(d));
                                auto last = model.run_blocks(&ctx, mid, step, layout, masks, checked_cache, model.num_layers() - run_cache.tail, model.num_layers());
                                out       = model.head(&ctx, model.drawn_tokens(&ctx, last, layout, checked_cache), step.time, make_input(x));
                            }
                            break;
                        }
                    }
                    if (!flash_attn_used) {
                        return static_cast<ggml_cgraph*>(nullptr);
                    }
                    ggml_build_forward_expand(graph, out);
                    return graph;
                };
                return restore_trailing_singleton_dims(GGMLRunner::compute(build, n_threads, false), x.dim());
            };
            // A step with the first-block cache: graph A, then B (computed fully) or C (skipped), decided here.
            auto run_block_cached = [&](const QwenImage21PrefixCache& active_cache) {
                auto& state       = block_cache_states[block_id];
                const int step    = extra->step;
                const int steps   = extra->steps;
                auto cached_shape = [&](const char* tensor) {
                    auto t = get_cache_tensor_by_name(block_cache_entry(block_id, tensor));
                    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] == config.hidden_size && t->ne[1] == target_length &&
                           t->ne[2] == 1 && t->ne[3] == 1;
                };
                // D fits this step: the drawn picture's tokens without a tail; with one, the sequence this step holds,
                // or, in REUSE, any D ending with the drawn picture's tokens
                auto usable_d = [&]() {
                    if (run_cache.tail == 0) {
                        return cached_shape("d");
                    }
                    auto t = get_cache_tensor_by_name(block_cache_entry(block_id, "d"));
                    const int64_t length = active_cache.mode == QwenImage21PrefixCache::Mode::REUSE
                                               ? target_length
                                               : static_cast<int64_t>(layout.positions.size());
                    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] == config.hidden_size && t->ne[2] == 1 && t->ne[3] == 1 &&
                           (t->ne[1] == length || (length == target_length && t->ne[1] > target_length));
                };
                block_compare        = cached_shape("r1");
                const int64_t t0     = ggml_time_us();
                auto sums            = run(active_cache, BlockCacheGraph::FIRST_BLOCK);
                const int64_t t1     = ggml_time_us();
                if (sums.empty()) {
                    return sums;
                }
                float change = -1.f;  // nothing to compare with
                if (block_compare && sums.numel() == 2) {
                    change = sums.data()[1] > 0.f ? sums.data()[0] / sums.data()[1] : INFINITY;
                }
                const bool listed = !run_cache.steps.empty();
                const bool forced = listed || step <= run_cache.first_full || step > steps - run_cache.last_full ||
                                    active_cache.mode == QwenImage21PrefixCache::Mode::STORE;
                bool skip = false;
                if (change >= 0.f && std::isfinite(change)) {
                    state.accumulated += change;
                    skip = !forced && state.accumulated < run_cache.threshold &&
                           state.skips_in_row < run_cache.max_skips && usable_d();
                }
                if (listed && std::find(run_cache.steps.begin(), run_cache.steps.end(), step) != run_cache.steps.end()) {
                    skip = active_cache.mode != QwenImage21PrefixCache::Mode::STORE && usable_d();
                    if (!skip) {
                        LOG_INFO("Qwen Image 2.1 first-block cache: step %d is on the list but computed fully (%s)", step,
                                 active_cache.mode == QwenImage21PrefixCache::Mode::STORE ? "it fills the prefix cache" : "no cached residual yet");
                    }
                }
                // Fewer of these tensors on the card at once: B makes a new D, so the old one goes before it, and hf is
                // only needed until the step's second graph has run.
                auto drop = [&](const char* tensor) {
                    const auto name = block_cache_entry(block_id, tensor);
                    if (get_cache_tensor_by_name(name) != nullptr) {
                        cache_.retain([&](const std::string& entry) { return entry != name; });
                        sync_runtime_residency();
                    }
                };
                if (!skip) {
                    drop("d");
                }
                auto result      = run(active_cache, skip ? BlockCacheGraph::HEAD : BlockCacheGraph::REMAINING_BLOCKS);
                const int64_t t2 = ggml_time_us();
                drop("hf");
                if (result.empty()) {
                    return result;
                }
                LOG_VERBOSE("Qwen Image 2.1 first-block cache (condition %" PRIu64 "): step %d/%d, change %.4f, since the last full step %.4f -> %s (graph A %.0f ms, %s %.0f ms)",
                            block_id, step, steps, change, state.accumulated,
                            skip ? "skipped" : (forced && !listed ? "computed (always)" : "computed"),
                            (t1 - t0) / 1000.0, skip ? "C" : "B", (t2 - t1) / 1000.0);
                char text[32];
                if (change >= 0.f) {
                    snprintf(text, sizeof(text), " %d:%.4f%s", step, change, skip ? "*" : "");
                } else {
                    snprintf(text, sizeof(text), " %d:-", step);
                }
                state.indicators += text;
                if (skip) {
                    ++state.skips_in_row;
                    state.skipped_steps.push_back(step);
                } else {
                    state.accumulated  = 0.f;
                    state.skips_in_row = 0;
                }
                if (step >= steps && !state.reported) {
                    std::string skipped;
                    for (int s : state.skipped_steps) {
                        skipped += (skipped.empty() ? "" : " ") + std::to_string(s);
                    }
                    const auto level = run_first_block_cache ? SD_LOG_VERBOSE : SD_LOG_INFO;
                    log_printf(level, __FILE__, __LINE__,
                               "Qwen Image 2.1 first-block cache (condition %" PRIu64 ", %s %.4g, head %d, tail %d): %zu of %d steps skipped (%s); the change per step (* skipped):%s",
                               block_id, listed ? "a step list, threshold" : "threshold", run_cache.threshold, run_cache.head, run_cache.tail,
                               state.skipped_steps.size(), steps, skipped.c_str(), state.indicators.c_str());
                    state.reported = true;
                }
                return result;
            };
            auto run_step = [&](const QwenImage21PrefixCache& active_cache) {
                return block_id != 0 ? run_block_cached(active_cache) : run(active_cache, BlockCacheGraph::NONE);
            };
            auto result = run_step(cache);
            if (result.empty() && !flash_attn_used) {
                // Casting an F16 cache back to F32 cannot recover its original values.
                // Recompute the prefix before executing a graph that falls back from FA.
                free_cache_ctx_and_buffer();
                prefix_cache_auto_f32 = true;
                cache.type            = GGML_TYPE_F32;
                cache.mode            = QwenImage21PrefixCache::Mode::STORE;
                LOG_DEBUG("Qwen Image 2.1: Flash Attention unavailable; using F32 prefix caching for this sampling run");
                result = run_step(cache);
            }
            if (result.empty() && last_compute_status() == GGML_STATUS_ALLOC_FAILED &&
                (cache.mode != QwenImage21PrefixCache::Mode::NONE || !cache_.empty())) {
                // The failed graph has ended before persistent inputs are released.
                free_cache_ctx_and_buffer();
                prefix_cache_disabled = true;
                if (block_id != 0) {
                    block_cache_disabled = true;
                    LOG_WARN("Qwen Image 2.1: insufficient memory for the first-block cache; retrying without it (and without prefix caching) for this sampling run");
                } else {
                    LOG_WARN("Qwen Image 2.1: insufficient memory for prefix caching; retrying without it for this sampling run");
                }
                QwenImage21PrefixCache uncached;
                uncached.f16_target_keys = cache.f16_target_keys;
                return run(uncached, BlockCacheGraph::NONE);
            }
            if (!result.empty() && cache.mode == QwenImage21PrefixCache::Mode::STORE) {
                if (!has_prefix_cache(cache)) {
                    free_cache_ctx_and_buffer();
                    prefix_cache_disabled = true;
                    LOG_WARN("Qwen Image 2.1: incomplete prefix cache; disabling it for this sampling run");
                } else {
                    LOG_DEBUG("Qwen Image 2.1: cached prefix %s (%" PRId64 " tokens, %s)", cache.name.c_str(), layout.prefix_length, ggml_type_name(cache.type));
                    // The STORE graph's compute buffer is the run's largest (3,617 MB with one 1024x1024 example
                    // picture, 709 MB for the steps that reuse the prefix). Kept, it counted as this runner's memory
                    // in those steps' fit check and pushed them into layer segments where the budget is tight (8 GB
                    // cards with a 668x668 example: 34 segments, 1.83 s a step instead of 1.54 s in one piece; two
                    // example pictures on 12 GB: 3.19 s instead of 2.90 s). The same values either way.
                    workspace_.release();
                    sync_runtime_residency();
                }
            }
            return result;
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
