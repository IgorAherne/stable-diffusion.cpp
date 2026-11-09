#include "core/runner_cache.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "core/ggml_graph_cut.h"
#include "core/util.h"

namespace sd {
    static std::unordered_set<const ggml_tensor*> cache_graph_tensors(ggml_cgraph* graph) {
        std::unordered_set<const ggml_tensor*> tensors;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            tensors.insert(ggml_graph_node(graph, i));
        }
        for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
            tensors.insert(ggml_graph_cut::leaf_tensor(graph, i));
        }
        return tensors;
    }

    CachedTensor::~CachedTensor() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }

    std::unique_ptr<CachedTensor> CachedTensor::copy(ggml_backend_t backend,
                                                     const std::string& name,
                                                     ggml_tensor* source,
                                                     ggml_status& status,
                                                     ggml_backend_buffer_t reuse) {
        status = GGML_STATUS_FAILED;
        if (ggml_graph_cut::tensor_buffer(source) == nullptr) {
            return nullptr;
        }
        status         = GGML_STATUS_ALLOC_FAILED;
        auto entry     = std::make_unique<CachedTensor>();
        entry->context = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        if (entry->context == nullptr) {
            return nullptr;
        }
        entry->tensor = ggml_dup_tensor(entry->context, source);
        // Cut views are rebound with their original strides and offsets.
        std::copy(std::begin(source->nb), std::end(source->nb), std::begin(entry->tensor->nb));
        ggml_set_name(entry->tensor, name.c_str());
        if (reuse != nullptr) {
            // (the caller checked its type and size; its old tensor's context is gone)
            if (ggml_backend_tensor_alloc(reuse, entry->tensor, ggml_backend_buffer_get_base(reuse)) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(reuse);
                return nullptr;
            }
            entry->buffer = reuse;
        } else {
            entry->buffer = ggml_backend_alloc_ctx_tensors(entry->context, backend);
        }
        if (entry->buffer == nullptr) {
            return nullptr;
        }
        if (source->view_src != nullptr || !ggml_is_contiguous(source) || source->buffer == nullptr) {
            std::vector<uint8_t> data(ggml_nbytes(source));
            ggml_backend_tensor_get(source, data.data(), 0, data.size());
            ggml_backend_tensor_set(entry->tensor, data.data(), 0, data.size());
        } else {
            // On the backend's stream, after the graph that wrote the source; both callers synchronize once after
            // all their copies (a blocking copy waited for each, ~2 ms under WDDM, three a segment of a cut graph).
            ggml_backend_tensor_copy_async(backend, backend, source, entry->tensor);
        }
        status = GGML_STATUS_SUCCESS;
        return entry;
    }

    static ggml_tensor* cached_tensor(const CachedTensors& tensors, const std::string& name) {
        auto entry = tensors.find(name);
        return entry == tensors.end() ? nullptr : entry->second->tensor;
    }

    static size_t resident_bytes(const CachedTensors& tensors, ggml_backend_dev_t device) {
        size_t bytes = 0;
        for (const auto& entry : tensors) {
            auto buffer = entry.second->buffer;
            if (!ggml_backend_buffer_is_host(buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) == device) {
                const size_t size = ggml_backend_buffer_get_size(buffer);
                bytes             = size > SIZE_MAX - bytes ? SIZE_MAX : bytes + size;
            }
        }
        return bytes;
    }

    ggml_tensor* RunnerCache::get(const std::string& name) const {
        return cached_tensor(committed_, name);
    }

    void RunnerCache::stage(const std::string& name, ggml_tensor* tensor) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
            outputs_[name] = tensor;
        }
    }

    size_t RunnerCache::pending_bytes(ggml_cgraph* graph) const {
        if (outputs_.empty()) {
            return 0;
        }
        auto tensors = cache_graph_tensors(graph);
        auto buft    = ggml_backend_get_default_buffer_type(backend_);
        size_t bytes = 0;
        for (const auto& output : outputs_) {
            if (pending_.count(output.first) || !tensors.count(output.second)) {
                continue;
            }
            const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, output.second),
                                         ggml_backend_buft_get_alignment(buft));
            bytes             = size > SIZE_MAX - bytes ? SIZE_MAX : bytes + size;
        }
        return bytes;
    }

    size_t RunnerCache::resident_bytes(ggml_backend_dev_t device) const {
        const size_t committed = sd::resident_bytes(committed_, device);
        const size_t pending   = sd::resident_bytes(pending_, device);
        return pending > SIZE_MAX - committed ? SIZE_MAX : committed + pending;
    }

    ggml_status RunnerCache::capture(ggml_cgraph* graph) {
        if (outputs_.empty()) {
            return GGML_STATUS_SUCCESS;
        }
        const auto tensors = cache_graph_tensors(graph);
        for (const auto& output : outputs_) {
            if (pending_.count(output.first) || !tensors.count(output.second)) {
                continue;
            }
            GGML_ASSERT(ggml_is_contiguous(output.second));
            ggml_status status;
            auto entry = CachedTensor::copy(backend_, output.first, output.second, status);
            if (entry == nullptr) {
                return status;
            }
            pending_[output.first] = std::move(entry);
        }
        ggml_backend_synchronize(backend_);
        return GGML_STATUS_SUCCESS;
    }

    void RunnerCache::graph_end(bool success) {
        // Graph inputs can still reference the previous generation until graph end.
        if (success) {
            for (auto& entry : pending_) {
                committed_[entry.first] = std::move(entry.second);
            }
        }
        pending_.clear();
        outputs_.clear();
    }

    void RunnerCache::clear() {
        graph_end(false);
        committed_.clear();
    }

    void RunnerCache::retain(const std::function<bool(const std::string&)>& keep) {
        for (auto it = committed_.begin(); it != committed_.end();) {
            it = keep && keep(it->first) ? std::next(it) : committed_.erase(it);
        }
    }

    ggml_tensor* GraphCutTensorCache::get(const std::string& name) const {
        return cached_tensor(tensors_, name);
    }

    size_t GraphCutTensorCache::resident_bytes(ggml_backend_dev_t device) const {
        size_t bytes = sd::resident_bytes(tensors_, device);
        for (const auto& spare : spare_) {
            auto buffer = spare->buffer;
            if (buffer != nullptr && !ggml_backend_buffer_is_host(buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) == device) {
                const size_t size = ggml_backend_buffer_get_size(buffer);
                bytes             = size > SIZE_MAX - bytes ? SIZE_MAX : bytes + size;
            }
        }
        return bytes;
    }

    size_t GraphCutTensorCache::estimate_output_bytes(
        ggml_cgraph* graph,
        const ggml_graph_cut::Segment& segment) const {
        ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend_);
        if (buffer_type == nullptr) {
            return SIZE_MAX;
        }
        const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
        size_t total_size      = 0;
        for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
            ggml_tensor* output = ggml_graph_cut::output_tensor(graph, segment, output_idx);
            if (output == nullptr || !ggml_graph_cut::is_graph_cut_tensor(output) ||
                !segment.future_cut_names.count(output->name)) {
                continue;
            }
            ggml_tensor* source      = ggml_graph_cut::cache_source_tensor(output);
            const size_t tensor_size = GGML_PAD(
                ggml_backend_buft_get_alloc_size(buffer_type, source), alignment);
            total_size = tensor_size > SIZE_MAX - total_size ? SIZE_MAX : total_size + tensor_size;
        }
        return total_size;
    }

    void GraphCutTensorCache::prune(const std::unordered_set<std::string>& keep_names) {
        for (auto it = tensors_.begin(); it != tensors_.end();) {
            if (keep_names.count(it->first)) {
                ++it;
                continue;
            }
            if (spare_.size() < 2 && it->second != nullptr && it->second->buffer != nullptr &&
                !ggml_backend_buffer_is_host(it->second->buffer)) {
                spare_.push_back(std::move(it->second));
            }
            it = tensors_.erase(it);
        }
    }

    ggml_status GraphCutTensorCache::capture(ggml_cgraph* graph,
                                             const ggml_graph_cut::Segment& segment,
                                             const char* log_desc) {
        size_t copied_bytes = 0;
        size_t copied_count = 0;
        for (int index : segment.output_node_indices) {
            auto output = ggml_graph_node(graph, index);
            if (!ggml_graph_cut::is_graph_cut_tensor(output) ||
                !segment.future_cut_names.count(output->name)) {
                continue;
            }
            ggml_status status;
            ggml_tensor* source         = ggml_graph_cut::cache_source_tensor(output);
            ggml_backend_buffer_t reuse = nullptr;
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
            const size_t need           = buft != nullptr ? ggml_backend_buft_get_alloc_size(buft, source) : 0;
            for (auto spare = spare_.begin(); spare != spare_.end() && need > 0; ++spare) {
                ggml_backend_buffer_t buffer = (*spare)->buffer;
                if (ggml_backend_buffer_get_type(buffer) == buft &&
                    ggml_backend_buffer_get_size(buffer) == GGML_PAD(need, ggml_backend_buft_get_alignment(buft))) {
                    reuse              = buffer;
                    (*spare)->buffer   = nullptr;  // (~CachedTensor frees its context only)
                    spare_.erase(spare);
                    break;
                }
            }
            auto entry = CachedTensor::copy(backend_, output->name, source, status, reuse);
            if (entry == nullptr) {
                LOG_ERROR("%s failed to capture graph cut tensor: %s", log_desc, output->name);
                return status;
            }
            const size_t size = ggml_backend_buffer_get_size(entry->buffer);
            copied_bytes      = size > SIZE_MAX - copied_bytes ? SIZE_MAX : copied_bytes + size;
            ++copied_count;
            tensors_[output->name] = std::move(entry);
        }
        ggml_backend_synchronize(backend_);
        if (copied_count > 0) {
            LOG_DEBUG("%s graph cut cache added %6.2f MB (%zu tensors)",
                      log_desc, copied_bytes / (1024.f * 1024.f), copied_count);
        }
        return GGML_STATUS_SUCCESS;
    }
}
