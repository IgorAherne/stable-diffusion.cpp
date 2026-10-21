#include "model_manager.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/util.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// Weight staging from pageable host memory (the model files' mappings) to a GPU.
//
// ggml_backend_tensor_copy from pageable memory is a cudaMemcpy that the driver stages through a small pinned buffer
// of its own, with one thread touching the source page by page. Measured with Qwen-Image 2.1's 4,005 MB of diffusion
// weights (RTX 3060, PCIe 4 x16, Windows 11, medians of 3): 0.31 s when the pages are in the process's working set,
// 1.2 s when they are in the file cache but not yet in the process (a million 4 KB soft page faults, one after
// another), 6.4 s when they are on the disk (the faults read the file in small clusters, one at a time). And the
// copies leave every mapped weight in the process's working set: 9 GB of Qwen-Image's, which Windows trims as soon
// as RAM runs short (a 16 GB PC), after which the next copy pays the page faults again.
//
// Here several threads fill a ring of page-locked slots, and each filled slot goes to the card by DMA while the next
// ones fill. By default they read the model files (ReadFile / pread through the system's file cache) instead of
// touching the mappings: 0.37 s from the file cache, 1.7 s from the disk (its sequential read speed), and the
// process's working set stays at 2.7 GB instead of 11 GB. Reading from the cache is bound by the RAM's bandwidth,
// which the copies share with the DMA (~10 GB/s here), so a copy from pages already in the working set is faster
// (0.29 s): SD_STAGING_RING=mmap fills the ring from the mappings, taking their page faults in parallel (0.36 s from
// the file cache), with one more thread paging the mapped ranges in ahead with PrefetchVirtualMemory
// (madvise(MADV_WILLNEED) elsewhere), which reads the file in large concurrent requests: 2.0 s from the disk, where
// the copy threads' own faults took 6.1 s. Sources that cannot be read from their file take that path too.

static constexpr size_t STAGING_SLOT_BYTES = 8u << 20;
// More slots than copy threads: every thread fills one while the DMA engine drains the others.
static constexpr size_t STAGING_SLOTS = 12;
// The mappings' page faults stop scaling there (4 threads 0.46 s, 8 0.35 s, 12 0.41 s); the file reads are bound by
// the RAM's bandwidth from 2 threads on (0.42-0.46 s with 2 to 8), and read the disk no faster with 16.
static constexpr int STAGING_MAX_THREADS     = 8;
static constexpr size_t STAGING_PIECE_ALIGN  = 256;
static constexpr size_t PREFETCH_PIECE_BYTES = 64u << 20;
static constexpr size_t PREFETCH_PROBE_STEP  = 1u << 20;
static constexpr int64_t PREFETCH_PROBE_US   = 40;  // a soft fault takes a few us, a read from the disk 60 us or more

using HostRange = std::pair<const uint8_t*, size_t>;

enum class StagingRingMode {
    Off,      // every weight copied by ggml_backend_tensor_copy, as before the ring
    Mapping,  // the ring, filled from the model files' mappings
    File,     // the ring, filled by reading the model files (the mappings where that cannot be done)
};

// SD_STAGING_RING: unset = File; "mmap" = Mapping (for comparisons); "off" = Off.
static StagingRingMode staging_ring_mode() {
    static const StagingRingMode mode = [] {
        const char* value = std::getenv("SD_STAGING_RING");
        if (value == nullptr) {
            return StagingRingMode::File;
        }
        std::string text = trim(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (text == "0" || text == "off" || text == "false" || text == "no") {
            return StagingRingMode::Off;
        }
        if (text == "mmap" || text == "mapping") {
            return StagingRingMode::Mapping;
        }
        if (!text.empty() && text != "1" && text != "on" && text != "file") {
            LOG_WARN("SD_STAGING_RING: unknown value '%s' ignored (off, mmap, file)", text.c_str());
        }
        return StagingRingMode::File;
    }();
    return mode;
}

static int staging_threads() {
    const unsigned hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        return 4;
    }
    return std::max(2, std::min(STAGING_MAX_THREADS, static_cast<int>(hardware / 2)));
}

static bool is_pinned_host_buft(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || !ggml_backend_buft_is_host(buft)) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    return dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

static size_t host_page_size() {
#ifdef _WIN32
    static const size_t size = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return static_cast<size_t>(info.dwPageSize);
    }();
#else
    static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    return size;
}

#ifdef _WIN32
// WIN32_MEMORY_RANGE_ENTRY; memoryapi.h declares it only for _WIN32_WINNT >= Windows 8.
struct PrefetchRangeEntry {
    void* address;
    size_t bytes;
};
using PrefetchVirtualMemoryFn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PrefetchRangeEntry*, ULONG);

// Resolved at run time, so that the library still loads on Windows 7.
static PrefetchVirtualMemoryFn prefetch_virtual_memory() {
    static const PrefetchVirtualMemoryFn fn = [] {
        HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        return kernel32 != nullptr
                   ? reinterpret_cast<PrefetchVirtualMemoryFn>(GetProcAddress(kernel32, "PrefetchVirtualMemory"))
                   : nullptr;
    }();
    return fn;
}
#endif

static void page_in(const HostRange& range) {
#ifdef _WIN32
    PrefetchVirtualMemoryFn fn = prefetch_virtual_memory();
    if (fn != nullptr) {
        PrefetchRangeEntry entry{const_cast<uint8_t*>(range.first), range.second};
        fn(GetCurrentProcess(), 1, &entry, 0);
    }
#elif defined(MADV_WILLNEED)
    madvise(const_cast<uint8_t*>(range.first), range.second, MADV_WILLNEED);
#else
    (void)range;
#endif
}

// Pages the mapped source ranges in ahead of the copies, piece by piece in copy order, from a thread of its own
// (PrefetchVirtualMemory returns when its reads are done). Pieces the copies have passed are skipped, and so are
// pieces already in RAM: prefetching pages from the file cache gains nothing and costs ~35 ms of kernel time per GB,
// which slowed the copies by 0.07 s per 4 GB. A piece counts as in RAM when touching one page per MB of it found
// each in RAM (a touch slower than PREFETCH_PROBE_US read the disk).
class MappedRangePrefetcher {
public:
    MappedRangePrefetcher(std::vector<HostRange> pieces, const std::atomic<uintptr_t>& copy_front)
        : pieces_(std::move(pieces)), copy_front_(copy_front) {
        if (pieces_.empty()) {
            return;
        }
        try {
            thread_ = std::thread([this] { run(); });
        } catch (const std::exception&) {
            // no prefetch: the copies fault the pages in themselves
        }
    }
    ~MappedRangePrefetcher() {
        stop_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    MappedRangePrefetcher(const MappedRangePrefetcher&)            = delete;
    MappedRangePrefetcher& operator=(const MappedRangePrefetcher&) = delete;

private:
    void run() {
        for (const HostRange& piece : pieces_) {
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            if (reinterpret_cast<uintptr_t>(piece.first + piece.second) <=
                copy_front_.load(std::memory_order_relaxed)) {
                continue;
            }
            bool on_disk = false;
            for (size_t offset = 0; offset < piece.second && !on_disk; offset += PREFETCH_PROBE_STEP) {
                const int64_t t0 = ggml_time_us();
                (void)*static_cast<const volatile uint8_t*>(piece.first + offset);
                on_disk = ggml_time_us() - t0 > PREFETCH_PROBE_US;
            }
            if (on_disk) {
                page_in(piece);
            }
        }
    }

    std::vector<HostRange> pieces_;
    const std::atomic<uintptr_t>& copy_front_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

class StagingRing {
public:
    StagingRing(ggml_backend_buffer_t buffer, std::vector<ggml_backend_event_t> events, size_t slot_bytes)
        : buffer_(buffer), events_(std::move(events)), slot_bytes_(slot_bytes) {}
    ~StagingRing() {
        for (ggml_backend_event_t event : events_) {
            ggml_backend_event_free(event);
        }
        ggml_backend_buffer_free(buffer_);
    }
    StagingRing(const StagingRing&)            = delete;
    StagingRing& operator=(const StagingRing&) = delete;

    size_t slots() const { return events_.size(); }
    size_t slot_bytes() const { return slot_bytes_; }
    uint8_t* slot(size_t index) const {
        return static_cast<uint8_t*>(ggml_backend_buffer_get_base(buffer_)) + index * slot_bytes_;
    }
    ggml_backend_event_t event(size_t index) const { return events_[index]; }

    static std::shared_ptr<StagingRing> create(ggml_backend_dev_t device) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (!is_pinned_host_buft(host_buft)) {
            return nullptr;
        }
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(host_buft, STAGING_SLOTS * STAGING_SLOT_BYTES);
        if (buffer == nullptr) {
            return nullptr;
        }
        if (ggml_backend_buffer_get_type(buffer) != host_buft) {
            ggml_backend_buffer_free(buffer);  // the system refused to lock it: pageable slots would gain nothing
            return nullptr;
        }
        std::vector<ggml_backend_event_t> events;
        for (size_t i = 0; i < STAGING_SLOTS; ++i) {
            ggml_backend_event_t event = ggml_backend_event_new(device);
            if (event == nullptr) {
                for (ggml_backend_event_t created : events) {
                    ggml_backend_event_free(created);
                }
                ggml_backend_buffer_free(buffer);
                return nullptr;
            }
            events.push_back(event);
        }
        return std::make_shared<StagingRing>(buffer, std::move(events), STAGING_SLOT_BYTES);
    }

private:
    ggml_backend_buffer_t buffer_;
    std::vector<ggml_backend_event_t> events_;
    size_t slot_bytes_;
};

void ModelManager::release_staging_rings() {
    staging_rings_.clear();
}

bool ModelManager::copy_through_staging_ring(ggml_backend_t compute_backend,
                                             ggml_backend_buffer_type_t staging_buft,
                                             const std::vector<std::pair<TensorState*, ggml_tensor*>>& tensors,
                                             std::vector<char>& copied) {
    // CUDA only: the path measured. Other GPU backends (Vulkan, ROCm, ...) keep the per-tensor copies.
    const StagingRingMode mode = staging_ring_mode();
    if (mode == StagingRingMode::Off || tensors.empty() || compute_backend == nullptr ||
        sd_backend_is_cpu(compute_backend) || !sd_backend_is(compute_backend, "CUDA")) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(compute_backend);
    if (device == nullptr) {
        return false;
    }
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(device, &props);
    if (!props.caps.async || !props.caps.events || !props.caps.host_buffer) {
        return false;
    }
    // The device's own buffer type only (not a split one): set_tensor_async requires it.
    if (staging_buft == nullptr || staging_buft != ggml_backend_get_default_buffer_type(compute_backend)) {
        return false;
    }

    // Sources in page-locked buffers keep their direct DMA; only contiguous tensors in pageable RAM come here.
    std::vector<size_t> picked;
    picked.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor* src = tensors[i].first != nullptr ? tensors[i].first->tensor : nullptr;
        const ggml_tensor* dst = tensors[i].second;
        if (src == nullptr || dst == nullptr || src->data == nullptr || src->buffer == nullptr ||
            dst->data == nullptr || dst->buffer == nullptr || dst->view_src != nullptr) {
            continue;
        }
        if (!ggml_backend_buffer_is_host(src->buffer) ||
            is_pinned_host_buft(ggml_backend_buffer_get_type(src->buffer))) {
            continue;
        }
        if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || ggml_nbytes(src) != ggml_nbytes(dst) ||
            ggml_nbytes(src) == 0) {
            continue;
        }
        picked.push_back(i);
    }
    if (picked.empty()) {
        return false;
    }
    ggml_backend_t transfer_backend = prefetch_backend_for(compute_backend);
    if (transfer_backend == nullptr || staging_buft != ggml_backend_get_default_buffer_type(transfer_backend)) {
        return false;
    }

    auto ring_entry = staging_rings_.find(device);
    if (ring_entry == staging_rings_.end()) {
        std::shared_ptr<StagingRing> ring = StagingRing::create(device);
        if (ring != nullptr) {
            LOG_VERBOSE("model manager staging ring: %zu x %.0f MB of page-locked memory (%s), %d copy threads, %s",
                        ring->slots(), ring->slot_bytes() / (1024.0 * 1024.0), ggml_backend_dev_name(device),
                        staging_threads(),
                        mode == StagingRingMode::File ? "reading the model files" : "copying from their mappings");
        } else {
            LOG_WARN(
                "model manager could not set up a page-locked staging ring for %s; weights in pageable RAM "
                "are copied to it one tensor at a time, which is slower",
                ggml_backend_dev_name(device));
        }
        ring_entry = staging_rings_.emplace(device, std::move(ring)).first;  // a failure is not retried
    }
    StagingRing* ring = ring_entry->second.get();
    if (ring == nullptr) {
        return false;
    }

    // In file order: the copies and the prefetch then walk the files front to back.
    auto source_of = [&](size_t index) {
        return static_cast<const uint8_t*>(tensors[index].first->tensor->data);
    };
    std::stable_sort(picked.begin(), picked.end(), [&](size_t a, size_t b) { return source_of(a) < source_of(b); });

    struct MappedFile {
        const MmapWrapper* file;
        const uint8_t* begin;
        size_t size;
    };
    std::vector<MappedFile> mappings;
    for (const auto& block : params_storage_blocks_) {
        for (const auto& store : block->mmap_tensor_stores) {
            if (store.mmapped != nullptr) {
                mappings.push_back({store.mmapped.get(), store.mmapped->data(), store.mmapped->size()});
            }
        }
    }
    auto mapping_of = [&](size_t index) -> const MappedFile* {
        const uint8_t* src = source_of(index);
        const size_t size  = ggml_nbytes(tensors[index].second);
        for (const MappedFile& mapping : mappings) {
            if (src >= mapping.begin && size <= mapping.size &&
                static_cast<size_t>(src - mapping.begin) <= mapping.size - size) {
                return &mapping;
            }
        }
        return nullptr;
    };
    // A tensor is read from its file when its bytes are the file's: it lies in a read-only mapping (a copy-on-write
    // one may carry LoRAs merged in place), and its one source is stored there unconverted, at the very offset the
    // tensor's address gives (the loader maps a tensor only when it needs no conversion, so this is a cross-check).
    auto reads_file = [&](size_t index, const MappedFile* mapping) {
        if (mode != StagingRingMode::File || mapping == nullptr || mapping->file->writable()) {
            return false;
        }
        const TensorState& state = *tensors[index].first;
        if (state.sources.size() != 1) {
            return false;
        }
        const TensorStorage& storage = state.sources.front();
        return storage.index_in_zip < 0 && !storage.is_f64 && !storage.is_i64 && storage.type == state.tensor->type &&
               storage.offset == static_cast<uint64_t>(source_of(index) - mapping->begin) &&
               storage.nbytes() == static_cast<int64_t>(ggml_nbytes(state.tensor));
    };

    // Pieces of at most one slot, packed into slot-sized batches; batch b fills slot b % slots.
    struct Piece {
        ggml_tensor* dst;
        size_t dst_offset;
        const uint8_t* src;
        size_t size;
        size_t slot_offset;
        const MmapWrapper* file;  // read from this file at file_offset; nullptr: copied from src
        size_t file_offset;
    };
    std::vector<Piece> pieces;
    std::vector<size_t> batch_begin;  // the pieces of batch b: [batch_begin[b], batch_begin[b + 1])
    std::vector<HostRange> mapped;    // the sources copied from a mapping, page-aligned and merged
    const size_t slot_bytes = ring->slot_bytes();
    const uintptr_t page    = host_page_size();
    size_t slot_used        = slot_bytes;
    for (size_t index : picked) {
        ggml_tensor* dst          = tensors[index].second;
        const uint8_t* src        = source_of(index);
        const size_t size         = ggml_nbytes(dst);
        const MappedFile* mapping = mapping_of(index);
        const MmapWrapper* file   = reads_file(index, mapping) ? mapping->file : nullptr;
        const size_t file_offset  = file != nullptr ? static_cast<size_t>(src - mapping->begin) : 0;
        for (size_t offset = 0; offset < size;) {
            if (slot_used >= slot_bytes) {
                batch_begin.push_back(pieces.size());
                slot_used = 0;
            }
            const size_t piece_size = std::min(size - offset, slot_bytes - slot_used);
            pieces.push_back({dst, offset, src + offset, piece_size, slot_used, file, file_offset + offset});
            offset += piece_size;
            slot_used = GGML_PAD(slot_used + piece_size, STAGING_PIECE_ALIGN);
        }
        if (mapping == nullptr || file != nullptr) {
            continue;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(src) & ~(page - 1);
        const uintptr_t end   = (reinterpret_cast<uintptr_t>(src) + size + page - 1) & ~(page - 1);
        if (!mapped.empty()) {
            const uintptr_t last_begin = reinterpret_cast<uintptr_t>(mapped.back().first);
            const uintptr_t last_end   = last_begin + mapped.back().second;
            if (begin >= last_begin && begin <= last_end) {
                mapped.back().second = std::max(last_end, end) - last_begin;
                continue;
            }
        }
        mapped.push_back({reinterpret_cast<const uint8_t*>(begin), end - begin});
    }
    const size_t n_batches = batch_begin.size();
    batch_begin.push_back(pieces.size());

    // The page-in thread is for the mappings only: file reads need no help to read the disk in large requests.
    std::vector<HostRange> prefetch_pieces;
    for (const HostRange& range : mapped) {
        for (size_t offset = 0; offset < range.second; offset += PREFETCH_PIECE_BYTES) {
            prefetch_pieces.push_back({range.first + offset, std::min(PREFETCH_PIECE_BYTES, range.second - offset)});
        }
    }

    // Workers fill the slots they are handed (file reads, or copies from a mapping); this thread queues each filled
    // slot's DMA in order, records the slot's event and hands the slot on to a later batch once that event has
    // passed. All backend calls stay on this thread.
    struct Pipeline {
        std::mutex mutex;
        std::condition_variable work_cv;
        std::condition_variable filled_cv;
        size_t handed = 0;  // batches [0, handed) may be filled
        size_t taken  = 0;  // the next batch for a worker
        std::vector<char> filled;
        bool stop = false;
    } pipeline;
    pipeline.filled.assign(n_batches, 0);
    const size_t n_slots = ring->slots();
    std::atomic<uintptr_t> copy_front{0};

    std::atomic<size_t> file_read_failures{0};
    auto worker = [&]() {
        // A reader of its own per file: positional reads on one shared handle would be serialized.
        std::vector<std::pair<const MmapWrapper*, std::unique_ptr<MmapWrapper::FileReader>>> readers;
        auto reader_for = [&](const MmapWrapper* file) {
            for (auto& entry : readers) {
                if (entry.first == file) {
                    return entry.second.get();
                }
            }
            std::unique_ptr<MmapWrapper::FileReader> reader;
            try {
                reader = file->open_reader();
                readers.emplace_back(file, std::move(reader));
            } catch (const std::exception&) {
                return static_cast<MmapWrapper::FileReader*>(nullptr);  // copied from the mapping instead
            }
            return readers.back().second.get();
        };
        for (;;) {
            size_t batch;
            {
                std::unique_lock<std::mutex> lock(pipeline.mutex);
                pipeline.work_cv.wait(lock, [&] { return pipeline.stop || pipeline.taken < pipeline.handed; });
                if (pipeline.taken >= pipeline.handed) {
                    return;
                }
                batch = pipeline.taken++;
            }
            uint8_t* slot = ring->slot(batch % n_slots);
            for (size_t p = batch_begin[batch]; p < batch_begin[batch + 1]; ++p) {
                const Piece& piece = pieces[p];
                if (piece.file != nullptr) {
                    MmapWrapper::FileReader* reader = reader_for(piece.file);
                    if (reader != nullptr && reader->read(slot + piece.slot_offset, piece.file_offset, piece.size)) {
                        continue;
                    }
                    file_read_failures.fetch_add(1, std::memory_order_relaxed);  // the same bytes, from the mapping
                }
                std::memcpy(slot + piece.slot_offset, piece.src, piece.size);
            }
            {
                std::lock_guard<std::mutex> lock(pipeline.mutex);
                pipeline.filled[batch] = 1;
            }
            pipeline.filled_cv.notify_one();
        }
    };
    auto hand = [&](size_t count) {
        {
            std::lock_guard<std::mutex> lock(pipeline.mutex);
            pipeline.handed = count;
        }
        copy_front.store(reinterpret_cast<uintptr_t>(pieces[batch_begin[count - 1]].src), std::memory_order_relaxed);
        pipeline.work_cv.notify_all();
    };

    MappedRangePrefetcher prefetcher(std::move(prefetch_pieces), copy_front);
    std::vector<std::thread> workers;
    const size_t n_workers = std::min(static_cast<size_t>(staging_threads()), n_batches);
    try {
        for (size_t i = 0; i < n_workers; ++i) {
            workers.emplace_back(worker);
        }
    } catch (const std::exception&) {
        // fewer workers
    }
    if (workers.empty()) {
        return false;
    }

    hand(std::min(n_slots, n_batches));
    for (size_t batch = 0; batch < n_batches; ++batch) {
        {
            std::unique_lock<std::mutex> lock(pipeline.mutex);
            pipeline.filled_cv.wait(lock, [&] { return pipeline.filled[batch] != 0; });
        }
        const size_t slot_index = batch % n_slots;
        uint8_t* slot           = ring->slot(slot_index);
        for (size_t p = batch_begin[batch]; p < batch_begin[batch + 1]; ++p) {
            ggml_backend_tensor_set_async(transfer_backend, pieces[p].dst, slot + pieces[p].slot_offset,
                                          pieces[p].dst_offset, pieces[p].size);
        }
        ggml_backend_event_record(ring->event(slot_index), transfer_backend);
        // Waits for the previous batch's DMA, not this one's, so that one stays queued and the copy engine does not
        // wait for this thread; its slot goes to batch (batch - 1 + n_slots).
        if (batch >= 1 && batch - 1 + n_slots < n_batches) {
            ggml_backend_event_synchronize(ring->event((batch - 1) % n_slots));
            hand(batch + n_slots);
        }
    }
    ggml_backend_synchronize(transfer_backend);
    {
        std::lock_guard<std::mutex> lock(pipeline.mutex);
        pipeline.stop = true;
    }
    pipeline.work_cv.notify_all();
    for (std::thread& thread : workers) {
        thread.join();
    }
    if (file_read_failures.load() > 0) {
        LOG_WARN("model manager could not read %zu pieces of the model files; they were copied from their mappings",
                 file_read_failures.load());
    }

    for (size_t index : picked) {
        copied[index] = 1;
    }
    return true;
}
