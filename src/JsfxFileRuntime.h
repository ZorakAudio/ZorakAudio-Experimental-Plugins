// SPDX-License-Identifier: Zlib
// Production file caches, background loader and sample registry for every code loader.
#pragma once
#include "DspJsfxSamplePool.h"
#include "JsfxSamplePoolOperations.h"
#include "JsfxSliderDeclarations.h"
#include "DspJsfxAudioFilePreflight.h"
#include "ZAAudioImportRecipe.h"
#include "JsfxSerialization.h"
#if DSPJSFX_HAS_TASKS
#include "JsfxTaskRuntimeHooks.h"
#endif
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>
namespace za::jsfx {
class FileRuntime : public SamplePoolOperations<FileRuntime> {
public:
    enum class FileState : int32_t
    {
        Unassigned   = 0,
        Loading      = 1,
        ReadyActive  = 2,
        ReadyPending = 3,
        PendingClear = 4,
        Error        = 5,
        PendingDirect = 6,
    };

    enum class FileLoadMode : int32_t
    {
        SeparateEntries    = 0,
        AppendAsSingleFile = 1,
    };

    struct CachedFileData
    {
        bool isText = false;
        int channels = 0;
        int64_t frames = 0;      // For audio files: sample frames per channel
        double sampleRate = 0.0; // For audio files

        // Items exposed via file_mem/file_var. For audio files this is an
        // interleaved sample array: [f0c0, f0c1, ... f1c0, f1c1, ...].
        std::vector<double> items;
    };

    struct CachedFileEntry
    {
        juce::String path;
        std::unique_ptr<CachedFileData> data;
    };

    struct CachedFileSet
    {
        std::vector<CachedFileEntry> entries;
    };

    struct FileSlot
    {
        juce::String name;
        juce::String description;

        // UI/state thread only (guarded by filePathMutex)
        std::vector<juce::String> currentPaths;
        FileLoadMode loadMode = FileLoadMode::SeparateEntries;
        juce::String currentRecipeXml;
        std::shared_ptr<const za::jsfx::DspJsfxSamplePoolMemorySourceList> currentSamplePoolMemorySources;

        // UI/loader take this mutex; the audio consumer only try-locks it.
        // It orders generation, pending data, and state as one transaction.
        std::mutex handoffMutex;
        std::atomic<FileState> state { FileState::Unassigned };
        std::atomic<uint64_t> generation { 0 };
        std::atomic<uint64_t> pendingGeneration { 0 };
        std::atomic<CachedFileSet*> pending { nullptr };
        std::atomic<int> errorCode { 0 };

        // Audio thread only
        std::unique_ptr<CachedFileSet> active;
        uint64_t activeGeneration = 0;

        FileSlot() = default;

        ~FileSlot() noexcept
        {
            // pending is owned by this slot until the audio thread adopts it.
            if (auto* p = pending.exchange (nullptr, std::memory_order_acq_rel))
                delete p;
        }

        FileSlot (const FileSlot&) = delete;
        FileSlot& operator= (const FileSlot&) = delete;

        FileSlot (FileSlot&& other) noexcept
        {
            *this = std::move (other);
        }

        FileSlot& operator= (FileSlot&& other) noexcept
        {
            if (this == &other)
                return *this;

            name = std::move (other.name);
            description = std::move (other.description);
            currentPaths = std::move (other.currentPaths);
            loadMode = other.loadMode;
            currentRecipeXml = std::move (other.currentRecipeXml);
            currentSamplePoolMemorySources = std::move (other.currentSamplePoolMemorySources);

            state.store (other.state.load (std::memory_order_relaxed), std::memory_order_relaxed);
            generation.store (other.generation.load (std::memory_order_relaxed), std::memory_order_relaxed);
            pendingGeneration.store (other.pendingGeneration.load (std::memory_order_relaxed), std::memory_order_relaxed);
            errorCode.store (other.errorCode.load (std::memory_order_relaxed), std::memory_order_relaxed);

            activeGeneration = other.activeGeneration;
            active = std::move (other.active);

            // transfer pending pointer ownership
            if (auto* old = pending.exchange (nullptr, std::memory_order_acq_rel))
                delete old;

            pending.store (other.pending.exchange (nullptr, std::memory_order_acq_rel), std::memory_order_release);

            // leave other in a benign state
            other.state.store (FileState::Unassigned, std::memory_order_relaxed);
            other.generation.store (0, std::memory_order_relaxed);
            other.pendingGeneration.store (0, std::memory_order_relaxed);
            other.errorCode.store (0, std::memory_order_relaxed);
            other.activeGeneration = 0;
            other.loadMode = FileLoadMode::SeparateEntries;
            other.currentRecipeXml.clear();
            other.currentSamplePoolMemorySources.reset();

            return *this;
        }
    };

    struct FileHandle
    {
        int slot = -1;
        uint64_t boundGeneration = 0;
        int currentFileIndex = 0;
        int64_t cursor = 0; // item cursor (not frames)
    };

    struct SamplePoolHandle
    {
        int slot = -1;
        std::uint64_t nameHash = 0;
        std::unique_ptr<za::jsfx::DspJsfxSamplePool> pool;
    };

    struct FileLoadRequest
    {
        int slot = -1;
        std::vector<juce::File> files;
        FileLoadMode mode = FileLoadMode::SeparateEntries;
        juce::String importRecipeXml;
        uint64_t generation = 0;
    };

    FileHandle* getRuntimeFileHandle (double handle) noexcept
    {
        const int64_t hid = (int64_t) (handle + 1.0e-5);
        if (hid <= 0)
            return nullptr;

        const int idx = (int) (hid - 1);
        if (idx < 0 || idx >= (int) fileHandles.size())
            return nullptr;

        auto& h = fileHandles[(size_t) idx];
        if (h.slot < 0 || h.slot >= (int) fileSlots.size())
            return nullptr;

        return &h;
    }

    FileSlot* getBoundFileSlotForHandle (FileHandle& h) noexcept
    {
        if (h.slot < 0 || h.slot >= (int) fileSlots.size())
            return nullptr;

        auto& slot = fileSlots[(size_t) h.slot];
        if (h.boundGeneration != slot.activeGeneration)
        {
            h.boundGeneration = slot.activeGeneration;
            h.currentFileIndex = 0;
            h.cursor = 0;
        }

        return &slot;
    }

    const CachedFileSet* getActiveFileSetForHandle (FileHandle& h) noexcept
    {
        auto* slot = getBoundFileSlotForHandle (h);
        return slot != nullptr ? slot->active.get() : nullptr;
    }

    const CachedFileData* getSelectedFileDataForHandle (FileHandle& h) noexcept
    {
        const auto* fileSet = getActiveFileSetForHandle (h);
        if (fileSet == nullptr || fileSet->entries.empty())
            return nullptr;

        if (h.currentFileIndex < 0 || h.currentFileIndex >= (int) fileSet->entries.size())
        {
            h.currentFileIndex = 0;
            h.cursor = 0;
        }

        const auto& entry = fileSet->entries[(size_t) h.currentFileIndex];
        return entry.data.get();
    }

    double rt_file_open_common (DSPJSFX_State* state, double indexOrSlot, double mode) noexcept
    {
        juce::ignoreUnused (mode);
        juce::ignoreUnused (state);

        if (fileSlots.empty())
            return -1.0;

        const int64_t slotIndex = (int64_t) (indexOrSlot + 1.0e-5);
        if (slotIndex < 0 || slotIndex >= (int64_t) fileSlots.size())
            return -1.0;

        // If nothing is assigned to this slot and we don't have cached data yet, fail.
        {
            const auto& slot = fileSlots[(size_t) slotIndex];
            const auto stt = slot.state.load();
            const bool hasActive = (slot.active != nullptr);

            if (! hasActive && (stt == FileState::Unassigned || stt == FileState::Error))
                return -1.0;
        }

        // Allocate handle (1-based; 0 is reserved for @serialize in REAPER JSFX).
        int handleIdx = -1;
        if (! freeFileHandles.empty())
        {
            handleIdx = freeFileHandles.back();
            freeFileHandles.pop_back();
        }
        else
        {
            handleIdx = (int) fileHandles.size();
            fileHandles.push_back ({});
        }

        auto& h = fileHandles[(size_t) handleIdx];
        h.slot = (int) slotIndex;
        h.boundGeneration = fileSlots[(size_t) slotIndex].activeGeneration;
        h.currentFileIndex = 0;
        h.cursor = 0;

        return (double) (handleIdx + 1);
    }

    double rt_file_open (DSPJSFX_State* state, double indexOrSlot, double mode) noexcept
    {
        return rt_file_open_common (state, indexOrSlot, mode);
    }

    double rt_file_open_multi (DSPJSFX_State* state, double indexOrSlot, double mode) noexcept
    {
        return rt_file_open_common (state, indexOrSlot, mode);
    }

    double rt_file_close (DSPJSFX_State* state, double handle) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const int idx = (int) (h - fileHandles.data());
        fileHandles[(size_t) idx] = {};
        freeFileHandles.push_back (idx);
        return 0.0;
    }

    double rt_file_rewind (DSPJSFX_State* state, double handle) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        (void) getBoundFileSlotForHandle (*h);
        h->cursor = 0;
        return 0.0;
    }

    double rt_file_seek (DSPJSFX_State* state, double handle, double offset) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* data = getSelectedFileDataForHandle (*h);
        if (! data)
        {
            h->cursor = 0;
            return 0.0;
        }

        int64_t off = (int64_t) (offset + 1.0e-5);
        if (off < 0)
            off = 0;

        const int64_t maxItems = (int64_t) data->items.size();
        if (off > maxItems)
            off = maxItems;

        h->cursor = off;
        return (double) h->cursor;
    }

    double rt_file_avail (DSPJSFX_State* state, double handle) noexcept
    {
        if(auto value=serialization.dispatch(*state,Serialization::Avail,handle))return *value;
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* data = getSelectedFileDataForHandle (*h);
        if (! data)
            return 0.0;

        const int64_t remaining = (int64_t) data->items.size() - h->cursor;
        if (data->isText)
            return remaining > 0 ? 1.0 : 0.0;

        return remaining > 0 ? (double) remaining : 0.0;
    }

    double rt_file_text (DSPJSFX_State* state, double handle) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* data = getSelectedFileDataForHandle (*h);
        return (data && data->isText) ? 1.0 : 0.0;
    }

    double rt_file_riff (DSPJSFX_State* state, double handle, double* outNch, double* outSr) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* data = getSelectedFileDataForHandle (*h);
        if (! data || data->isText || data->channels <= 0)
        {
            if (outNch) jsfxCellStore (outNch, 0.0);
            if (outSr)  jsfxCellStore (outSr, 0.0);
            return 0.0;
        }

        if (outNch) jsfxCellStore (outNch, (double) data->channels);
        if (outSr)  jsfxCellStore (outSr, data->sampleRate);
        return 1.0;
    }

    double rt_file_var (DSPJSFX_State* state, double handle, double* outVar) noexcept
    {
        if(auto value=serialization.dispatch(*state,Serialization::Var,handle,outVar))return *value;
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
        {
            if (outVar) jsfxCellStore (outVar, 0.0);
            return 0.0;
        }

        const auto* data = getSelectedFileDataForHandle (*h);
        if (! data || h->cursor < 0 || h->cursor >= (int64_t) data->items.size())
        {
            if (outVar) jsfxCellStore (outVar, 0.0);
            return 0.0;
        }

        const double v = data->items[(size_t) h->cursor];
        ++h->cursor;

        if (outVar)
            jsfxCellStore (outVar, v);
        return v;
    }

    double rt_file_mem (DSPJSFX_State* state, double handle, double destIndex, double length) noexcept
    {
        if(auto value=serialization.dispatch(*state,Serialization::Mem,handle,nullptr,destIndex,length))return *value;
        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* data = getSelectedFileDataForHandle (*h);
        if (! data)
            return 0.0;

        int64_t dst = (int64_t) (destIndex + 1.0e-5);
        int64_t len = (int64_t) (length + 1.0e-5);
        if (dst < 0) dst = 0;
        if (len <= 0) return 0.0;

        const int64_t avail = (int64_t) data->items.size() - h->cursor;
        if (avail <= 0)
            return 0.0;

        const int64_t toCopy = std::min (len, avail);

        // Ensure mem is big enough.
        const int64_t needed = dst + toCopy;
        if (needed > state->memN)
            jsfx_ensure_mem (state, (int64_t) needed);

        if (state->mem == nullptr || state->memN <= 0)
            return 0.0;

        const int64_t clamped = std::min<int64_t> (toCopy, state->memN - dst);
        if (clamped <= 0)
            return 0.0;

        auto* mem = state->mem;
        const double* src = data->items.data() + h->cursor;

        jsfxWriteCells (mem + dst, src, (size_t) clamped);
        noteTrackedJsfxMemUsed (state, dst + clamped);
        h->cursor += clamped;
        return (double) clamped;
    }

    double rt_file_multi_count (DSPJSFX_State* state, double handle) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* fileSet = getActiveFileSetForHandle (*h);
        return fileSet != nullptr ? (double) fileSet->entries.size() : 0.0;
    }

    double rt_file_multi_select (DSPJSFX_State* state, double handle, double fileIndex) noexcept
    {
        juce::ignoreUnused (state);

        auto* h = getRuntimeFileHandle (handle);
        if (h == nullptr)
            return 0.0;

        const auto* fileSet = getActiveFileSetForHandle (*h);
        if (fileSet == nullptr || fileSet->entries.empty())
            return 0.0;

        const int64_t requestedIndex = (int64_t) std::floor (fileIndex + 1.0e-5);
        if (requestedIndex < 0 || requestedIndex >= (int64_t) fileSet->entries.size())
            return 0.0;

        h->currentFileIndex = (int) requestedIndex;
        h->cursor = 0;
        return 1.0;
    }

    struct ScopedSamplePoolReads
    {
        explicit ScopedSamplePoolReads(FileRuntime& p) noexcept : owner(p), previous(current)
        { current = this; }
        ~ScopedSamplePoolReads() { current = previous; }
        ScopedSamplePoolReads(const ScopedSamplePoolReads&) = delete;
        ScopedSamplePoolReads& operator=(const ScopedSamplePoolReads&) = delete;
        FileRuntime& owner;
        ScopedSamplePoolReads* previous;
        za::jsfx::DspJsfxSamplePool::ReadBatch generations;
        std::array<za::jsfx::DspJsfxSamplePool*, 64> handles {};
        inline static thread_local ScopedSamplePoolReads* current = nullptr;
    };

    za::jsfx::DspJsfxSamplePool* getSamplePoolForHandle (double handle) noexcept
    {
        if (!std::isfinite(handle) || handle < 0.5 || handle > 2147483647.0)
            return nullptr;
        const int idx = static_cast<int>(std::llround(handle)) - 1;
        auto* batch = ScopedSamplePoolReads::current;
        const bool cacheable = batch != nullptr && &batch->owner == this
                           && idx >= 0 && idx < static_cast<int>(batch->handles.size());
        if (cacheable && batch->handles[static_cast<size_t>(idx)] != nullptr)
            return batch->handles[static_cast<size_t>(idx)];
        std::lock_guard<std::mutex> lock(samplePoolMutex);
        if (idx < 0 || idx >= static_cast<int>(samplePools.size()) || samplePools[static_cast<size_t>(idx)] == nullptr)
            return nullptr;
        auto* pool = samplePools[static_cast<size_t>(idx)]->pool.get();
        if (cacheable)
            batch->handles[static_cast<size_t>(idx)] = pool;
        return pool;
    }

    double rt_sample_pool_from_slot (DSPJSFX_State* state, double slotValue, double nameHandle) noexcept
    {
        const int slot = (int) std::llround (slotValue);
        if (slot < 0 || slot >= (int) fileSlots.size())
            return 0.0;

        const std::uint64_t nameHash = jsfx_string_hash (state, nameHandle);
        const auto key = std::to_string (slot) + ":" + std::to_string ((unsigned long long) nameHash);

        std::lock_guard<std::mutex> lock (samplePoolMutex);
        auto it = samplePoolByKey.find (key);
        if (it != samplePoolByKey.end())
            return (double) (it->second + 1);

        auto h = std::make_unique<SamplePoolHandle>();
        h->slot = slot;
        h->nameHash = nameHash;
        h->pool = std::make_unique<za::jsfx::DspJsfxSamplePool>();
        configureSamplePoolForCurrentEngineRate (*h->pool);
        const int idx = (int) samplePools.size();
        samplePools.push_back (std::move (h));
        samplePoolByKey[key] = idx;
        return (double) (idx + 1);
    }

    double rt_sample_pool_commit (double poolHandle) noexcept
    {
        const int hid = (int) std::llround (poolHandle);
        if (hid <= 0)
            return 0.0;

        int slot = -1;
        za::jsfx::DspJsfxSamplePool* pool = nullptr;
        {
            std::lock_guard<std::mutex> lock (samplePoolMutex);
            const int idx = hid - 1;
            if (idx < 0 || idx >= (int) samplePools.size() || samplePools[(size_t) idx] == nullptr)
                return 0.0;
            slot = samplePools[(size_t) idx]->slot;
            pool = samplePools[(size_t) idx]->pool.get();
        }

        if (pool == nullptr || slot < 0 || slot >= (int) fileSlots.size())
            return 0.0;

        configureSamplePoolForCurrentEngineRate (*pool);

        std::vector<juce::String> paths;
        std::shared_ptr<const za::jsfx::DspJsfxSamplePoolMemorySourceList> memorySources;
        std::uint64_t gen = 0;
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            paths = fileSlots[(size_t) slot].currentPaths;
            memorySources = fileSlots[(size_t) slot].currentSamplePoolMemorySources;
            gen = fileSlots[(size_t) slot].generation.load (std::memory_order_acquire);
        }

        if (memorySources != nullptr && ! memorySources->empty())
            return pool->commitFromMemory (std::move (memorySources), gen) ? 1.0 : 0.0;

        return pool->commitFromPaths (paths, gen) ? 1.0 : 0.0;
    }

    std::unique_ptr<CachedFileData> makeCachedFileDataFromImportAudio (const za::fileimport::AudioFileData& src) const
    {
        const int channels = juce::jlimit (1, 64, src.buffer.getNumChannels());
        const int64_t frames = juce::jmax<int64_t> (0, src.buffer.getNumSamples());

        auto out = std::make_unique<CachedFileData>();
        out->isText = false;
        out->channels = channels;
        out->frames = frames;
        out->sampleRate = src.sampleRate > 0.0 ? src.sampleRate : 48000.0;

        const int64_t totalItems64 = frames * (int64_t) channels;
        if (totalItems64 <= 0)
            return out;

        if (totalItems64 > (int64_t) (std::numeric_limits<size_t>::max() / sizeof (double)))
            return nullptr;

        out->items.resize ((size_t) totalItems64);
        for (int64_t i = 0; i < frames; ++i)
        {
            const size_t base = (size_t) (i * (int64_t) channels);
            for (int ch = 0; ch < channels; ++ch)
                out->items[base + (size_t) ch] = (double) src.buffer.getSample (ch, (int) i);
        }

        if (auto converted = resampleCachedAudioForCurrentEngineIfNeeded (*out))
            return converted;

        return out;
    }

    std::shared_ptr<const za::jsfx::DspJsfxSamplePoolMemorySourceList> makeSamplePoolSourcesFromImportAudio (const std::vector<za::fileimport::AudioFileData>& renderedAudio) const
    {
        using Source = za::jsfx::DspJsfxSamplePoolMemorySource;
        using SourceList = za::jsfx::DspJsfxSamplePoolMemorySourceList;

        auto out = std::make_shared<SourceList>();
        out->reserve (renderedAudio.size());

        for (const auto& audio : renderedAudio)
        {
            const int channels = juce::jlimit (1, 64, audio.buffer.getNumChannels());
            const int frames = audio.buffer.getNumSamples();
            if (frames <= 0)
                continue;

            const int64_t totalItems64 = (int64_t) frames * (int64_t) channels;
            if (totalItems64 <= 0 || totalItems64 > (int64_t) std::numeric_limits<int>::max())
                continue;

            Source src;
            src.name = (audio.sourceName.isNotEmpty() ? audio.sourceName : juce::String ("memory sample")).toStdString();
            src.sampleRate = (std::uint32_t) juce::jmax (1, (int) std::llround (audio.sampleRate > 0.0 ? audio.sampleRate : 48000.0));
            src.channels = (std::uint16_t) channels;
            src.audio.resize ((size_t) totalItems64);

            for (int i = 0; i < frames; ++i)
            {
                const size_t base = (size_t) ((int64_t) i * (int64_t) channels);
                for (int ch = 0; ch < channels; ++ch)
                    src.audio[base + (size_t) ch] = audio.buffer.getSample (ch, i);
            }

            out->push_back (std::move (src));
        }

        if (out->empty())
            return {};

        return out;
    }

    std::shared_ptr<const za::jsfx::DspJsfxSamplePoolMemorySourceList> makeSamplePoolSourcesFromCachedFileSet (const CachedFileSet& set) const
    {
        using Source = za::jsfx::DspJsfxSamplePoolMemorySource;
        using SourceList = za::jsfx::DspJsfxSamplePoolMemorySourceList;

        auto out = std::make_shared<SourceList>();
        out->reserve (set.entries.size());

        for (const auto& entry : set.entries)
        {
            if (entry.data == nullptr || entry.data->isText || entry.data->frames <= 0 || entry.data->channels <= 0)
                continue;

            const int channels = juce::jlimit (1, 64, entry.data->channels);
            const int64_t frames = entry.data->frames;
            const int64_t totalItems64 = frames * (int64_t) channels;
            if (totalItems64 <= 0 || totalItems64 > (int64_t) std::numeric_limits<int>::max())
                continue;

            if ((int64_t) entry.data->items.size() < totalItems64)
                continue;

            Source src;
            src.name = (entry.path.isNotEmpty() ? entry.path : juce::String ("memory sample")).toStdString();
            src.sampleRate = (std::uint32_t) juce::jmax (1, (int) std::llround (entry.data->sampleRate > 0.0 ? entry.data->sampleRate : 48000.0));
            src.channels = (std::uint16_t) channels;
            src.audio.resize ((size_t) totalItems64);

            for (int64_t i = 0; i < totalItems64; ++i)
                src.audio[(size_t) i] = (float) entry.data->items[(size_t) i];

            out->push_back (std::move (src));
        }

        if (out->empty())
            return {};

        return out;
    }

    void notifyFileSlotChangedForUiAndSamplePools (int slotIndex0)
    {
        recommitSamplePoolsForSlot (slotIndex0);

        // Wake DSP-JSFX's own Smart Idle lifecycle and refresh @gfx/UI state.
        // Host-level CLAP processing is deliberately left alone: our Smart Idle
        // still receives processBlock() and decides whether jsfx_process_block() runs.
        requestExternalWakeEvent (true);
    }

    void recommitSamplePoolsForSlot (int slotIndex0)
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return;

        std::vector<za::jsfx::DspJsfxSamplePool*> pools;
        {
            std::lock_guard<std::mutex> lock (samplePoolMutex);
            for (auto& handle : samplePools)
                if (handle != nullptr && handle->slot == slotIndex0 && handle->pool != nullptr)
                    pools.push_back (handle->pool.get());
        }

        if (pools.empty())
            return;

        std::vector<juce::String> paths;
        std::shared_ptr<const za::jsfx::DspJsfxSamplePoolMemorySourceList> memorySources;
        std::uint64_t gen = 0;
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            paths = fileSlots[(size_t) slotIndex0].currentPaths;
            memorySources = fileSlots[(size_t) slotIndex0].currentSamplePoolMemorySources;
            gen = fileSlots[(size_t) slotIndex0].generation.load (std::memory_order_acquire);
        }

        for (auto* pool : pools)
        {
            if (pool == nullptr)
                continue;

            configureSamplePoolForCurrentEngineRate (*pool);

            if (memorySources != nullptr && ! memorySources->empty())
                pool->commitFromMemory (memorySources, gen);
            else
                pool->commitFromPaths (paths, gen);
        }
    }

    void shutdownFileRuntime()
    {
        fileLoadExit.store (true);
        fileLoadCv.notify_all();

        if (fileLoadThread.joinable())
            fileLoadThread.join();

        {
            std::lock_guard<std::mutex> lk (fileLoadMutex);
            fileLoadQueue.clear();
        }

        for (auto& slot : fileSlots)
        {
            if (auto* p = slot.pending.exchange (nullptr))
                delete p;
        }

        fileHandles.clear();
        freeFileHandles.clear();
        {
            std::lock_guard<std::mutex> lock (samplePoolMutex);
            samplePoolByKey.clear();
            samplePools.clear();
        }
        fileSlots.clear();
    }

    bool promotePendingFileLoads() noexcept
    {
        bool changed = false;
        for (auto& slot : fileSlots)
        {
            const auto hint = slot.state.load (std::memory_order_acquire);
            if (hint != FileState::ReadyPending && hint != FileState::PendingClear
                && hint != FileState::PendingDirect)
                continue; // No handoff lock on the ordinary idle path.
            std::unique_lock<std::mutex> lock (slot.handoffMutex, std::try_to_lock);
            if (! lock.owns_lock())
                continue; // Try again next block, never wait behind UI/loader work.
            const auto stt = slot.state.load();
            if (stt == FileState::ReadyPending)
            {
                CachedFileSet* p = slot.pending.exchange (nullptr);
                if (p != nullptr)
                {
                    if (slot.pendingGeneration.load() != slot.generation.load())
                    {
                        delete p;
                        continue;
                    }
                    slot.active.reset (p);
                    slot.activeGeneration = slot.pendingGeneration.load();
                    slot.state.store (FileState::ReadyActive);
                    changed = true;
                }
                else
                {
                    slot.state.store (FileState::Error);
                    changed = true;
                }
            }
            else if (stt == FileState::PendingClear || stt == FileState::PendingDirect)
            {
                slot.active.reset();
                slot.activeGeneration = slot.generation.load();
                slot.state.store (stt == FileState::PendingDirect ? FileState::ReadyActive : FileState::Unassigned);
                changed = true;
            }
        }
        return changed;
    }

    void enqueueFileLoad (FileLoadRequest req)
    {
        if (fileDecls.empty())
            return;

        {
            std::lock_guard<std::mutex> lk (fileLoadMutex);
            fileLoadQueue.push_back (std::move (req));
        }
        fileLoadCv.notify_one();
    }

    virtual juce::File resolveFileToken (const juce::String& token) const
    {
        juce::File f (token);
        if (juce::File::isAbsolutePath (token))
            return f;
        if(sourceDirectory.isDirectory())return sourceDirectory.getChildFile(token);

        auto cand = juce::File::getCurrentWorkingDirectory().getChildFile (token);
        if (cand.existsAsFile())
            return cand;

        auto exeDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
        cand = exeDir.getChildFile (token);
        if (cand.existsAsFile())
            return cand;

        return cand;
    }

    void fileLoaderThreadMain()
    {
        while (! fileLoadExit.load())
        {
            FileLoadRequest req;
            {
                std::unique_lock<std::mutex> lk (fileLoadMutex);
                fileLoadCv.wait (lk, [this] { return fileLoadExit.load() || rateReloadRequested.load() || ! fileLoadQueue.empty(); });

                if (fileLoadExit.load())
                    break;

                if(rateReloadRequested.exchange(false)) {
                    lk.unlock();reloadCurrentSlotsForRate();continue;
                }

                req = std::move (fileLoadQueue.front());
                fileLoadQueue.pop_front();
            }

            if (req.slot < 0 || req.slot >= (int) fileSlots.size())
                continue;

            auto data = req.importRecipeXml.isNotEmpty()
                            ? loadImportRecipeToMemory (req.files, req.importRecipeXml, req.mode)
                            : loadFilesToMemory (req.files, req.mode);

            auto& slot = fileSlots[(size_t) req.slot];
            auto samplePoolSources = data && req.importRecipeXml.isNotEmpty()
                ? makeSamplePoolSourcesFromCachedFileSet (*data) : nullptr;
            {
                // Recheck at the commit boundary, under the same lock as Clear
                // and selection. A stale worker cannot publish state or sources.
                std::lock_guard<std::mutex> handoffLock (slot.handoffMutex);
                if (slot.generation.load() != req.generation)
                    continue;
                if (! data)
                {
                    slot.errorCode.store (2);
                    slot.state.store (FileState::Error);
                }
                else
                {
                    if (req.importRecipeXml.isNotEmpty())
                    {
                        std::lock_guard<std::mutex> pathLock (filePathMutex);
                        slot.currentSamplePoolMemorySources = std::move (samplePoolSources);
                    }
                    if (auto* old = slot.pending.exchange (data.release()))
                        delete old;
                    slot.pendingGeneration.store (req.generation);
                    slot.state.store (FileState::ReadyPending);
                }
            }
            notifyFileSlotChangedForUiAndSamplePools (req.slot);
        }
    }

    std::unique_ptr<CachedFileData> convertAudioDataToFormat (const CachedFileData& src,
                                                               int dstChannels,
                                                               double dstSampleRate) const
    {
        if (src.isText || src.channels <= 0 || src.sampleRate <= 0.0)
            return nullptr;

        dstChannels = juce::jlimit (1, 64, dstChannels);
        if (dstSampleRate <= 0.0)
            dstSampleRate = src.sampleRate;

        auto out = std::make_unique<CachedFileData>();
        out->isText = false;
        out->channels = dstChannels;
        out->sampleRate = dstSampleRate;

        if (src.frames <= 0)
        {
            out->frames = 0;
            return out;
        }

        const auto scaledFrames = (double) src.frames * dstSampleRate / src.sampleRate;
        if (! std::isfinite (scaledFrames)
            || scaledFrames <= 0.0
            || scaledFrames > (double) std::numeric_limits<int64_t>::max())
            return nullptr;

        out->frames = juce::jmax<int64_t> (1, (int64_t) std::llround (scaledFrames));

        if (out->frames > std::numeric_limits<int64_t>::max() / (int64_t) dstChannels)
            return nullptr;

        const int64_t totalItems64 = out->frames * (int64_t) dstChannels;
        if (totalItems64 <= 0
            || totalItems64 > (int64_t) (std::numeric_limits<size_t>::max() / sizeof (double)))
            return nullptr;

        try
        {
            out->items.resize ((size_t) totalItems64);
        }
        catch (...)
        {
            return nullptr;
        }

        auto sampleForDestChannelAtFrame = [&src, dstChannels] (int64_t frameIndex, int dstChannel) -> double
        {
            if (src.frames <= 0)
                return 0.0;

            frameIndex = juce::jlimit<int64_t> (0, src.frames - 1, frameIndex);
            const auto base = (size_t) (frameIndex * src.channels);

            if (dstChannels == 1)
            {
                double sum = 0.0;
                for (int ch = 0; ch < src.channels; ++ch)
                    sum += src.items[base + (size_t) ch];

                return sum / (double) juce::jmax (1, src.channels);
            }

            if (src.channels == 1)
                return src.items[base];

            if (dstChannel < src.channels)
                return src.items[base + (size_t) dstChannel];

            return src.items[base + (size_t) (src.channels - 1)];
        };

        const bool needsResample = std::abs (src.sampleRate - dstSampleRate) > 1.0e-9;

        for (int64_t frame = 0; frame < out->frames; ++frame)
        {
            const double srcPos = needsResample
                                      ? ((double) frame * src.sampleRate) / dstSampleRate
                                      : (double) frame;

            const int64_t pos0 = juce::jlimit<int64_t> (0, src.frames - 1, (int64_t) std::floor (srcPos));
            const int64_t pos1 = juce::jmin (src.frames - 1, pos0 + 1);
            const double frac = juce::jlimit (0.0, 1.0, srcPos - (double) pos0);
            const auto dstBase = (size_t) (frame * dstChannels);

            for (int ch = 0; ch < dstChannels; ++ch)
            {
                const double s0 = sampleForDestChannelAtFrame (pos0, ch);
                const double s1 = sampleForDestChannelAtFrame (pos1, ch);
                out->items[dstBase + (size_t) ch] = s0 + (s1 - s0) * frac;
            }
        }

        return out;
    }
    std::unique_ptr<CachedFileSet> loadRenderedAudioToMemory (const std::vector<za::fileimport::AudioFileData>& renderedAudio)
    {
        if (renderedAudio.empty())
            return nullptr;

        auto out = std::make_unique<CachedFileSet>();
        out->entries.reserve (renderedAudio.size());

        for (const auto& audio : renderedAudio)
        {
            auto data = makeCachedFileDataFromImportAudio (audio);
            if (! data)
                return nullptr;

            CachedFileEntry entry;
            entry.path = audio.sourceName.isNotEmpty() ? audio.sourceName : juce::String ("memory://za-import/audio");
            entry.data = std::move (data);
            out->entries.push_back (std::move (entry));
        }

        return out;
    }

    std::unique_ptr<CachedFileSet> loadImportRecipeToMemory (const std::vector<juce::File>& sourceFiles,
                                                             const juce::String& importRecipeXml,
                                                             FileLoadMode fallbackMode)
    {
        if (importRecipeXml.isEmpty())
            return nullptr;

        auto xml = juce::parseXML (importRecipeXml);
        if (xml == nullptr)
            return nullptr;

        auto tree = juce::ValueTree::fromXml (*xml);
        if (! tree.isValid())
            return nullptr;

        auto recipe = za::fileimport::recipeFromValueTree (tree);
        auto recipeFiles = sourceFiles;
        if (! recipe.inputs.empty())
        {
            recipeFiles.clear();
            for (const auto& input : recipe.inputs) recipeFiles.emplace_back (input.path);
        }
        auto result = za::fileimport::renderImportAction (recipeFiles, recipe.action, recipe.rules);
        if (! result.ok)
            return nullptr;

        if (! result.renderedAudio.empty())
            return loadRenderedAudioToMemory (result.renderedAudio);

        if (! result.files.empty())
        {
            const auto mode = result.loadMode == za::fileimport::RenderedLoadMode::AppendAsSingleFile
                                ? FileLoadMode::AppendAsSingleFile
                                : fallbackMode;
            return loadFilesToMemory (result.files, mode);
        }

        return nullptr;
    }

    std::unique_ptr<CachedFileData> appendLoadedFilesInMemory (const std::vector<juce::File>& files)
    {
        std::unique_ptr<CachedFileData> combined;
        bool referenceIsText = false;
        int referenceChannels = 0;
        double referenceSampleRate = 0.0;

        for (const auto& file : files)
        {
            auto data = loadFileToMemory (file);
            if (! data)
                return nullptr;

            if (combined == nullptr)
            {
                referenceIsText = data->isText;
                referenceChannels = data->channels;
                referenceSampleRate = data->sampleRate;
                combined = std::move (data);
                continue;
            }

            if (referenceIsText != data->isText)
                return nullptr;

            if (referenceIsText)
            {
                combined->items.insert (combined->items.end(), data->items.begin(), data->items.end());
                continue;
            }

            if (data->channels != referenceChannels || std::abs (data->sampleRate - referenceSampleRate) > 1.0e-9)
            {
                data = convertAudioDataToFormat (*data, referenceChannels, referenceSampleRate);
                if (! data)
                    return nullptr;
            }

            combined->items.insert (combined->items.end(), data->items.begin(), data->items.end());
            combined->frames += data->frames;
        }

        return combined;
    }

    std::unique_ptr<CachedFileSet> loadFilesToMemory (const std::vector<juce::File>& files,
                                                       FileLoadMode loadMode)
    {
        if (files.empty())
            return nullptr;

        if (loadMode == FileLoadMode::AppendAsSingleFile)
        {
            auto data = appendLoadedFilesInMemory (files);
            if (! data)
                return nullptr;

            auto out = std::make_unique<CachedFileSet>();
            CachedFileEntry entry;
            entry.path = files.front().getFullPathName();
            entry.data = std::move (data);
            out->entries.push_back (std::move (entry));
            return out;
        }

        auto out = std::make_unique<CachedFileSet>();
        out->entries.reserve (files.size());

        for (const auto& file : files)
        {
            if (! file.existsAsFile())
                return nullptr;

            auto data = loadFileToMemory (file);
            if (! data)
                return nullptr;

            CachedFileEntry entry;
            entry.path = file.getFullPathName();
            entry.data = std::move (data);
            out->entries.push_back (std::move (entry));
        }

        return out;
    }

    std::unique_ptr<CachedFileData> loadFileToMemory (const juce::File& file)
    {
        const auto ext = file.getFileExtension().toLowerCase();
        if (ext == ".txt" || ext == ".csv" || ext == ".dat")
            return loadTextFile (file);

        if (auto a = loadAudioFile (file))
            return a;

        return nullptr;
    }


    std::unique_ptr<CachedFileData> loadAudioFile (const juce::File& file)
    {
        std::unique_ptr<juce::AudioFormatReader> reader (fileFormatManager.createReaderFor (file));
        if (! reader)
            return nullptr;

        bool skipUnsafeMalformedFile = false;
        const int64_t safeFrames = za::jsfx::chooseSafeDecodedFrameCount (file, *reader, skipUnsafeMalformedFile);
        if (skipUnsafeMalformedFile)
            return nullptr;

        auto out = std::make_unique<CachedFileData>();
        out->isText = false;
        out->channels = std::max<int> (1, (int) reader->numChannels);
        out->channels = std::min (out->channels, 64);
        out->frames = juce::jmax<int64_t> (0, safeFrames);
        out->sampleRate = reader->sampleRate;

        if (out->frames <= 0)
            return out;

        if (out->frames > std::numeric_limits<int64_t>::max() / (int64_t) out->channels)
            return nullptr;

        const int64_t totalItems64 = out->frames * (int64_t) out->channels;
        if (totalItems64 <= 0)
            return out;

        // Guard against overflow / absurd allocations before the decode buffer is sized.
        if (totalItems64 > (int64_t) (std::numeric_limits<size_t>::max() / sizeof (double)))
            return nullptr;

        try
        {
            out->items.resize ((size_t) totalItems64);
        }
        catch (...)
        {
            return nullptr;
        }

        juce::AudioBuffer<float> temp;
        const int chunk = 65536;
        int64_t decodedFrames = 0;

        for (int64_t pos = 0; pos < out->frames; pos += chunk)
        {
            const int toRead = (int) juce::jmin<int64_t> (chunk, out->frames - pos);

            temp.setSize (out->channels, toRead, false, false, true);
            temp.clear();

            if (! reader->read (&temp, 0, toRead, pos, true, true))
                break;

            for (int i = 0; i < toRead; ++i)
            {
                const int64_t frame = pos + i;
                const size_t base = (size_t) (frame * out->channels);

                for (int ch = 0; ch < out->channels; ++ch)
                    out->items[base + (size_t) ch] = (double) temp.getSample (ch, i);
            }

            decodedFrames = pos + toRead;
        }

        if (decodedFrames < out->frames)
        {
            out->frames = juce::jmax<int64_t> (0, decodedFrames);
            out->items.resize ((size_t) (out->frames * (int64_t) out->channels));
        }

        if (auto converted = resampleCachedAudioForCurrentEngineIfNeeded (*out))
            return converted;

        return out;
    }

    static bool parseIntToken (const juce::String& tok, int base, int64_t& out)
    {
        auto s = tok.trim();
        if (s.isEmpty())
            return false;

        const auto utf8 = s.toRawUTF8();
        char* end = nullptr;
        errno = 0;

        const long long v = std::strtoll (utf8, &end, base);
        if (end == utf8 || errno != 0)
            return false;

        out = (int64_t) v;
        return true;
    }

    std::unique_ptr<CachedFileData> loadTextFile (const juce::File& file)
    {
        auto content = file.loadFileAsString();
        if (content.isEmpty())
            return nullptr;

        auto out = std::make_unique<CachedFileData>();
        out->isText = true;
        out->channels = 0;
        out->frames = 0;
        out->sampleRate = 0.0;

        juce::StringArray lines;
        lines.addLines (content);

        for (auto line : lines)
        {
            // Strip comments starting with ';' or '#'
            const int semi = line.indexOfChar (';');
            const int hash = line.indexOfChar ('#');

            int cut = -1;
            if (semi >= 0) cut = semi;
            if (hash >= 0) cut = (cut < 0) ? hash : std::min (cut, hash);

            if (cut >= 0)
                line = line.substring (0, cut);

            line = line.replaceCharacter (',', ' ');
            line = line.trim();

            if (line.isEmpty())
                continue;

            juce::StringArray toks;
            toks.addTokens (line, " \t", "");
            toks.trim();
            toks.removeEmptyStrings();

            for (auto tok : toks)
            {
                tok = tok.trim();
                if (tok.isEmpty())
                    continue;

                int64_t iv = 0;

                // Support 'b' (binary) and 'x' (hex) prefixes, plus 0x...
                if (tok.startsWithChar ('b') && parseIntToken (tok.substring (1), 2, iv))
                {
                    out->items.push_back ((double) iv);
                    continue;
                }

                if (tok.startsWithChar ('x') && parseIntToken (tok.substring (1), 16, iv))
                {
                    out->items.push_back ((double) iv);
                    continue;
                }

                if (tok.startsWithIgnoreCase ("0x") && parseIntToken (tok.substring (2), 16, iv))
                {
                    out->items.push_back ((double) iv);
                    continue;
                }

                out->items.push_back (tok.getDoubleValue());
            }
        }

        return out;
    }

    bool hasDeferredSamplePoolAdoptionPending() noexcept
    {
        if(!usesSamplePool)return false;
        // Fast idle path: do not touch the pool registry unless a worker has
        // completed a publication since the last scan. Deferred generations
        // need JSFX @block to reach the script-owned sample_pool_adopt() safe
        // boundary, so Smart Idle must not suppress those blocks.
        if (! pendingDeferredSamplePoolAdoption.load (std::memory_order_acquire))
            return false;

        // Never wait on a loader/UI thread from the audio callback. Contention
        // means conservatively remain awake and recheck on the next block.
        std::unique_lock<std::mutex> lock (samplePoolMutex, std::try_to_lock);
        if (! lock.owns_lock())
            return true;

        for (const auto& handle : samplePools)
            if (handle != nullptr && handle->pool != nullptr && handle->pool->hasPendingAdoption())
                return true;

        pendingDeferredSamplePoolAdoption.store (false, std::memory_order_release);

        return false;
    }

    void configureSamplePoolForCurrentEngineRate (za::jsfx::DspJsfxSamplePool& pool)
    {
        pool.setTargetSampleRate (getCurrentFileCacheTargetSampleRate());
        pool.setCompletionCallback ([this] (std::uint64_t, int)
        {
            // The worker has published either an immediately-active generation
            // or a deferred generation awaiting sample_pool_adopt(). Mark the
            // aggregate dirty; the audio callback performs the cheap exact scan.
            if(usesSamplePool)pendingDeferredSamplePoolAdoption.store (true, std::memory_order_release);
    
            requestExternalWakeEvent (true);
        });
    }

    void recommitAllSamplePoolsForEngineRate()
    {
        std::vector<int> slots;
        {
            std::lock_guard<std::mutex> lock (samplePoolMutex);
            for (auto& handle : samplePools)
            {
                if (handle == nullptr || handle->pool == nullptr)
                    continue;

                configureSamplePoolForCurrentEngineRate (*handle->pool);
                if (handle->slot >= 0)
                    slots.push_back (handle->slot);
            }
        }

        std::sort (slots.begin(), slots.end());
        slots.erase (std::unique (slots.begin(), slots.end()), slots.end());

        for (int slot : slots)
            recommitSamplePoolsForSlot (slot);
    }

    static bool shouldResampleCachedAudioToTarget (double sourceRate, double targetRate) noexcept
    {
        return std::isfinite (sourceRate)
            && std::isfinite (targetRate)
            && sourceRate > 1000.0
            && targetRate > 1000.0
            && std::abs (sourceRate - targetRate) > 1.0;
    }

    std::unique_ptr<CachedFileData> resampleCachedAudioForCurrentEngineIfNeeded (const CachedFileData& src) const
    {
        const double target = getCurrentFileCacheTargetSampleRate();
        if (! shouldResampleCachedAudioToTarget (src.sampleRate, target))
            return nullptr;

        return convertAudioDataToFormat (src, src.channels, target);
    }

    void setFileSlotPaths (int slotIndex0, const std::vector<juce::File>& files, bool shouldRememberForUi = true)
    {
        setFileSlotPathsWithMode (slotIndex0, files, FileLoadMode::SeparateEntries, shouldRememberForUi);
    }

    void setFileSlotPathsWithMode (int slotIndex0,
                                   const std::vector<juce::File>& files,
                                   FileLoadMode loadMode,
                                   bool shouldRememberForUi = true,
                                   const juce::String& importRecipeXml = {})
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return;

        if (files.empty())
        {
            clearFileSlot (slotIndex0);
            return;
        }

        auto& slot = fileSlots[(size_t) slotIndex0];

        std::vector<juce::File> normalisedFiles;
        std::vector<juce::String> fullPaths;
        normalisedFiles.reserve (files.size());
        fullPaths.reserve (files.size());

        for (const auto& file : files)
        {
            const auto fullPath = file.getFullPathName();
            if (fullPath.isEmpty())
                continue;

            normalisedFiles.push_back (file);
            fullPaths.push_back (fullPath);
        }

        if (normalisedFiles.empty())
        {
            clearFileSlot (slotIndex0);
            return;
        }

        std::unique_lock<std::mutex> handoffLock (slot.handoffMutex);

        if (shouldRememberForUi)
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            slot.currentPaths = fullPaths;
            slot.loadMode = loadMode;
            slot.currentRecipeXml = importRecipeXml;
            slot.currentSamplePoolMemorySources.reset();
            rememberFileSelectionLocked (normalisedFiles, loadMode, importRecipeXml);
        }
        else
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            slot.currentPaths = fullPaths;
            slot.loadMode = loadMode;
            slot.currentRecipeXml = importRecipeXml;
            slot.currentSamplePoolMemorySources.reset();
        }

        // Cancel any pending, not-yet-promoted data.
        if (auto* oldPending = slot.pending.exchange (nullptr))
            delete oldPending;

        slot.errorCode.store (0);

        // Bump generation and request a new load (or mark error if any file does not exist).
        const uint64_t gen = slot.generation.fetch_add (1) + 1;
        slot.pendingGeneration.store (gen);

        const bool allExist = std::all_of (normalisedFiles.begin(), normalisedFiles.end(),
                                           [] (const juce::File& file) { return file.existsAsFile(); });

        if (allExist)
        {
            if(usesSamplePool && !usesLegacyFileIO) {
            if (importRecipeXml.isEmpty())
            {
                // sample_pool_* owns large-bank decoding for direct on-disk sources.
                // Avoid the legacy file_mem() double cache when the compiled DSP
                // does not call the old file_* APIs.
                slot.state.store (FileState::PendingDirect);
            }
            else
            {
                // Recipe-backed selections must be rendered in memory so
                // sample_pool users receive the processed/segmented result
                // instead of reloading the original source paths.
                slot.state.store (FileState::Loading);

                FileLoadRequest req;
                req.slot = slotIndex0;
                req.files = normalisedFiles;
                req.mode = loadMode;
                req.importRecipeXml = importRecipeXml;
                req.generation = gen;
                enqueueFileLoad (std::move (req));
            }
            } else {
            slot.state.store (FileState::Loading);

            FileLoadRequest req;
            req.slot = slotIndex0;
            req.files = normalisedFiles;
            req.mode = loadMode;
            req.importRecipeXml = importRecipeXml;
            req.generation = gen;
            enqueueFileLoad (std::move (req));
            }
        }
        else
        {
            slot.errorCode.store (1);
            slot.state.store (FileState::Error);
        }

        handoffLock.unlock();
        if (shouldRememberForUi)
            savePersistentFileUiState();
        notifyFileSlotChangedForUiAndSamplePools (slotIndex0);
    }

    void clearFileSlot (int slotIndex0)
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return;

        auto& slot = fileSlots[(size_t) slotIndex0];
        std::unique_lock<std::mutex> handoffLock (slot.handoffMutex);

        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            slot.currentPaths.clear();
            slot.loadMode = FileLoadMode::SeparateEntries;
            slot.currentRecipeXml.clear();
            slot.currentSamplePoolMemorySources.reset();
        }

        if (auto* oldPending = slot.pending.exchange (nullptr))
            delete oldPending;

        slot.errorCode.store (0);

        const uint64_t gen = slot.generation.fetch_add (1) + 1;
        slot.pendingGeneration.store (gen);
        slot.state.store (FileState::PendingClear);
        handoffLock.unlock();
        notifyFileSlotChangedForUiAndSamplePools (slotIndex0);
    }




protected:
    std::vector<JsfxFileDecl> fileDecls;
    std::vector<FileSlot> fileSlots;
    std::vector<FileHandle> fileHandles;
    std::vector<int> freeFileHandles;
    std::mutex samplePoolMutex;
    std::vector<std::unique_ptr<SamplePoolHandle>> samplePools;
    std::unordered_map<std::string, int> samplePoolByKey;
    std::atomic<bool> pendingDeferredSamplePoolAdoption { false };
    mutable std::mutex filePathMutex;
    std::mutex fileLoadMutex;
    std::condition_variable fileLoadCv;
    std::deque<FileLoadRequest> fileLoadQueue;
    std::thread fileLoadThread;
    std::atomic<bool> fileLoadExit { false };
    juce::AudioFormatManager fileFormatManager;
    bool usesSamplePool=DSPJSFX_USES_SAMPLE_POOL,usesLegacyFileIO=DSPJSFX_USES_LEGACY_FILE_IO;
    virtual void rememberFileSelectionLocked(const std::vector<juce::File>&,FileLoadMode,const juce::String&) {}
    virtual void savePersistentFileUiState() {}
public:
    Serialization serialization;
#if DSPJSFX_HAS_TASKS
    TaskHeapPort taskHeap;
#endif
    std::function<void()> externalWake;
    std::atomic<double> cacheTargetRate{0};
    std::atomic<bool> rateReloadRequested{false};
    juce::File sourceDirectory;
    virtual ~FileRuntime() {shutdownFileRuntime();}
    virtual double getCurrentFileCacheTargetSampleRate() const noexcept {return cacheTargetRate.load();}
    void requestRateReload() {
        {std::lock_guard lock(fileLoadMutex);rateReloadRequested.store(true);}
        fileLoadCv.notify_one();
    }
    void reloadCurrentSlotsForRate() {
        struct Slot {int index;std::vector<juce::File> files;FileLoadMode mode;juce::String recipe;};
        std::vector<Slot> jobs;
        {std::lock_guard lock(filePathMutex);for(int n=0;n<(int)fileSlots.size();++n){auto& slot=fileSlots[(size_t)n];Slot job{n,{},slot.loadMode,slot.currentRecipeXml};for(auto& path:slot.currentPaths)if(path.isNotEmpty())job.files.emplace_back(path);if(!job.files.empty())jobs.push_back(std::move(job));}}
        for(auto& job:jobs)setFileSlotPathsWithMode(job.index,job.files,job.mode,false,job.recipe);
    }
    virtual void requestExternalWakeEvent(bool=true) {if(externalWake)externalWake();}
    void setFileRuntimeCapabilities(bool pool,bool legacy)noexcept {usesSamplePool=pool;usesLegacyFileIO=legacy;}
    void initialiseFileSlots(std::vector<JsfxFileDecl> declarations) {
        fileDecls=std::move(declarations);
        if (fileDecls.empty())
            return;

        int maxIdx = -1;
        for (const auto& d : fileDecls)
            maxIdx = std::max (maxIdx, d.index0);

        if (maxIdx < 0)
            return;

        fileSlots.clear();
        fileSlots.resize ((size_t) (maxIdx + 1));

        for (int i = 0; i <= maxIdx; ++i)
            fileSlots[(size_t) i].name = "File " + juce::String (i);

        for (const auto& d : fileDecls)
        {
            auto& slot = fileSlots[(size_t) d.index0];
            slot.name = d.name;
            slot.description = d.description;
        }

        // Start the background loader thread (once).
        if (! fileLoadThread.joinable())
        {
            fileFormatManager.registerBasicFormats();
            fileLoadExit.store (false);
            fileLoadThread = std::thread ([this] { fileLoaderThreadMain(); });
        }

    }
    std::vector<juce::String> slotPaths(int index)const {
        std::lock_guard<std::mutex> lock(filePathMutex);
        return index>=0 && index<(int)fileSlots.size()?fileSlots[(size_t)index].currentPaths:std::vector<juce::String>{};
    }
    int fileSlotCount()const noexcept {return (int)fileSlots.size();}
    void setSlot(int slot,const juce::StringArray& paths) {
        std::vector<juce::File> files;for(auto& path:paths)files.emplace_back(path);
        setFileSlotPaths(slot,files,false);
    }
};
}
