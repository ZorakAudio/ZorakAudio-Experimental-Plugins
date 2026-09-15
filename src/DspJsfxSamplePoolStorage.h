#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace za::jsfx
{
// Immutable-generation ownership, independent of decoding/JUCE. Generation must
// provide publicationRequestId and decodedBytes. Reclamation is worker-only;
// readers never increment shared_ptr counts or destroy audio allocations.
template<class Generation>
class SamplePoolStorage
{
public:
    class ReadBatch
    {
    public:
        ReadBatch() noexcept : previous_(current_) { current_ = this; }
        ~ReadBatch()
        {
            current_ = previous_;
            for (std::size_t i = 0; i < count_; ++i)
                pins_[i].store->readers_.fetch_sub(1);
        }
        ReadBatch(const ReadBatch&) = delete;
        ReadBatch& operator=(const ReadBatch&) = delete;
    private:
        friend class SamplePoolStorage;
        struct Pin { const SamplePoolStorage* store = nullptr; const Generation* generation = nullptr; };
        Pin* find(const SamplePoolStorage& store) noexcept
        {
            if (last_ != nullptr && last_->store == &store)
                return last_;
            for (std::size_t i = 0; i < count_; ++i)
                if (pins_[i].store == &store)
                    return last_ = &pins_[i];
            return nullptr;
        }
        const Generation* pin(const SamplePoolStorage& store, bool& pinned) noexcept
        {
            // Nested callbacks inherit an outer pin instead of seeing a different bank.
            for (auto* batch = this; batch != nullptr; batch = batch->previous_)
                if (auto* entry = batch->find(store))
                {
                    pinned = true;
                    return entry->generation;
                }
            if (count_ == pins_.size())
            {
                pinned = false; // excess pools safely use per-call protection
                return nullptr;
            }
            store.readers_.fetch_add(1); // SC: protect before loading snapshot
            auto& entry = pins_[count_++];
            entry.store = &store;
            entry.generation = store.active_.load();
            last_ = &entry;
            pinned = true;
            return entry.generation;
        }
        void refresh(const SamplePoolStorage& store, const Generation* gen) noexcept
        {
            for (auto* batch = this; batch != nullptr; batch = batch->previous_)
                if (auto* entry = batch->find(store))
                    entry->generation = gen;
        }
        inline static thread_local ReadBatch* current_ = nullptr;
        ReadBatch* previous_ = nullptr;
        std::array<Pin, 64> pins_ {};
        std::size_t count_ = 0;
        Pin* last_ = nullptr;
    };

    class ReaderScope
    {
    public:
        explicit ReaderScope(const SamplePoolStorage& store) noexcept : store_(store)
        {
            bool pinned = false;
            if (ReadBatch::current_ != nullptr)
                generation = ReadBatch::current_->pin(store, pinned);
            if (!pinned)
            {
                individual_ = true;
                store_.readers_.fetch_add(1);
                generation = store_.active_.load();
            }
        }
        ~ReaderScope() { if (individual_) store_.readers_.fetch_sub(1); }
        ReaderScope(const ReaderScope&) = delete;
        ReaderScope& operator=(const ReaderScope&) = delete;
        const Generation* generation = nullptr;
    private:
        const SamplePoolStorage& store_;
        bool individual_ = false;
    };

    SamplePoolStorage() = default;
    SamplePoolStorage(const SamplePoolStorage&) = delete;
    SamplePoolStorage& operator=(const SamplePoolStorage&) = delete;
    // Owner must join its worker and finish all readers before destruction.

    void requestStarted(std::uint64_t id) noexcept
    {
        auto previous = requestedId_.load(std::memory_order_acquire);
        while (id > previous && !requestedId_.compare_exchange_weak(previous, id, std::memory_order_acq_rel)) {}
    }
    bool deferred() const noexcept { return deferred_.load(); }
    void setDeferred(bool value) noexcept
    {
        deferred_.store(value);
        if (!value)
            adoptPending(); // worker retries if this nonblocking attempt loses
    }
    int adoptPending() noexcept
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock())
            return -1;
        int adopted = 0;
        if (const auto* next = pending_.exchange(nullptr))
        {
            const auto* current = active_.load();
            if (current == nullptr || next->publicationRequestId > current->publicationRequestId)
            {
                active_.store(next);
                adopted = 1;
            }
        }
        // Explicit client adoption declares a safe boundary for this thread.
        // Other simultaneous batches continue to hold their own older snapshot.
        if (ReadBatch::current_ != nullptr)
            ReadBatch::current_->refresh(*this, active_.load());
        return adopted;
    }
    bool publish(std::shared_ptr<Generation> gen, std::uint64_t requestId)
    {
        if (!gen)
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (requestId != requestedId_.load(std::memory_order_acquire) || requestId <= publishedId_)
            return false;
        gen->publicationRequestId = requestId;
        generations_.push_back(gen);
        publishedId_ = requestId;
        if (deferred_.load())
            pending_.store(gen.get());
        else
        {
            // setDeferred(false) can fail try_lock. A newer immediate publish
            // must discard that old pending pointer or worker retry rolls back.
            pending_.store(nullptr);
            active_.store(gen.get());
        }
        bytes_.fetch_add(gen->decodedBytes, std::memory_order_acq_rel);
        return true;
    }
    void reclaimRetired() // worker/non-realtime owner only
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto* active = active_.load();
        const auto* pending = pending_.load();
        if (readers_.load() != 0)
            return;
        // A reader arriving after this SC check can only obtain active (which is
        // retained). No publication/adoption can interleave while mutex_ is held.
        generations_.erase(std::remove_if(generations_.begin(), generations_.end(),
            [active, pending](const auto& gen) { return gen.get() != active && gen.get() != pending; }), generations_.end());
        std::uint64_t bytes = 0;
        for (const auto& gen : generations_)
            bytes += gen->decodedBytes;
        bytes_.store(bytes, std::memory_order_release);
    }
    std::uint64_t decodedBytes() const noexcept { return bytes_.load(std::memory_order_acquire); }
private:
    // Test friend exposes counters/lock only to the deterministic native fixtures;
    // production code uses the same public API above.
    friend struct SamplePoolStorageTestAccess;
    mutable std::atomic<unsigned> readers_ { 0 };
    std::atomic<bool> deferred_ { false };
    std::atomic<const Generation*> active_ { nullptr }, pending_ { nullptr };
    std::atomic<std::uint64_t> requestedId_ { 0 }, bytes_ { 0 };
    std::uint64_t publishedId_ = 0; // mutex protected
    std::mutex mutex_;
    std::vector<std::shared_ptr<const Generation>> generations_;
};
} // namespace za::jsfx
