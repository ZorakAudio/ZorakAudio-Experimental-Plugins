#include "DspJsfxMessageBus.h"
#include "DspJsfxRuntime.h"
#include "DspJsfxSharedMemory.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>

namespace za::jsfx
{

namespace
{
static constexpr std::uint32_t kIpcMagic = 0x5a4a4d42u; // ZJMB
static constexpr std::uint32_t kIpcVersion = 3u;
static constexpr std::uint32_t kIpcMaxInstances = 256u;
static constexpr std::uint32_t kIpcMaxChannelsPerInstance = 24u;
static constexpr std::uint32_t kIpcRingSize = 4096u;
static constexpr std::uint32_t kIpcMaxPayloadCells = 64u;
static constexpr std::uint64_t kIpcStaleSeqWindow = kIpcRingSize * 16ull;

static std::string hex64(std::uint64_t v)
{
    std::ostringstream os;
    os << std::hex << v;
    return os.str();
}

template <typename T>
static T loadPlain(const T& v) noexcept { return v; }

struct IpcChannelSlot
{
    std::atomic<std::uint64_t> hash { 0 };
    std::atomic<std::uint64_t> caps { 0 };
    std::atomic<std::uint32_t> subscribed { 0 };
    std::atomic<std::uint32_t> advertised { 0 };
};

struct IpcInstanceSlot
{
    std::atomic<std::uint64_t> instanceId { 0 };
    std::atomic<std::uint64_t> domainHash { 0 };
    std::atomic<std::uint64_t> lastSeenSeq { 0 };
    std::atomic<std::uint64_t> nameHandle { 0 }; // process-local handle; useful only for same-process peers
    std::atomic<std::uint64_t> mergedCaps { 0 };
    IpcChannelSlot channels[kIpcMaxChannelsPerInstance];
};

struct IpcMessageSlot
{
    // seq == 0 means writer is filling this slot or the slot is empty.
    std::atomic<std::uint64_t> seq { 0 };
    std::uint64_t sourceId = 0;
    std::uint64_t targetId = 0; // 0 = broadcast
    std::uint64_t channelHash = 0;
    std::uint32_t kind = 0;
    std::uint32_t payloadLen = 0;
    std::uint32_t direct = 0;
    std::uint32_t reserved = 0;
    double tag = 0.0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double payload[kIpcMaxPayloadCells] {};
};

struct IpcHeader
{
    std::atomic<std::uint32_t> magic { 0 };
    std::atomic<std::uint32_t> version { 0 };
    std::atomic<std::uint32_t> lock { 0 };
    std::atomic<std::uint32_t> reserved { 0 };
    std::atomic<std::uint64_t> domainHash { 0 };
    std::atomic<std::uint64_t> globalSeq { 0 };
    IpcInstanceSlot instances[kIpcMaxInstances];
    IpcMessageSlot messages[kIpcRingSize];
};

static bool tryLock(IpcHeader* h) noexcept
{
    if (h == nullptr)
        return false;

    for (int spins = 0; spins < 4096; ++spins)
    {
        std::uint32_t expected = 0;
        if (h->lock.compare_exchange_weak(expected, 1u, std::memory_order_acq_rel, std::memory_order_acquire))
            return true;
    }
    return false;
}

static void unlock(IpcHeader* h) noexcept
{
    if (h != nullptr)
        h->lock.store(0u, std::memory_order_release);
}

struct IpcLockGuard
{
    explicit IpcLockGuard(IpcHeader* headerIn) : h(headerIn), locked(tryLock(headerIn)) {}
    ~IpcLockGuard() { if (locked) unlock(h); }
    IpcHeader* h = nullptr;
    bool locked = false;
};

static bool channelMatches(const IpcInstanceSlot& inst, std::uint64_t channelHash, int role) noexcept
{
    const bool wantSub = role == 1 || role == 3 || role <= 0;
    const bool wantPub = role == 2 || role == 3 || role <= 0;

    for (std::uint32_t i = 0; i < kIpcMaxChannelsPerInstance; ++i)
    {
        const auto h = inst.channels[i].hash.load(std::memory_order_acquire);
        if (h != channelHash)
            continue;

        const bool isSub = inst.channels[i].subscribed.load(std::memory_order_acquire) != 0u;
        const bool isPub = inst.channels[i].advertised.load(std::memory_order_acquire) != 0u;
        return (wantSub && isSub) || (wantPub && isPub);
    }
    return false;
}

static bool isActiveInstance(const IpcInstanceSlot& inst, std::uint64_t domainHash, std::uint64_t nowSeq) noexcept
{
    const auto id = inst.instanceId.load(std::memory_order_acquire);
    if (id == 0)
        return false;
    if (inst.domainHash.load(std::memory_order_acquire) != domainHash)
        return false;

    const auto last = inst.lastSeenSeq.load(std::memory_order_acquire);
    return last == 0 || nowSeq <= last + kIpcStaleSeqWindow;
}

static void clearInstanceSlot(IpcInstanceSlot& inst) noexcept
{
    inst.instanceId.store(0, std::memory_order_release);
    inst.domainHash.store(0, std::memory_order_release);
    inst.lastSeenSeq.store(0, std::memory_order_release);
    inst.nameHandle.store(0, std::memory_order_release);
    inst.mergedCaps.store(0, std::memory_order_release);
    for (auto& ch : inst.channels)
    {
        ch.hash.store(0, std::memory_order_release);
        ch.caps.store(0, std::memory_order_release);
        ch.subscribed.store(0, std::memory_order_release);
        ch.advertised.store(0, std::memory_order_release);
    }
}

static bool messageTargetsRuntime(const IpcMessageSlot& slot,
                                  std::uint64_t instanceId,
                                  const std::unordered_set<std::uint64_t>& subscriptions) noexcept
{
    const auto source = loadPlain(slot.sourceId);
    const auto target = loadPlain(slot.targetId);
    const auto channel = loadPlain(slot.channelHash);

    if (target != 0)
        return target == instanceId;

    if (source == instanceId)
        return false;

    return subscriptions.find(channel) != subscriptions.end();
}

static DspJsfxMessage readMessageSlot(const IpcMessageSlot& slot)
{
    DspJsfxMessage msg;
    msg.kind = static_cast<DspJsfxMessageKind> (slot.kind);
    msg.channelHash = slot.channelHash;
    msg.sourceId = slot.sourceId;
    msg.targetId = slot.targetId;
    msg.direct = slot.direct != 0u;
    msg.tag = slot.tag;
    msg.a = slot.a;
    msg.b = slot.b;
    msg.c = slot.c;
    msg.d = slot.d;

    if (msg.kind == DspJsfxMessageKind::Buffer)
    {
        const auto len = std::min<std::uint32_t> (slot.payloadLen, kIpcMaxPayloadCells);
        msg.buffer.assign(slot.payload, slot.payload + len);
    }
    return msg;
}

static void writeMessageSlot(IpcMessageSlot& slot, std::uint64_t seq, const DspJsfxMessage& msg)
{
    slot.seq.store(0, std::memory_order_release);
    slot.sourceId = msg.sourceId;
    slot.targetId = msg.direct ? msg.targetId : 0;
    slot.channelHash = msg.channelHash;
    slot.kind = static_cast<std::uint32_t> (msg.kind);
    slot.direct = msg.direct ? 1u : 0u;
    slot.tag = msg.tag;
    slot.a = msg.a;
    slot.b = msg.b;
    slot.c = msg.c;
    slot.d = msg.d;

    if (msg.kind == DspJsfxMessageKind::Buffer)
    {
        const auto len = std::min<std::size_t> (msg.buffer.size(), kIpcMaxPayloadCells);
        slot.payloadLen = static_cast<std::uint32_t> (len);
        for (std::size_t i = 0; i < len; ++i)
            slot.payload[i] = msg.buffer[i];
        for (std::size_t i = len; i < kIpcMaxPayloadCells; ++i)
            slot.payload[i] = 0.0;
    }
    else
    {
        slot.payloadLen = 0;
    }

    slot.seq.store(seq, std::memory_order_release);
}
} // namespace

struct DspJsfxMessageBus::DomainState
{
    explicit DomainState(std::uint64_t hash) : domainHash(hash) {}

    std::uint64_t domainHash = 0;
    mutable DspJsfxSharedMemorySegment shm;
    mutable IpcHeader* header = nullptr;
    mutable bool openAttempted = false;
};

DspJsfxMessageBus& DspJsfxMessageBus::instance()
{
    static DspJsfxMessageBus bus;
    return bus;
}

DspJsfxMessageBus::DomainState* DspJsfxMessageBus::domainFor(std::uint64_t domainHash) const
{
    std::lock_guard<std::mutex> domainLock(domainsMutex_);

    auto it = domains_.find(domainHash);
    if (it == domains_.end())
    {
        auto inserted = domains_.emplace(domainHash, std::make_unique<DomainState>(domainHash));
        it = inserted.first;
    }

    auto* domain = it->second.get();
    if (domain == nullptr)
        return nullptr;

    if (! domain->openAttempted)
    {
        domain->openAttempted = true;
        bool created = false;
        const std::string objectName = std::string("msg_") + hex64(domainHash);
        if (domain->shm.openOrCreate(objectName, sizeof(IpcHeader), &created))
        {
            domain->header = static_cast<IpcHeader*> (domain->shm.data());

            const bool needsInit = created
                || domain->header->magic.load(std::memory_order_acquire) != kIpcMagic
                || domain->header->version.load(std::memory_order_acquire) != kIpcVersion
                || domain->header->domainHash.load(std::memory_order_acquire) != domainHash;

            if (needsInit)
            {
                std::memset(domain->header, 0, sizeof(IpcHeader));
                domain->header->domainHash.store(domainHash, std::memory_order_release);
                domain->header->version.store(kIpcVersion, std::memory_order_release);
                domain->header->magic.store(kIpcMagic, std::memory_order_release);
            }
        }
    }

    if (domain->header == nullptr)
        return nullptr;

    if (domain->header->magic.load(std::memory_order_acquire) != kIpcMagic
        || domain->header->version.load(std::memory_order_acquire) != kIpcVersion)
        return nullptr;

    return domain;
}

void DspJsfxMessageBus::registerRuntime(DspJsfxRuntime* runtime,
                                        std::uint64_t instanceId,
                                        std::uint64_t domainHash,
                                        const std::string& uid)
{
    InstanceRecord rec;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& dst = instances_[instanceId];
        dst.runtime = runtime;
        dst.instanceId = instanceId;
        dst.domainHash = domainHash;
        dst.uid = uid;
        rec = dst;
    }
    upsertIpcInstance(rec);
}

void DspJsfxMessageBus::unregisterRuntime(std::uint64_t instanceId)
{
    std::uint64_t oldDomain = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it != instances_.end())
        {
            oldDomain = it->second.domainHash;
            instances_.erase(it);
        }
    }

    if (oldDomain != 0)
        removeIpcInstance(instanceId, oldDomain);
}

void DspJsfxMessageBus::updateDomain(std::uint64_t instanceId, std::uint64_t domainHash)
{
    InstanceRecord rec;
    std::uint64_t oldDomain = 0;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it != instances_.end())
        {
            oldDomain = it->second.domainHash;
            it->second.domainHash = domainHash;
            rec = it->second;
            found = true;
        }
    }
    if (! found)
        return;
    if (oldDomain != 0 && oldDomain != domainHash)
        removeIpcInstance(instanceId, oldDomain);
    upsertIpcInstance(rec);
}

void DspJsfxMessageBus::updateNameHandle(std::uint64_t instanceId, std::int64_t nameHandle)
{
    InstanceRecord rec;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it != instances_.end())
        {
            it->second.nameHandle = nameHandle;
            rec = it->second;
            found = true;
        }
    }
    if (found)
        upsertIpcInstance(rec);
}

void DspJsfxMessageBus::updateSubscription(std::uint64_t instanceId, std::uint64_t channelHash, bool subscribed)
{
    InstanceRecord rec;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it == instances_.end())
            return;
        if (subscribed)
            it->second.subscriptions.insert(channelHash);
        else
            it->second.subscriptions.erase(channelHash);
        rec = it->second;
        found = true;
    }
    if (found)
        upsertIpcInstance(rec);
}

void DspJsfxMessageBus::updateAdvertisement(std::uint64_t instanceId, std::uint64_t channelHash, std::uint64_t caps)
{
    InstanceRecord rec;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it == instances_.end())
            return;
        if (caps == 0)
            it->second.advertisedCaps.erase(channelHash);
        else
            it->second.advertisedCaps[channelHash] = caps;
        rec = it->second;
        found = true;
    }
    if (found)
        upsertIpcInstance(rec);
}

bool DspJsfxMessageBus::matchesRole(const InstanceRecord& rec, std::uint64_t channelHash, int role)
{
    const bool isSub = rec.subscriptions.find(channelHash) != rec.subscriptions.end();
    const bool isPub = rec.advertisedCaps.find(channelHash) != rec.advertisedCaps.end();
    switch (role)
    {
        case 1: return isSub;
        case 2: return isPub;
        case 3: return isSub || isPub;
        default: return isSub || isPub;
    }
}

void DspJsfxMessageBus::upsertIpcInstance(const InstanceRecord& rec) const
{
    auto* domain = domainFor(rec.domainHash);
    if (domain == nullptr || domain->header == nullptr)
        return;

    IpcLockGuard guard(domain->header);
    if (! guard.locked)
        return;

    auto* h = domain->header;
    const auto nowSeq = h->globalSeq.load(std::memory_order_acquire);
    IpcInstanceSlot* selected = nullptr;
    IpcInstanceSlot* firstStale = nullptr;

    for (auto& slot : h->instances)
    {
        const auto id = slot.instanceId.load(std::memory_order_acquire);
        if (id == rec.instanceId)
        {
            selected = &slot;
            break;
        }
        if (firstStale == nullptr && ! isActiveInstance(slot, rec.domainHash, nowSeq))
            firstStale = &slot;
    }

    if (selected == nullptr)
        selected = firstStale;
    if (selected == nullptr)
        return;

    selected->instanceId.store(rec.instanceId, std::memory_order_release);
    selected->domainHash.store(rec.domainHash, std::memory_order_release);
    selected->lastSeenSeq.store(nowSeq, std::memory_order_release);
    selected->nameHandle.store(static_cast<std::uint64_t> (rec.nameHandle), std::memory_order_release);

    for (auto& ch : selected->channels)
    {
        ch.hash.store(0, std::memory_order_release);
        ch.caps.store(0, std::memory_order_release);
        ch.subscribed.store(0, std::memory_order_release);
        ch.advertised.store(0, std::memory_order_release);
    }

    std::uint64_t mergedCaps = 0;
    std::uint32_t out = 0;
    for (const auto channelHash : rec.subscriptions)
    {
        if (out >= kIpcMaxChannelsPerInstance)
            break;
        auto& ch = selected->channels[out++];
        ch.hash.store(channelHash, std::memory_order_release);
        ch.caps.store(0, std::memory_order_release);
        ch.subscribed.store(1u, std::memory_order_release);
        ch.advertised.store(0u, std::memory_order_release);
    }

    for (const auto& kv : rec.advertisedCaps)
    {
        mergedCaps |= kv.second;

        IpcChannelSlot* existing = nullptr;
        for (std::uint32_t i = 0; i < out; ++i)
        {
            if (selected->channels[i].hash.load(std::memory_order_acquire) == kv.first)
            {
                existing = &selected->channels[i];
                break;
            }
        }
        if (existing == nullptr)
        {
            if (out >= kIpcMaxChannelsPerInstance)
                continue;
            existing = &selected->channels[out++];
            existing->hash.store(kv.first, std::memory_order_release);
            existing->subscribed.store(0u, std::memory_order_release);
        }
        existing->caps.store(kv.second, std::memory_order_release);
        existing->advertised.store(kv.second != 0 ? 1u : 0u, std::memory_order_release);
    }

    selected->mergedCaps.store(mergedCaps, std::memory_order_release);
}

void DspJsfxMessageBus::removeIpcInstance(std::uint64_t instanceId, std::uint64_t domainHash) const
{
    auto* domain = domainFor(domainHash);
    if (domain == nullptr || domain->header == nullptr)
        return;

    IpcLockGuard guard(domain->header);
    if (! guard.locked)
        return;

    for (auto& slot : domain->header->instances)
    {
        if (slot.instanceId.load(std::memory_order_acquire) == instanceId)
        {
            clearInstanceSlot(slot);
            break;
        }
    }
}

void DspJsfxMessageBus::flushOutbox(std::uint64_t instanceId,
                                    std::uint64_t domainHash,
                                    const std::vector<DspJsfxMessage>& outbox,
                                    std::unordered_map<std::uint64_t, std::uint64_t>& droppedByChannel)
{
    if (outbox.empty())
        return;

    auto* domain = domainFor(domainHash);
    if (domain == nullptr || domain->header == nullptr)
    {
        for (const auto& msg : outbox)
            droppedByChannel[msg.channelHash] += 1u;
        return;
    }

    IpcLockGuard guard(domain->header);
    if (! guard.locked)
    {
        for (const auto& msg : outbox)
            droppedByChannel[msg.channelHash] += 1u;
        return;
    }

    auto* h = domain->header;
    const auto nowSeq = h->globalSeq.load(std::memory_order_acquire);

    for (const auto& input : outbox)
    {
        if (input.kind == DspJsfxMessageKind::Buffer && input.buffer.size() > kIpcMaxPayloadCells)
        {
            droppedByChannel[input.channelHash] += 1u;
            continue;
        }

        bool hasTarget = false;
        if (input.direct)
        {
            for (const auto& slot : h->instances)
            {
                if (isActiveInstance(slot, domainHash, nowSeq)
                    && slot.instanceId.load(std::memory_order_acquire) == input.targetId)
                {
                    hasTarget = true;
                    break;
                }
            }
        }
        else
        {
            for (const auto& slot : h->instances)
            {
                const auto dst = slot.instanceId.load(std::memory_order_acquire);
                if (dst == 0 || dst == instanceId)
                    continue;
                if (isActiveInstance(slot, domainHash, nowSeq) && channelMatches(slot, input.channelHash, 1))
                {
                    hasTarget = true;
                    break;
                }
            }
        }

        if (! hasTarget)
        {
            droppedByChannel[input.channelHash] += 1u;
            continue;
        }

        DspJsfxMessage msg = input;
        msg.sourceId = instanceId;
        if (! msg.direct)
            msg.targetId = 0;

        const auto seq = h->globalSeq.fetch_add(1u, std::memory_order_acq_rel) + 1u;
        auto& slot = h->messages[seq % kIpcRingSize];
        writeMessageSlot(slot, seq, msg);
    }
}

void DspJsfxMessageBus::collectInbox(std::uint64_t instanceId,
                                     std::uint64_t domainHash,
                                     const std::unordered_set<std::uint64_t>& subscriptions,
                                     std::uint64_t& lastReadSeq,
                                     std::unordered_map<std::uint64_t, std::deque<DspJsfxMessage>>& readyInbox,
                                     std::unordered_map<std::uint64_t, std::uint64_t>& droppedByChannel)
{
    auto* domain = domainFor(domainHash);
    if (domain == nullptr || domain->header == nullptr)
        return;

    auto* h = domain->header;
    const auto newestSeq = h->globalSeq.load(std::memory_order_acquire);
    if (newestSeq <= lastReadSeq)
    {
        InstanceRecord self;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = instances_.find(instanceId);
            if (it != instances_.end())
            {
                self = it->second;
                found = true;
            }
        }
        if (found)
            upsertIpcInstance(self);
        return;
    }

    auto firstSeq = lastReadSeq + 1u;
    if (newestSeq > kIpcRingSize && firstSeq + kIpcRingSize <= newestSeq)
    {
        // Receiver lagged past the ring window. Resume from the oldest retained slot.
        for (const auto ch : subscriptions)
            droppedByChannel[ch] += 1u;
        firstSeq = newestSeq - kIpcRingSize + 1u;
    }

    std::uint64_t maxSeen = lastReadSeq;
    for (auto seq = firstSeq; seq <= newestSeq; ++seq)
    {
        const auto& slot = h->messages[seq % kIpcRingSize];
        const auto slotSeq = slot.seq.load(std::memory_order_acquire);
        if (slotSeq != seq)
            continue;

        if (! messageTargetsRuntime(slot, instanceId, subscriptions))
            continue;

        auto msg = readMessageSlot(slot);
        readyInbox[msg.channelHash].push_back(std::move(msg));
        maxSeen = seq;
    }

    lastReadSeq = newestSeq;

    InstanceRecord self;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instances_.find(instanceId);
        if (it != instances_.end())
        {
            self = it->second;
            found = true;
        }
    }
    if (found)
        upsertIpcInstance(self);

    (void) maxSeen;
}

bool DspJsfxMessageBus::hasPendingFor(std::uint64_t instanceId,
                                      std::uint64_t domainHash,
                                      const std::unordered_set<std::uint64_t>& subscriptions,
                                      std::uint64_t lastReadSeq)
{
    auto* domain = domainFor(domainHash);
    if (domain == nullptr || domain->header == nullptr)
        return false;

    auto* h = domain->header;
    const auto newestSeq = h->globalSeq.load(std::memory_order_acquire);
    if (newestSeq <= lastReadSeq)
        return false;

    auto firstSeq = lastReadSeq + 1u;
    if (newestSeq > kIpcRingSize && firstSeq + kIpcRingSize <= newestSeq)
        firstSeq = newestSeq - kIpcRingSize + 1u;

    for (auto seq = firstSeq; seq <= newestSeq; ++seq)
    {
        const auto& slot = h->messages[seq % kIpcRingSize];
        if (slot.seq.load(std::memory_order_acquire) != seq)
            continue;
        if (messageTargetsRuntime(slot, instanceId, subscriptions))
            return true;
    }
    return false;
}

std::size_t DspJsfxMessageBus::peerCount(std::uint64_t domainHash, std::uint64_t channelHash, int role) const
{
    auto* domain = domainFor(domainHash);
    if (domain != nullptr && domain->header != nullptr)
    {
        const auto nowSeq = domain->header->globalSeq.load(std::memory_order_acquire);
        std::size_t count = 0;
        for (const auto& slot : domain->header->instances)
        {
            if (! isActiveInstance(slot, domainHash, nowSeq))
                continue;
            if (channelMatches(slot, channelHash, role))
                ++count;
        }
        return count;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t count = 0;
    for (const auto& kv : instances_)
        if (kv.second.domainHash == domainHash && matchesRole(kv.second, channelHash, role))
            ++count;
    return count;
}

std::uint64_t DspJsfxMessageBus::peerId(std::uint64_t domainHash, std::uint64_t channelHash, int role, std::size_t index) const
{
    std::vector<std::uint64_t> ids;

    auto* domain = domainFor(domainHash);
    if (domain != nullptr && domain->header != nullptr)
    {
        const auto nowSeq = domain->header->globalSeq.load(std::memory_order_acquire);
        ids.reserve(kIpcMaxInstances);
        for (const auto& slot : domain->header->instances)
        {
            if (! isActiveInstance(slot, domainHash, nowSeq))
                continue;
            if (channelMatches(slot, channelHash, role))
                ids.push_back(slot.instanceId.load(std::memory_order_acquire));
        }
    }
    else
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ids.reserve(instances_.size());
        for (const auto& kv : instances_)
            if (kv.second.domainHash == domainHash && matchesRole(kv.second, channelHash, role))
                ids.push_back(kv.first);
    }

    std::sort(ids.begin(), ids.end());
    if (index >= ids.size())
        return 0;
    return ids[index];
}

bool DspJsfxMessageBus::peerInfo(std::uint64_t peerId,
                                 std::int64_t* nameHandle,
                                 std::string* uid,
                                 std::uint64_t* caps,
                                 bool* alive) const
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = instances_.find(peerId);
        if (it != instances_.end())
        {
            if (nameHandle != nullptr)
                *nameHandle = it->second.nameHandle;
            if (uid != nullptr)
                *uid = it->second.uid;
            if (caps != nullptr)
            {
                std::uint64_t mergedCaps = 0;
                for (const auto& kv : it->second.advertisedCaps)
                    mergedCaps |= kv.second;
                *caps = mergedCaps;
            }
            if (alive != nullptr)
                *alive = (it->second.runtime != nullptr);
            return true;
        }
    }

    // Cross-process fallback: peer is known by ID/caps/alive, but string handles
    // are process-local and cannot safely cross IPC boundaries.
    std::vector<DomainState*> domainSnapshot;
    {
        std::lock_guard<std::mutex> domainLock(domainsMutex_);
        domainSnapshot.reserve(domains_.size());
        for (const auto& kv : domains_)
            if (kv.second != nullptr)
                domainSnapshot.push_back(kv.second.get());
    }

    for (auto* domain : domainSnapshot)
    {
        if (domain == nullptr || domain->header == nullptr)
            continue;

        const auto nowSeq = domain->header->globalSeq.load(std::memory_order_acquire);
        for (const auto& slot : domain->header->instances)
        {
            if (slot.instanceId.load(std::memory_order_acquire) != peerId)
                continue;

            const bool live = isActiveInstance(slot, slot.domainHash.load(std::memory_order_acquire), nowSeq);
            if (nameHandle != nullptr)
                *nameHandle = 0;
            if (uid != nullptr)
                *uid = std::string("ipc-") + hex64(peerId);
            if (caps != nullptr)
                *caps = slot.mergedCaps.load(std::memory_order_acquire);
            if (alive != nullptr)
                *alive = live;
            return true;
        }
    }

    return false;
}

} // namespace za::jsfx
