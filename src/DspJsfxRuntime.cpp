#include "DspJsfxRuntime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>

namespace za::jsfx
{

namespace
{
std::atomic<std::uint64_t> gNextInstanceId { 1 };
std::mutex gRuntimeRegistryMutex;
std::unordered_map<DSPJSFX_State*, DspJsfxRuntime*> gRuntimeRegistry;

static std::string makeUid(std::uint64_t instanceId)
{
    char buf[48] = {};
    std::snprintf(buf, sizeof(buf), "jsfx-%016llx", static_cast<unsigned long long> (instanceId));
    return std::string(buf);
}

static int clampIntArg(double v) noexcept
{
    if (! std::isfinite(v))
        return 0;
    if (v <= static_cast<double> (std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    if (v >= static_cast<double> (std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    return static_cast<int> (std::llround(v));
}
} // namespace

DspJsfxRuntime::DspJsfxRuntime()
    : instanceId_ (gNextInstanceId.fetch_add(1u, std::memory_order_relaxed)),
      uid_ (makeUid(instanceId_))
{
}

DspJsfxRuntime::~DspJsfxRuntime()
{
    detachFromState();
}

void DspJsfxRuntime::attachToState(DSPJSFX_State* st)
{
    if (state_ == st)
        return;

    detachFromState();
    state_ = st;
    {
        std::lock_guard<std::mutex> lock(gRuntimeRegistryMutex);
        gRuntimeRegistry[state_] = this;
    }
    DspJsfxMessageBus::instance().registerRuntime(this, instanceId_, domainHash_, uid_);
}

void DspJsfxRuntime::detachFromState()
{
    if (state_ != nullptr)
    {
        std::lock_guard<std::mutex> lock(gRuntimeRegistryMutex);
        gRuntimeRegistry.erase(state_);
    }
    DspJsfxMessageBus::instance().unregisterRuntime(instanceId_);
    state_ = nullptr;
}

DspJsfxRuntime* DspJsfxRuntime::findForState(DSPJSFX_State* st) noexcept
{
    std::lock_guard<std::mutex> lock(gRuntimeRegistryMutex);
    const auto it = gRuntimeRegistry.find(st);
    return it != gRuntimeRegistry.end() ? it->second : nullptr;
}

void DspJsfxRuntime::reset()
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    readyInbox_.clear();
    pendingInbox_.clear();
    outbox_.clear();
    droppedByChannel_.clear();
    lastMessageLength_ = 0;
}

void DspJsfxRuntime::beginBlock(DSPJSFX_State&, int)
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    for (auto& kv : pendingInbox_)
    {
        auto& dst = readyInbox_[kv.first];
        auto& src = kv.second;
        while (! src.empty())
        {
            dst.push_back(std::move(src.front()));
            src.pop_front();
        }
    }
}

void DspJsfxRuntime::endBlock(DSPJSFX_State&)
{
    if (! outbox_.empty())
    {
        DspJsfxMessageBus::instance().flushOutbox(instanceId_, domainHash_, outbox_, droppedByChannel_);
        outbox_.clear();
    }
}

bool DspJsfxRuntime::hasReadyMessages() const noexcept
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    for (const auto& kv : readyInbox_)
        if (! kv.second.empty())
            return true;
    for (const auto& kv : pendingInbox_)
        if (! kv.second.empty())
            return true;
    return false;
}

bool DspJsfxRuntime::hasPendingForThisInstance() const noexcept
{
    return hasReadyMessages();
}

bool DspJsfxRuntime::enqueuePendingMessage(const DspJsfxMessage& message)
{
    constexpr std::size_t kMaxQueuedMessages = 1024;
    constexpr std::size_t kMaxQueuedBufferCells = 65536;

    std::lock_guard<std::mutex> lock(inboxMutex_);
    std::size_t queuedMessages = 0;
    std::size_t queuedBufferCells = 0;
    for (const auto& kv : pendingInbox_)
    {
        queuedMessages += kv.second.size();
        for (const auto& msg : kv.second)
            queuedBufferCells += msg.buffer.size();
    }
    if (queuedMessages >= kMaxQueuedMessages)
        return false;
    if (queuedBufferCells + message.buffer.size() > kMaxQueuedBufferCells)
        return false;
    pendingInbox_[message.channelHash].push_back(message);
    return true;
}

bool DspJsfxRuntime::joinDomain(std::uint64_t domainHash)
{
    if (domainHash == 0)
        domainHash = 0x9ae16a3b2f90404full;
    domainHash_ = domainHash;
    DspJsfxMessageBus::instance().updateDomain(instanceId_, domainHash_);
    return true;
}

bool DspJsfxRuntime::setNameHandle(std::int64_t handle)
{
    nameHandle_ = handle;
    DspJsfxMessageBus::instance().updateNameHandle(instanceId_, nameHandle_);
    return true;
}

bool DspJsfxRuntime::subscribe(std::uint64_t channelHash)
{
    subscriptions_.insert(channelHash);
    DspJsfxMessageBus::instance().updateSubscription(instanceId_, channelHash, true);
    return true;
}

bool DspJsfxRuntime::unsubscribe(std::uint64_t channelHash)
{
    subscriptions_.erase(channelHash);
    DspJsfxMessageBus::instance().updateSubscription(instanceId_, channelHash, false);
    return true;
}

bool DspJsfxRuntime::advertise(std::uint64_t channelHash, std::uint64_t caps)
{
    if (caps == 0)
        advertisedCaps_.erase(channelHash);
    else
        advertisedCaps_[channelHash] = caps;
    DspJsfxMessageBus::instance().updateAdvertisement(instanceId_, channelHash, caps);
    return true;
}

bool DspJsfxRuntime::enqueueMessage(DspJsfxMessage&& message)
{
    constexpr std::size_t kMaxOutboxMessages = 1024;
    constexpr std::size_t kMaxOutboxBufferCells = 65536;

    std::size_t bufferCells = 0;
    for (const auto& msg : outbox_)
        bufferCells += msg.buffer.size();
    if (outbox_.size() >= kMaxOutboxMessages || bufferCells + message.buffer.size() > kMaxOutboxBufferCells)
    {
        droppedByChannel_[message.channelHash] += 1u;
        return false;
    }
    outbox_.push_back(std::move(message));
    return true;
}

bool DspJsfxRuntime::queueScalar(std::uint64_t channelHash, double tag, double a, double b, double c, double d)
{
    DspJsfxMessage msg;
    msg.kind = DspJsfxMessageKind::Scalar;
    msg.channelHash = channelHash;
    msg.sourceId = instanceId_;
    msg.tag = tag;
    msg.a = a;
    msg.b = b;
    msg.c = c;
    msg.d = d;
    return enqueueMessage(std::move(msg));
}

bool DspJsfxRuntime::queueScalarTo(std::uint64_t targetId, std::uint64_t channelHash, double tag, double a, double b, double c, double d)
{
    DspJsfxMessage msg;
    msg.kind = DspJsfxMessageKind::Scalar;
    msg.channelHash = channelHash;
    msg.sourceId = instanceId_;
    msg.targetId = targetId;
    msg.direct = true;
    msg.tag = tag;
    msg.a = a;
    msg.b = b;
    msg.c = c;
    msg.d = d;
    return enqueueMessage(std::move(msg));
}

bool DspJsfxRuntime::queueBuffer(std::uint64_t channelHash, double tag, const double* src, int len)
{
    if (src == nullptr || len <= 0)
        return false;
    DspJsfxMessage msg;
    msg.kind = DspJsfxMessageKind::Buffer;
    msg.channelHash = channelHash;
    msg.sourceId = instanceId_;
    msg.tag = tag;
    msg.buffer.assign(src, src + len);
    return enqueueMessage(std::move(msg));
}

bool DspJsfxRuntime::queueBufferTo(std::uint64_t targetId, std::uint64_t channelHash, double tag, const double* src, int len)
{
    if (src == nullptr || len <= 0)
        return false;
    DspJsfxMessage msg;
    msg.kind = DspJsfxMessageKind::Buffer;
    msg.channelHash = channelHash;
    msg.sourceId = instanceId_;
    msg.targetId = targetId;
    msg.direct = true;
    msg.tag = tag;
    msg.buffer.assign(src, src + len);
    return enqueueMessage(std::move(msg));
}

std::deque<DspJsfxMessage>* DspJsfxRuntime::inboxFor(std::uint64_t channelHash)
{
    const auto it = readyInbox_.find(channelHash);
    return it != readyInbox_.end() ? &it->second : nullptr;
}

const std::deque<DspJsfxMessage>* DspJsfxRuntime::inboxFor(std::uint64_t channelHash) const
{
    const auto it = readyInbox_.find(channelHash);
    return it != readyInbox_.end() ? &it->second : nullptr;
}

int DspJsfxRuntime::avail(std::uint64_t channelHash) const
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    const auto* q = inboxFor(channelHash);
    return q != nullptr ? static_cast<int> (q->size()) : 0;
}

int DspJsfxRuntime::kind(std::uint64_t channelHash) const
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    const auto* q = inboxFor(channelHash);
    if (q == nullptr || q->empty())
        return 0;
    return static_cast<int> (q->front().kind);
}

int DspJsfxRuntime::recvScalar(std::uint64_t channelHash, double* src, double* tag, double* a, double* b, double* c, double* d)
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    auto* q = inboxFor(channelHash);
    if (q == nullptr || q->empty() || q->front().kind != DspJsfxMessageKind::Scalar)
        return 0;
    const auto msg = std::move(q->front());
    q->pop_front();
    if (src != nullptr) *src = static_cast<double> (msg.sourceId);
    if (tag != nullptr) *tag = msg.tag;
    if (a != nullptr) *a = msg.a;
    if (b != nullptr) *b = msg.b;
    if (c != nullptr) *c = msg.c;
    if (d != nullptr) *d = msg.d;
    lastMessageLength_ = 0;
    return 1;
}

int DspJsfxRuntime::recvBuffer(std::uint64_t channelHash, double* src, double* tag, double* dst, int maxLen, DSPJSFX_State&)
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    auto* q = inboxFor(channelHash);
    if (q == nullptr || q->empty() || q->front().kind != DspJsfxMessageKind::Buffer)
        return 0;
    auto msg = std::move(q->front());
    q->pop_front();
    if (src != nullptr) *src = static_cast<double> (msg.sourceId);
    if (tag != nullptr) *tag = msg.tag;
    if (dst != nullptr && maxLen > 0)
    {
        const int copyLen = std::min<int> (maxLen, static_cast<int> (msg.buffer.size()));
        for (int i = 0; i < copyLen; ++i)
            dst[i] = msg.buffer[static_cast<std::size_t> (i)];
    }
    lastMessageLength_ = static_cast<int> (msg.buffer.size());
    if (maxLen >= static_cast<int> (msg.buffer.size()))
        return static_cast<int> (msg.buffer.size());
    return -static_cast<int> (msg.buffer.size());
}

int DspJsfxRuntime::clear(std::uint64_t channelHash)
{
    std::lock_guard<std::mutex> lock(inboxMutex_);
    auto* q = inboxFor(channelHash);
    if (q == nullptr)
        return 0;
    const int cleared = static_cast<int> (q->size());
    q->clear();
    return cleared;
}

double DspJsfxRuntime::dropped(std::uint64_t channelHash) const
{
    const auto it = droppedByChannel_.find(channelHash);
    return static_cast<double> (it != droppedByChannel_.end() ? it->second : 0u);
}

double DspJsfxRuntime::peerCount(std::uint64_t channelHash, int role) const
{
    return static_cast<double> (DspJsfxMessageBus::instance().peerCount(domainHash_, channelHash, role));
}

double DspJsfxRuntime::peerId(std::uint64_t channelHash, int role, int index) const
{
    if (index < 0)
        return 0.0;
    return static_cast<double> (DspJsfxMessageBus::instance().peerId(domainHash_, channelHash, role, static_cast<std::size_t> (index)));
}

bool DspJsfxRuntime::peerNameHandle(std::uint64_t peerIdValue, std::int64_t* outHandle) const
{
    return DspJsfxMessageBus::instance().peerInfo(peerIdValue, outHandle, nullptr, nullptr, nullptr);
}

bool DspJsfxRuntime::peerUid(std::uint64_t peerIdValue, std::string* outUid) const
{
    return DspJsfxMessageBus::instance().peerInfo(peerIdValue, nullptr, outUid, nullptr, nullptr);
}

double DspJsfxRuntime::peerCaps(std::uint64_t peerIdValue) const
{
    std::uint64_t caps = 0;
    if (! DspJsfxMessageBus::instance().peerInfo(peerIdValue, nullptr, nullptr, &caps, nullptr))
        return 0.0;
    return static_cast<double> (caps);
}

double DspJsfxRuntime::peerAlive(std::uint64_t peerIdValue) const
{
    bool alive = false;
    if (! DspJsfxMessageBus::instance().peerInfo(peerIdValue, nullptr, nullptr, nullptr, &alive))
        return 0.0;
    return alive ? 1.0 : 0.0;
}

bool DspJsfxRuntime::gmemAttach(std::uint64_t nameHash, std::uint64_t requestedCells)
{
    return gmem_.attach(domainHash_, nameHash, requestedCells, instanceId_);
}

double DspJsfxRuntime::gmemSize() const noexcept
{
    return static_cast<double> (gmem_.cellCount());
}

double DspJsfxRuntime::gmemLoad(double idx) const noexcept
{
    return gmem_.load(idx);
}

double DspJsfxRuntime::gmemStore(double idx, double value) noexcept
{
    return gmem_.store(idx, value, instanceId_);
}

double DspJsfxRuntime::gmemGet(DSPJSFX_State& st, double dstBase, double srcIdx, double count) noexcept
{
    return static_cast<double> (gmem_.bulkGet(st, clampIntArg(dstBase), clampIntArg(srcIdx), clampIntArg(count)));
}

double DspJsfxRuntime::gmemPut(DSPJSFX_State& st, double dstIdx, double srcBase, double count) noexcept
{
    return static_cast<double> (gmem_.bulkPut(st, clampIntArg(dstIdx), clampIntArg(srcBase), clampIntArg(count), instanceId_));
}

double DspJsfxRuntime::gmemFill(double dstIdx, double value, double count) noexcept
{
    return static_cast<double> (gmem_.fill(clampIntArg(dstIdx), value, clampIntArg(count), instanceId_));
}

double DspJsfxRuntime::gmemZero(double dstIdx, double count) noexcept
{
    return static_cast<double> (gmem_.zero(clampIntArg(dstIdx), clampIntArg(count), instanceId_));
}

double DspJsfxRuntime::gmemCopy(double dstIdx, double srcIdx, double count) noexcept
{
    return static_cast<double> (gmem_.copy(clampIntArg(dstIdx), clampIntArg(srcIdx), clampIntArg(count), instanceId_));
}

double DspJsfxRuntime::gmemSeq(double page) const noexcept
{
    return gmem_.pageSeq(clampIntArg(page));
}

double DspJsfxRuntime::gmemPage(double idx) const noexcept
{
    return static_cast<double> (gmem_.pageIndexForCell(clampCellIndex(idx)));
}

} // namespace za::jsfx
