#pragma once

#include "DspJsfxGmem.h"
#include "DspJsfxMessageBus.h"
#include "JSFXDSP.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace za::jsfx
{

class DspJsfxRuntime
{
public:
    DspJsfxRuntime();
    ~DspJsfxRuntime();

    DspJsfxRuntime(const DspJsfxRuntime&) = delete;
    DspJsfxRuntime& operator=(const DspJsfxRuntime&) = delete;

    void attachToState(DSPJSFX_State* st);
    void detachFromState();
    static DspJsfxRuntime* findForState(DSPJSFX_State* st) noexcept;

    void reset();
    void beginBlock(DSPJSFX_State& st, int numSamples);
    void endBlock(DSPJSFX_State& st);

    bool hasReadyMessages() const noexcept;
    bool hasPendingForThisInstance() const noexcept;
    bool enqueuePendingMessage(const DspJsfxMessage& message);

    std::uint64_t instanceId() const noexcept { return instanceId_; }
    const std::string& instanceUid() const noexcept { return uid_; }
    std::int64_t instanceNameHandle() const noexcept { return nameHandle_; }

    bool joinDomain(std::uint64_t domainHash);
    std::uint64_t domainHash() const noexcept { return domainHash_; }

    bool setNameHandle(std::int64_t handle);

    bool subscribe(std::uint64_t channelHash);
    bool unsubscribe(std::uint64_t channelHash);
    bool advertise(std::uint64_t channelHash, std::uint64_t caps);

    bool queueScalar(std::uint64_t channelHash, double tag, double a, double b, double c, double d);
    bool queueScalarTo(std::uint64_t targetId, std::uint64_t channelHash, double tag, double a, double b, double c, double d);
    bool queueBuffer(std::uint64_t channelHash, double tag, const double* src, int len);
    bool queueBufferTo(std::uint64_t targetId, std::uint64_t channelHash, double tag, const double* src, int len);

    int avail(std::uint64_t channelHash) const;
    int kind(std::uint64_t channelHash) const;
    int recvScalar(std::uint64_t channelHash, double* src, double* tag, double* a, double* b, double* c, double* d);
    int recvBuffer(std::uint64_t channelHash, double* src, double* tag, double* dst, int maxLen, DSPJSFX_State& st);
    int clear(std::uint64_t channelHash);
    double lastMessageLength() const noexcept { return static_cast<double> (lastMessageLength_); }
    double dropped(std::uint64_t channelHash) const;

    double peerCount(std::uint64_t channelHash, int role) const;
    double peerId(std::uint64_t channelHash, int role, int index) const;
    bool peerNameHandle(std::uint64_t peerId, std::int64_t* outHandle) const;
    bool peerUid(std::uint64_t peerId, std::string* outUid) const;
    double peerCaps(std::uint64_t peerId) const;
    double peerAlive(std::uint64_t peerId) const;

    bool gmemAttach(std::uint64_t nameHash, std::uint64_t requestedCells);
    double gmemSize() const noexcept;
    double gmemLoad(double idx) const noexcept;
    double gmemStore(double idx, double value) noexcept;
    double gmemGet(DSPJSFX_State& st, double dstBase, double srcIdx, double count) noexcept;
    double gmemPut(DSPJSFX_State& st, double dstIdx, double srcBase, double count) noexcept;
    double gmemFill(double dstIdx, double value, double count) noexcept;
    double gmemZero(double dstIdx, double count) noexcept;
    double gmemCopy(double dstIdx, double srcIdx, double count) noexcept;
    double gmemSeq(double page) const noexcept;
    double gmemPage(double idx) const noexcept;

private:
    bool enqueueMessage(DspJsfxMessage&& message);
    std::deque<DspJsfxMessage>* inboxFor(std::uint64_t channelHash);
    const std::deque<DspJsfxMessage>* inboxFor(std::uint64_t channelHash) const;

    DSPJSFX_State* state_ = nullptr;
    std::uint64_t instanceId_ = 0;
    std::string uid_;
    std::int64_t nameHandle_ = 0;
    std::uint64_t domainHash_ = 0x9ae16a3b2f90404full;

    std::unordered_set<std::uint64_t> subscriptions_;
    std::unordered_map<std::uint64_t, std::uint64_t> advertisedCaps_;

    mutable std::mutex inboxMutex_;
    std::unordered_map<std::uint64_t, std::deque<DspJsfxMessage>> readyInbox_;
    std::unordered_map<std::uint64_t, std::deque<DspJsfxMessage>> pendingInbox_;
    std::vector<DspJsfxMessage> outbox_;
    std::unordered_map<std::uint64_t, std::uint64_t> droppedByChannel_;
    int lastMessageLength_ = 0;

    DspJsfxGmemAttachment gmem_;
};

} // namespace za::jsfx
