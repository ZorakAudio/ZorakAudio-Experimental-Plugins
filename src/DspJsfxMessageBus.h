#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace za::jsfx
{

class DspJsfxRuntime;

enum class DspJsfxMessageKind : std::uint8_t
{
    None = 0,
    Scalar = 1,
    Buffer = 2,
};

struct DspJsfxMessage
{
    DspJsfxMessageKind kind { DspJsfxMessageKind::None };
    std::uint64_t channelHash = 0;
    std::uint64_t sourceId = 0;
    std::uint64_t targetId = 0;
    bool direct = false;
    double tag = 0.0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    std::vector<double> buffer;
};

class DspJsfxMessageBus
{
public:
    static DspJsfxMessageBus& instance();

    void registerRuntime(DspJsfxRuntime* runtime,
                         std::uint64_t instanceId,
                         std::uint64_t domainHash,
                         const std::string& uid);
    void unregisterRuntime(std::uint64_t instanceId);

    void updateDomain(std::uint64_t instanceId, std::uint64_t domainHash);
    void updateNameHandle(std::uint64_t instanceId, std::int64_t nameHandle);
    void updateSubscription(std::uint64_t instanceId, std::uint64_t channelHash, bool subscribed);
    void updateAdvertisement(std::uint64_t instanceId, std::uint64_t channelHash, std::uint64_t caps);

    void flushOutbox(std::uint64_t instanceId,
                     std::uint64_t domainHash,
                     const std::vector<DspJsfxMessage>& outbox,
                     std::unordered_map<std::uint64_t, std::uint64_t>& droppedByChannel);

    std::size_t peerCount(std::uint64_t domainHash, std::uint64_t channelHash, int role) const;
    std::uint64_t peerId(std::uint64_t domainHash, std::uint64_t channelHash, int role, std::size_t index) const;
    bool peerInfo(std::uint64_t peerId,
                  std::int64_t* nameHandle,
                  std::string* uid,
                  std::uint64_t* caps,
                  bool* alive) const;

private:
    DspJsfxMessageBus() = default;

    struct InstanceRecord
    {
        DspJsfxRuntime* runtime = nullptr;
        std::uint64_t instanceId = 0;
        std::uint64_t domainHash = 0;
        std::int64_t nameHandle = 0;
        std::string uid;
        std::unordered_set<std::uint64_t> subscriptions;
        std::unordered_map<std::uint64_t, std::uint64_t> advertisedCaps;
    };

    static bool matchesRole(const InstanceRecord& rec, std::uint64_t channelHash, int role);

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
};

} // namespace za::jsfx
