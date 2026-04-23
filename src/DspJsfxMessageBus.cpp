#include "DspJsfxMessageBus.h"
#include "DspJsfxRuntime.h"

#include <algorithm>
#include <mutex>

namespace za::jsfx
{

DspJsfxMessageBus& DspJsfxMessageBus::instance()
{
    static DspJsfxMessageBus bus;
    return bus;
}

void DspJsfxMessageBus::registerRuntime(DspJsfxRuntime* runtime,
                                        std::uint64_t instanceId,
                                        std::uint64_t domainHash,
                                        const std::string& uid)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& rec = instances_[instanceId];
    rec.runtime = runtime;
    rec.instanceId = instanceId;
    rec.domainHash = domainHash;
    rec.uid = uid;
}

void DspJsfxMessageBus::unregisterRuntime(std::uint64_t instanceId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    instances_.erase(instanceId);
}

void DspJsfxMessageBus::updateDomain(std::uint64_t instanceId, std::uint64_t domainHash)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(instanceId);
    if (it != instances_.end())
        it->second.domainHash = domainHash;
}

void DspJsfxMessageBus::updateNameHandle(std::uint64_t instanceId, std::int64_t nameHandle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(instanceId);
    if (it != instances_.end())
        it->second.nameHandle = nameHandle;
}

void DspJsfxMessageBus::updateSubscription(std::uint64_t instanceId, std::uint64_t channelHash, bool subscribed)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(instanceId);
    if (it == instances_.end())
        return;
    if (subscribed)
        it->second.subscriptions.insert(channelHash);
    else
        it->second.subscriptions.erase(channelHash);
}

void DspJsfxMessageBus::updateAdvertisement(std::uint64_t instanceId, std::uint64_t channelHash, std::uint64_t caps)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(instanceId);
    if (it == instances_.end())
        return;
    if (caps == 0)
        it->second.advertisedCaps.erase(channelHash);
    else
        it->second.advertisedCaps[channelHash] = caps;
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

void DspJsfxMessageBus::flushOutbox(std::uint64_t instanceId,
                                    std::uint64_t domainHash,
                                    const std::vector<DspJsfxMessage>& outbox,
                                    std::unordered_map<std::uint64_t, std::uint64_t>& droppedByChannel)
{
    std::vector<std::pair<DspJsfxRuntime*, DspJsfxMessage>> deliveries;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& msg : outbox)
        {
            if (msg.direct)
            {
                auto it = instances_.find(msg.targetId);
                if (it == instances_.end() || it->second.runtime == nullptr || it->second.domainHash != domainHash)
                {
                    droppedByChannel[msg.channelHash] += 1u;
                    continue;
                }
                deliveries.emplace_back(it->second.runtime, msg);
                continue;
            }

            bool delivered = false;
            for (const auto& kv : instances_)
            {
                const auto& rec = kv.second;
                if (rec.runtime == nullptr || rec.domainHash != domainHash || rec.instanceId == instanceId)
                    continue;
                if (rec.subscriptions.find(msg.channelHash) == rec.subscriptions.end())
                    continue;
                deliveries.emplace_back(rec.runtime, msg);
                delivered = true;
            }

            if (! delivered)
                droppedByChannel[msg.channelHash] += 1u;
        }
    }

    for (auto& delivery : deliveries)
    {
        if (! delivery.first->enqueuePendingMessage(delivery.second))
            droppedByChannel[delivery.second.channelHash] += 1u;
    }
}

std::size_t DspJsfxMessageBus::peerCount(std::uint64_t domainHash, std::uint64_t channelHash, int role) const
{
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
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = instances_.find(peerId);
    if (it == instances_.end())
        return false;

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

} // namespace za::jsfx
