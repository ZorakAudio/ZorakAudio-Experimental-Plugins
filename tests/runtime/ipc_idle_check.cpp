// Include the production bus to inspect its shared-memory lock and slots.
// Payload/registration implementations are not recreated by this test.
#include "../../src/DspJsfxMessageBus.cpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

int main(int argc,char** argv){
    using namespace za::jsfx;
    auto& bus=DspJsfxMessageBus::instance();
    if(argc==3){
        const auto childDomain=std::strtoull(argv[2],nullptr,16);
        bus.registerRuntime(nullptr,104,childDomain,"child");
        DspJsfxMessage message;message.kind=DspJsfxMessageKind::Scalar;message.channelHash=77;
        message.direct=true;message.targetId=103;message.a=4.5;
        std::unordered_map<uint64_t,uint64_t> drops;
        bus.flushOutbox(104,childDomain,{message},drops);bus.unregisterRuntime(104);
        return drops[77]?1:0;
    }
    const auto domain=uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    constexpr uint64_t sender=101,receiver=102,directReceiver=103,channel=77;
    bus.registerRuntime(nullptr,sender,domain,"sender");
    bus.registerRuntime(nullptr,receiver,domain,"receiver");
    bus.registerRuntime(nullptr,directReceiver,domain,"direct");
    bus.updateSubscription(receiver,channel,true);
    DspJsfxSharedMemorySegment mapping;bool created=true;
    assert(mapping.openOrCreate("msg_v4_"+hex64(domain),sizeof(IpcHeader),&created) && !created);
    auto* header=static_cast<IpcHeader*>(mapping.data());mapping.finishInitialization();
    uint64_t cursor=0,directCursor=0;
    std::unordered_set<uint64_t> subscriptions{channel},none;
    std::unordered_map<uint64_t,std::deque<DspJsfxMessage>> inbox,directInbox;
    std::unordered_map<uint64_t,uint64_t> drops;
    auto collect=[&]{bus.collectInbox(receiver,domain,subscriptions,cursor,inbox,drops);};
    // An empty ring does not acquire the IPC payload lock.
    header->lock.store(1);
    assert(!bus.hasPendingFor(receiver,domain,subscriptions,cursor));collect();assert(cursor==0);
    // Failed metadata publication must be retried, even on an empty ring.
    bus.updateSubscription(receiver,78,true);subscriptions.insert(78);
    assert(bus.peerCount(domain,78,1)==0);
    header->lock.store(0);collect();assert(bus.peerCount(domain,78,1)==1);
    DspJsfxMessage message;message.kind=DspJsfxMessageKind::Scalar;message.channelHash=channel;message.a=3.5;
    bus.flushOutbox(sender,domain,{message},drops);
    assert(bus.hasPendingFor(receiver,domain,subscriptions,cursor));
    // A contended ring retains its cursor and conservatively wakes the host.
    header->lock.store(1);assert(bus.hasPendingFor(receiver,domain,subscriptions,cursor));collect();assert(cursor==0);
    header->lock.store(0);collect();assert(inbox[channel].size()==1 && inbox[channel].front().a==3.5);inbox.clear();
    message.direct=true;message.targetId=directReceiver;
    bus.flushOutbox(sender,domain,{message},drops);
    assert(bus.hasPendingFor(directReceiver,domain,none,directCursor));
    bus.collectInbox(directReceiver,domain,none,directCursor,directInbox,drops);
    assert(directInbox[channel].size()==1); // Direct delivery needs no subscription.
    directInbox.clear();
    // The same wake/delivery contract holds across separate processes.
    const auto domainText=hex64(domain);
#if defined(_WIN32)
    assert(_spawnl(_P_WAIT,argv[0],argv[0],"--direct-child",domainText.c_str(),nullptr)==0);
#else
    const auto child=fork();assert(child>=0);
    if(child==0){execl(argv[0],argv[0],"--direct-child",domainText.c_str(),nullptr);_exit(127);}
    int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
#endif
    assert(bus.hasPendingFor(directReceiver,domain,none,directCursor));
    bus.collectInbox(directReceiver,domain,none,directCursor,directInbox,drops);
    assert(directInbox[channel].size()==1 && directInbox[channel].front().a==4.5);
    collect();assert(inbox.empty());
    // Sustained traffic past the stale window preserves a processing peer.
    message.direct=false;message.targetId=0;
    for(int batch=0;batch<140;++batch){
        bus.flushOutbox(sender,domain,std::vector<DspJsfxMessage>(512,message),drops);collect();
        assert(inbox[channel].size()==512);inbox.clear();assert(bus.peerCount(domain,channel,1)==1);
    }
    // Recover a registration reclaimed while the receiver was not processing.
    for(auto& slot:header->instances)if(slot.instanceId.load()==receiver)clearInstanceSlot(slot);
    message.direct=true;message.targetId=sender;bus.flushOutbox(sender,domain,{message},drops);
    collect();assert(bus.peerCount(domain,channel,1)==1);
    // Concurrent publication and consumption retain exact delivery below the
    // bounded ring capacity; no unlocked inspection of message payloads.
    message.direct=false;message.targetId=0;
    std::atomic<bool> finished{false};
    std::atomic<uint64_t> contendedDrops{0};
    std::thread publisher([&]{std::unordered_map<uint64_t,uint64_t> localDrops;for(int n=0;n<1000;++n)bus.flushOutbox(sender,domain,{message},localDrops);contendedDrops=localDrops[channel];finished=true;});
    size_t received=0;
    do{collect();received+=inbox[channel].size();inbox.clear();std::this_thread::yield();}while(!finished.load());
    publisher.join();collect();received+=inbox[channel].size();assert(received+contendedDrops==1000);
    bus.updateSubscription(receiver,channel,false);assert(bus.peerCount(domain,channel,1)==0);
    bus.updateDomain(receiver,domain+1);assert(bus.peerCount(domain,78,1)==0);
    bus.unregisterRuntime(sender);bus.unregisterRuntime(receiver);bus.unregisterRuntime(directReceiver);
#if !defined(_WIN32)
    shm_unlink(makeSharedMemoryObjectName("msg_v4_"+hex64(domain)).c_str());
    shm_unlink(makeSharedMemoryObjectName("msg_v4_"+hex64(domain+1)).c_str());
#endif
    std::cout<<"IPC idle, retries, direct wakeups, liveness and concurrency passed\n";
}
