// SPDX-License-Identifier: Zlib
#pragma once
#include <functional>
#include <mutex>
namespace za::jsfx {
struct TaskHeapPort {
    DSPJSFX_State* state=nullptr;
    int64_t capacity=0;
    std::function<void(DSPJSFX_State*)> rebind;
    std::mutex* lifecycleMutex=nullptr;
};
// The host binding only resolves its instance. Generation pinning and heap
// adoption semantics are shared for statically and dynamically loaded code.
template<auto ResolveOwner> struct TaskRuntimeHooks {
    static void bind(jsfx_tasks::Runtime& runtime) {
        runtime.setArenaHooks(
            [](DSPJSFX_State* state,double handle,double generation)->std::shared_ptr<void> {
                auto* owner=ResolveOwner(state);auto* pool=owner?owner->getSamplePoolForHandle(handle):nullptr;
                if(!pool)return {};
                auto pin=pool->pinRead();return pin->generation()==static_cast<uint64_t>(generation) && pin->isCurrent()?pin:nullptr;
            },
            [](const std::shared_ptr<void>& opaque,jsfx_tasks::Callback fn,DSPJSFX_State* state,const double* captures)->double {
                auto pin=std::static_pointer_cast<DspJsfxSamplePool::PinnedRead>(opaque);
                DspJsfxSamplePool::ReadBatch batch;
                if(!pin || !pin->bind(batch))throw std::runtime_error("Unable to bind analysis sample generation");
                return fn(state,0,captures);
            },
            [](const std::shared_ptr<void>& opaque)->bool {return opaque && static_cast<DspJsfxSamplePool::PinnedRead*>(opaque.get())->isCurrent();});
#if DSPJSFX_NATIVE_GFX_LEGACY
        runtime.setHeapSwapHooks(
            [](DSPJSFX_State* state)->bool {
                auto* owner=ResolveOwner(state);return owner && state==owner->taskHeap.state && state->mem && state->memN==owner->taskHeap.capacity;
            },
            [](DSPJSFX_State* state) {auto* owner=ResolveOwner(state);if(owner && owner->taskHeap.rebind)owner->taskHeap.rebind(state);},
            [](DSPJSFX_State* state)->bool {auto* owner=ResolveOwner(state);return owner && (!owner->taskHeap.lifecycleMutex || owner->taskHeap.lifecycleMutex->try_lock());},
            [](DSPJSFX_State* state) {auto* owner=ResolveOwner(state);if(owner && owner->taskHeap.lifecycleMutex)owner->taskHeap.lifecycleMutex->unlock();});
#endif
    }
};
}
