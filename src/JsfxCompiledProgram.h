// SPDX-License-Identifier: Zlib
#pragma once
#include <cstddef>
namespace za::jsfx {
// Code loading supplies this contract. State and runtime services belong to the
// host instance; section invocation does not depend on AOT versus JIT.
template<class State> struct CompiledProgram {
    using Section=void(*)(State*);
    using Process=void(*)(State*,const float* const*,float* const*,int32_t,int32_t);
    Section init=nullptr,slider=nullptr,block=nullptr,sample=nullptr,gfx=nullptr,serialize=nullptr;
    Process process=nullptr;
    void initialise(State& s)const {if(init)init(&s);}
    template<class RebindAliases> void initialiseAndPrime(State& s,RebindAliases&& rebind)const {
        initialise(s);
        rebind(); // @init may write alias cells; the host slider remains authoritative.
        parametersChanged(s);
    }
    void parametersChanged(State& s)const {if(slider)slider(&s);}
    void beginSection(State& s)const {if(block)block(&s);}
    void sampleSection(State& s)const {if(sample)sample(&s);}
    void render(State& s)const {if(gfx)gfx(&s);}
    void saveOrRestore(State& s)const {if(serialize)serialize(&s);}
    void processAudio(State& s,const float* const* in,float* const* out,int channels,int count)const {if(process)process(&s,in,out,channels,count);}
};
}
