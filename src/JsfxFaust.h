#pragma once
// Build-time Faust LLVM modules. No libfaust, JIT, allocation or compiler on the
// audio thread. Include after the generated JSFXDSP.h.
#if DSPJSFX_HAS_FAUST
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace jsfx_faust {
class Engine {
    struct Bank { std::vector<std::vector<uint64_t>> memory;std::vector<void*> pointers; };
    struct Instance { std::vector<uint64_t> memory;std::array<Bank,4> banks;int currentBank=0; };
    std::array<int,4> tableRates{};
    std::array<Instance,sizeof(stages)/sizeof(stages[0])> instances;
    std::vector<double> audio,work,streams;
    std::array<double*,128> inputs{},outputs{};
    int capacity=0;
    static constexpr int audioChannels=std::max(1,DSPJSFX_PROCESS_CHANNELS);
    static constexpr int workChannels=[] {int n=1;for(const auto& stage:stages)n=std::max(n,stage.outputs);return n;}();
    static constexpr int streamChannels=[] {int n=0;for(const auto& stage:stages)for(int i=0;i<stage.signalCount;++i)n+=stage.signals[i].kind!=2;return n;}();
    struct Capture { Binding source; int stream; };
    std::array<std::array<Capture,streamChannels>,sizeof(stages)/sizeof(stages[0])> captures{};
    std::array<int,sizeof(stages)/sizeof(stages[0])> captureCounts{};
    std::array<std::array<double*,streamChannels>,sizeof(stages)/sizeof(stages[0])> captureTargets{};
    void beginSample(int index,DSPJSFX_State& s) {
        if constexpr(streamChannels>0) {
            int stream=0;captureCounts[index]=0;captureTargets[index].fill(nullptr);
            for(const auto& consumer:stages) {
                const bool active=consumer.captureStage==index && (consumer.condition.kind<0 || read(s,consumer.condition)!=0);
                for(int k=0;k<consumer.signalCount;++k)if(consumer.signals[k].kind!=2) {
                    if(active){captures[index][captureCounts[index]++]={consumer.signals[k],stream};captureTargets[index][stream]=streams.data()+size_t(stream)*capacity;}
                    ++stream;
                }
            }
        }
    }
    static int streamOffset(int index) {int n=0;for(int j=0;j<index;++j)for(int k=0;k<stages[j].signalCount;++k)n+=stages[j].signals[k].kind!=2;return n;}
    static double read(DSPJSFX_State& s,Binding b) {
        switch(b.kind) {case 0:return double(s.vars[b.index]);case 1:return double(s.sliders[b.index]);case 2:return double(s.spl[b.index]);case 3:return double(s.srate);case 4:return double(s.samplesblock);default:return 0.;}
    }
    static double filtered(double v) {
#if !DSPJSFX_EEL2_STORES
        return v;
#else
        // Same observable assignment rule as the JSFX frontend.
        uint64_t bits;std::memcpy(&bits,&v,8);const auto exponent=(bits>>52)&2047;
        return exponent==0 || exponent==2047 ? 0.:v;
#endif
    }
    static void write(DSPJSFX_State& s,Binding b,double v) {
        v=filtered(v);if(b.kind==0)s.vars[b.index]=v;else if(b.kind==2)s.spl[b.index]=v;
        else if(b.kind==1) {
            s.sliders[b.index]=v;if(b.alias>=0)s.vars[b.alias]=v;
            s.pendingSliderChangeMask[b.index/64]|=uint64_t(1)<<(b.index%64);
        }
    }
    void control(const Stage& stage,void* memory,DSPJSFX_State& s) {
        for(int z=0;z<stage.zoneCount;++z)if(stage.zones[z].source.kind>=0) {
            const double v=read(s,stage.zones[z].source);
            std::memcpy(static_cast<char*>(memory)+stage.zones[z].offset,&v,8);
        }
    }
    void faust(int index,DSPJSFX_State& state,int first,int count,int channels) {
        const auto& stage=stages[index];if(stage.condition.kind>=0 && read(state,stage.condition)==0)return;void* memory=instances[index].memory.data();
        if(stage.tableCount)stage.bindTables(instances[index].banks[instances[index].currentBank].pointers.data());
        control(stage,memory,state);
        int stream=streamOffset(index);
        for(int c=0;c<stage.inputs;++c) {
            if(c<stage.signalCount && stage.signals[c].kind!=2)inputs[c]=streams.data()+size_t(stream++)*capacity+first;
            else {const int channel=c<stage.signalCount ? stage.signals[c].index : c-stage.signalCount;
                inputs[c]=audio.data()+size_t(channel)*capacity+first;}
        }
        for(int c=0;c<stage.outputs;++c)outputs[c]=work.data()+size_t(c)*capacity;
        stage.compute(memory,count,inputs.data(),outputs.data());
        count==1 ? ++scalarCalls : ++blockCalls;processedFrames+=uint64_t(count);
        for(int c=0;c<stage.audioOutputs;++c) {
            std::copy_n(outputs[c],count,audio.data()+size_t(c)*capacity+first);
            state.spl[c]=outputs[c][count-1];
        }
        for(int c=0;c<stage.exportCount;++c)write(state,stage.exports[c],outputs[stage.audioOutputs+c][count-1]);
    }
    void sample(const Stage& stage,DSPJSFX_State& s,int frame,int channels) {
        for(int c=0;c<channels;++c)s.spl[c]=audio[size_t(c)*capacity+frame];
        stage.eel(&s);
        if constexpr(streamChannels>0) {
            const int index=int(&stage-stages);
            for(int k=0;k<captureCounts[index];++k) {
                const auto& capture=captures[index][k];
                streams[size_t(capture.stream)*capacity+frame]=read(s,capture.source);
            }
        }
        for(int c=0;c<channels;++c)audio[size_t(c)*capacity+frame]=double(s.spl[c]);
    }
public:
    uint64_t blockCalls=0,scalarCalls=0,processedFrames=0;
    Engine()=default;Engine(const Engine&)=delete;Engine& operator=(const Engine&)=delete;
    ~Engine(){for(size_t i=0;i<instances.size();++i)if(stages[i].kind==2 && !instances[i].memory.empty()) {
        if(stages[i].tableCount)stages[i].bindTables(instances[i].banks[instances[i].currentBank].pointers.data());
        stages[i].destroy(instances[i].memory.data());
    }}
    void prepare(double rate,int expected,double hostRate=0) {
        if(!std::isfinite(rate) || rate<1 || rate>2147483647 || expected<0 || expected>1048576)throw std::invalid_argument("Invalid Faust preparation dimensions");
        // Reserve a bounded margin for hosts whose actual blocks vary. A host
        // exceeding this bound must prepare again; process() never reallocates.
        capacity=std::max(1024,expected);audio.assign(size_t(audioChannels)*capacity,0.);work.assign(size_t(workChannels)*capacity,0.);streams.assign(size_t(streamChannels)*capacity,0.);
        const double base=hostRate>0?hostRate:rate;
        if(!std::isfinite(base) || base<1 || base*8>2147483647)throw std::invalid_argument("Invalid Faust table preparation rates");
        for(int bank=0;bank<4;++bank)tableRates[bank]=int(base*(1<<bank));
        for(size_t i=0;i<instances.size();++i) {
            const auto& stage=stages[i];if(stage.kind!=2)continue;
            auto& instance=instances[i];
            if(instance.memory.empty()){instance.memory.resize((size_t(stage.size)+7)/8);stage.allocate(instance.memory.data());}
            if(stage.tableCount)for(int bank=0;bank<4;++bank) {
                auto& tables=instance.banks[bank];tables.memory.resize(stage.tableCount);tables.pointers.resize(stage.tableCount);
                for(int t=0;t<stage.tableCount;++t){tables.memory[t].resize((size_t(stage.tableSizes[t])+7)/8);tables.pointers[t]=tables.memory[t].data();}
                stage.bindTables(tables.pointers.data());stage.seedTables(tables.pointers.data());stage.classInit(tableRates[bank]);
            } else stage.classInit(int(rate));
        }
        reset(rate);
    }
    void reset(double rate) {
        for(size_t i=0;i<instances.size();++i) {
            const auto& stage=stages[i];if(stage.kind!=2)continue;
            if(instances[i].memory.empty())throw std::logic_error("Prepare Faust before resetting it");
            if(stage.tableCount) {
                int selected=-1;for(int bank=0;bank<4;++bank)if(tableRates[bank]==int(rate))selected=bank;
                if(selected<0)throw std::invalid_argument("Faust table rate was not prepared");
                instances[i].currentBank=selected;stage.bindTables(instances[i].banks[selected].pointers.data());
            }
            void* memory=instances[i].memory.data();stage.constants(memory,int(rate));stage.clear(memory);
            for(int z=0;z<stage.zoneCount;++z)std::memcpy(static_cast<char*>(memory)+stage.zones[z].offset,&stage.zones[z].initial,8);
        }
        blockCalls=scalarCalls=processedFrames=0;
    }
    void process(DSPJSFX_State& s,const float* const* in,float* const* out,int channels,int count) noexcept {
        channels=std::clamp(channels,0,64);
        if(count<0)return;
        s.samplesblock=count;s.currentBlockSize=count;s.currentSampleRate=double(s.srate);
        if(count==0){for(const auto& stage:stages)if(stage.kind==0) {
            stage.eel(&s);uint64_t changed=0;
            for(int word=0;word<DSPJSFX_SLIDER_MASK_WORDS;++word)changed|=s.pendingSliderChangeMask[word]|s.pendingSliderAutomateMask[word]|s.pendingSliderAutomateEndMask[word];
            if(changed)jsfx_slider(&s);
        }return;}
        if(count>capacity || !capacity) {s.memoryFault=1;for(int c=0;c<channels;++c)std::fill_n(out[c],count,0.f);return;}
        for(int c=audioChannels;c<channels;++c)if(in[c]!=out[c])std::memmove(out[c],in[c],size_t(count)*sizeof(float));
        channels=std::min(channels,audioChannels);
        for(int c=0;c<audioChannels;++c) {
            double* target=audio.data()+size_t(c)*capacity;
            if(c<channels)for(int f=0;f<count;++f)target[f]=in[c][f];else std::fill_n(target,count,0.);
        }
        // The optional quantum keeps reset feedback bounded without repeating
        // the host @block (MIDI ingestion, clocks and pool adoption run once).
        const int chunkSize=quantum ? quantum : count;
        const size_t firstStage=quantum ? 1 : 0;
        if(quantum) {
            stages[0].eel(&s);
            uint64_t changed=0;for(int word=0;word<DSPJSFX_SLIDER_MASK_WORDS;++word)changed|=s.pendingSliderChangeMask[word]|s.pendingSliderAutomateMask[word]|s.pendingSliderAutomateEndMask[word];
            if(changed)jsfx_slider(&s);
        }
        for(int first=0;first<count;) {
        int frames=std::min(chunkSize,count-first);
        if(quantum && count-first==chunkSize+1)--frames; // Avoid a one-frame remainder.
        for(size_t i=firstStage;i<instances.size();) {
            const auto& stage=stages[i];
            if(stage.kind==0) {
                stage.eel(&s);
                uint64_t changed=0;for(int word=0;word<DSPJSFX_SLIDER_MASK_WORDS;++word)changed|=s.pendingSliderChangeMask[word]|s.pendingSliderAutomateMask[word]|s.pendingSliderAutomateEndMask[word];
                if(changed)jsfx_slider(&s);
                ++i;continue;
            }
            if(stage.fused) {
                size_t end=i+1;while(end<instances.size() && stages[end].island==stage.island)++end;
                for(size_t j=i;j<end;++j)if(stages[j].kind==1)beginSample(int(j),s);
                for(int f=first;f<first+frames;++f)for(size_t j=i;j<end;++j) {
                    if(stages[j].kind==1)sample(stages[j],s,f,channels);else faust(int(j),s,f,1,channels);
                }
                i=end;
            } else {
                if(stage.kind==1){beginSample(int(i),s);if(stage.bulk)stage.bulk(&s,audio.data(),capacity,first,frames,channels,captureTargets[i].data());else for(int f=first;f<first+frames;++f)sample(stage,s,f,channels);}else faust(int(i),s,first,frames,channels);
                ++i;
            }
        }
        first+=frames;
        }
        for(int c=0;c<channels;++c)for(int f=0;f<count;++f)out[c][f]=float(audio[size_t(c)*capacity+f]);
    }
};
}
extern "C" void jsfx_faust_process(DSPJSFX_State*,const float* const*,float* const*,int32_t,int32_t);
#ifdef JSFX_FAUST_IMPLEMENTATION
extern "C" void jsfx_faust_process(DSPJSFX_State* s,const float* const* in,float* const* out,int32_t channels,int32_t count) {
    auto* engine=static_cast<jsfx_faust::Engine*>(s->faustContext);
    if(engine)engine->process(*s,in,out,channels,count);
    else {s->memoryFault=1;for(int c=0;c<std::clamp(int(channels),0,64);++c)std::fill_n(out[c],std::max(0,int(count)),0.f);}
}
#endif
#endif
