// SPDX-License-Identifier: Zlib
#pragma once
#include <juce_data_structures/juce_data_structures.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace za::jsfx {
// Sample-only, bounded state exchange. The audio side never takes a mutex,
// allocates, parses XML, or destroys a heap payload. Atomic cells make both
// seqlocks free of C++ data races, including a host save concurrent with a click.
class SampleBankUiState {
    static constexpr int capacity=16384,cells=capacity*2;
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    struct Slot {
        std::array<std::atomic<std::uint64_t>,cells> data{};
        std::atomic<std::uint64_t> sequence{0};
        std::atomic<int> count{0},metric{0},order{0},filterMode{0},filterMetric{0};
        std::atomic<std::uint64_t> filterCut{0};
    } published,pending;
    std::array<double,cells> scratch{};
    std::mutex stageMutex;
    int countIndex=-1,revisionIndex=-1,restoreIndex=-1,metricIndex=-1,orderIndex=-1;
    int filterModeIndex=-1,filterMetricIndex=-1,filterCutIndex=-1;
    double lastRevision=std::numeric_limits<double>::quiet_NaN();
    int lastCount=-1,lastMetric=-1,lastOrder=-1,lastFilterMode=-1,lastFilterMetric=-1;
    double lastFilterCut=std::numeric_limits<double>::quiet_NaN();
    std::uint64_t appliedSequence=0;
    static std::uint64_t bits(double v)noexcept{std::uint64_t b;std::memcpy(&b,&v,sizeof(b));return b;}
    static double number(std::uint64_t b)noexcept{double v;std::memcpy(&v,&b,sizeof(v));return v;}
public:
    template<class Lookup> explicit SampleBankUiState(Lookup lookup){
        countIndex=lookup("sample_count");revisionIndex=lookup("bank_ui_state_revision");
        restoreIndex=lookup("bank_ui_restore_requested");metricIndex=lookup("bank_ui_metric");orderIndex=lookup("bank_ui_order");
        filterModeIndex=lookup("bank_filter_mode");filterMetricIndex=lookup("bank_filter_metric");filterCutIndex=lookup("bank_filter_cut");
    }
    template<class State> void publish(const State& st)noexcept{
        if(countIndex<0||revisionIndex<0||!st.mem)return;
        const int count=std::clamp(int(st.vars[countIndex]),0,capacity);
        if(st.memN<count*2)return;
        const double revision=st.vars[revisionIndex];
        const int metric=metricIndex>=0?std::clamp(int(st.vars[metricIndex]),0,6):0,order=orderIndex>=0?std::clamp(int(st.vars[orderIndex]),0,2):0;
        const int filterMode=filterModeIndex>=0?std::clamp(int(st.vars[filterModeIndex]),0,2):0;
        const int filterMetric=filterMetricIndex>=0?std::clamp(int(st.vars[filterMetricIndex]),0,6):0;
        const double filterCut=filterCutIndex>=0?st.vars[filterCutIndex]:0.;
        if(revision==lastRevision&&count==lastCount&&metric==lastMetric&&order==lastOrder&&filterMode==lastFilterMode&&filterMetric==lastFilterMetric&&filterCut==lastFilterCut)return;
        const auto seq=published.sequence.load(std::memory_order_relaxed);
        published.sequence.store(seq+1,std::memory_order_release);
        for(int i=0;i<count*2;++i)published.data[i].store(bits(st.mem[i]),std::memory_order_relaxed);
        published.count.store(count,std::memory_order_relaxed);published.metric.store(metric,std::memory_order_relaxed);published.order.store(order,std::memory_order_relaxed);
        published.filterMode.store(filterMode,std::memory_order_relaxed);published.filterMetric.store(filterMetric,std::memory_order_relaxed);published.filterCut.store(bits(filterCut),std::memory_order_relaxed);
        published.sequence.store(seq+2,std::memory_order_release);
        lastRevision=revision;lastCount=count;lastMetric=metric;lastOrder=order;lastFilterMode=filterMode;lastFilterMetric=filterMetric;lastFilterCut=filterCut;
    }
    juce::ValueTree makeTree()const{
        std::vector<double> copy;int count=0,metric=0,order=0,filterMode=0,filterMetric=0;double filterCut=0.;
        for(;;){const auto seq=published.sequence.load(std::memory_order_acquire);if(seq&1u)continue;
            count=published.count.load(std::memory_order_relaxed);metric=published.metric.load(std::memory_order_relaxed);order=published.order.load(std::memory_order_relaxed);
            filterMode=published.filterMode.load(std::memory_order_relaxed);filterMetric=published.filterMetric.load(std::memory_order_relaxed);filterCut=number(published.filterCut.load(std::memory_order_relaxed));
            copy.resize(std::size_t(count)*2);for(int i=0;i<count*2;++i)copy[i]=number(published.data[i].load(std::memory_order_relaxed));
            if(seq==published.sequence.load(std::memory_order_acquire))break;
        }
        juce::ValueTree tree("SAMPLE_BANK_UI");tree.setProperty("schema",1,nullptr);tree.setProperty("metric",metric,nullptr);tree.setProperty("order",order,nullptr);
        tree.setProperty("filterMode",filterMode,nullptr);tree.setProperty("filterMetric",filterMetric,nullptr);tree.setProperty("filterCut",filterCut,nullptr);
        for(int i=0;i<count;++i)if(copy[i*2]>0&&copy[i*2+1]>.5){juce::ValueTree source("EXCLUDED");source.setProperty("index",i,nullptr);source.setProperty("identity",juce::String(juce::int64(copy[i*2])),nullptr);tree.addChild(source,-1,nullptr);}
        return tree;
    }
    void stage(const juce::ValueTree& tree){
        const std::lock_guard<std::mutex> lock(stageMutex);
        std::vector<double> values(cells,0.);
        int metric=0,order=0,filterMode=0,filterMetric=0;double filterCut=0.;
        if(tree.isValid()&&int(tree.getProperty("schema",0))==1){
            metric=std::clamp(int(tree.getProperty("metric",0)),0,6);order=std::clamp(int(tree.getProperty("order",0)),0,2);
            filterMode=std::clamp(int(tree.getProperty("filterMode",0)),0,2);filterMetric=std::clamp(int(tree.getProperty("filterMetric",0)),0,6);
            filterCut=double(tree.getProperty("filterCut",0.));
            if(!std::isfinite(filterCut)){filterCut=0.;filterMode=0;}
            filterCut=std::clamp(filterCut,-1000.,10000000.);
            for(int j=0;j<tree.getNumChildren();++j){const auto child=tree.getChild(j);const int index=int(child.getProperty("index",-1));const auto id=child.getProperty("identity").toString().getLargeIntValue();if(child.hasType("EXCLUDED")&&index>=0&&index<capacity&&id>0&&id<=0x000fffffffffffffll){values[index*2]=double(id);values[index*2+1]=1;}}
        }
        const auto seq=pending.sequence.load(std::memory_order_relaxed);pending.sequence.store(seq+1,std::memory_order_release);
        for(int i=0;i<cells;++i)pending.data[i].store(bits(values[i]),std::memory_order_relaxed);
        pending.filterMode.store(filterMode,std::memory_order_relaxed);pending.filterMetric.store(filterMetric,std::memory_order_relaxed);pending.filterCut.store(bits(filterCut),std::memory_order_relaxed);
        pending.metric.store(metric,std::memory_order_relaxed);pending.order.store(order,std::memory_order_relaxed);pending.sequence.store(seq+2,std::memory_order_release);
    }
    template<class State,class MemWriter,class VarWriter> bool apply(State& st,MemWriter writeMem,VarWriter writeVar)noexcept{
        const auto seq=pending.sequence.load(std::memory_order_acquire);
        if(!seq||(seq&1u)||seq==appliedSequence||restoreIndex<0||!st.mem||st.memN<cells)return false;
        for(int i=0;i<cells;++i)scratch[i]=number(pending.data[i].load(std::memory_order_relaxed));
        const int metric=pending.metric.load(std::memory_order_relaxed),order=pending.order.load(std::memory_order_relaxed);
        const int filterMode=pending.filterMode.load(std::memory_order_relaxed),filterMetric=pending.filterMetric.load(std::memory_order_relaxed);
        const double filterCut=number(pending.filterCut.load(std::memory_order_relaxed));
        if(seq!=pending.sequence.load(std::memory_order_acquire))return false;
        for(int i=0;i<cells;++i)writeMem(i,scratch[i]);
        if(metricIndex>=0)writeVar(metricIndex,metric);if(orderIndex>=0)writeVar(orderIndex,order);
        if(filterModeIndex>=0)writeVar(filterModeIndex,filterMode);if(filterMetricIndex>=0)writeVar(filterMetricIndex,filterMetric);if(filterCutIndex>=0)writeVar(filterCutIndex,filterCut);
        writeVar(restoreIndex,1);appliedSequence=seq;lastRevision=std::numeric_limits<double>::quiet_NaN();return true;
    }
};
}
