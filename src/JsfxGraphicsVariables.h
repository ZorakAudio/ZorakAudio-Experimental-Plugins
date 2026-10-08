// SPDX-License-Identifier: Zlib
#pragma once
#include "JsfxSharedCells.h"
#include <memory>
#include <array>
#include <cstdint>
#include <vector>

namespace za::jsfx {
// The graphics adapter executes a frozen scalar view. Scratch locals remain
// private; changed writable globals return through latest-value mailboxes at
// callback boundaries, following the production GFX publication contract.
// Allocation happens while preparing the program, never in the audio callback.
class GraphicsVariables {
    struct Cell {std::atomic<double> value{0};std::atomic<uint64_t> version{0};uint64_t applied=0;};
    std::unique_ptr<Cell[]> incoming;
    std::vector<uint8_t> directions;
    std::vector<double> before;
    struct Snapshot {
        std::vector<double> vars;
        std::vector<uint64_t> acknowledgments;
        std::array<double,DSPJSFX_MAX_SLIDERS> sliders{};
        std::array<uint64_t,DSPJSFX_MAX_SLIDERS> sliderAcknowledgments{};
        std::array<uint64_t,DSPJSFX_SLIDER_MASK_WORDS> visibility{};
        std::array<uint64_t,DSPJSFX_MAX_SLIDERS> visibilityAcknowledgments{};
        double rate=0,block=0;
    };
    std::array<Snapshot,3> snapshots;
    std::array<std::atomic<int>,3> readers{};
    std::atomic<int> latest{-1};
    std::array<Cell,DSPJSFX_MAX_SLIDERS> sliderIncoming;
    std::array<Cell,DSPJSFX_MAX_SLIDERS> visibilityIncoming;
    std::array<uint64_t,DSPJSFX_SLIDER_MASK_WORDS> visibilityBefore{};
    std::atomic<bool> queued{false};
    int extent=0;
public:
    void configure(std::vector<uint8_t> flags){
        directions=std::move(flags);extent=(int)directions.size();
        incoming=std::make_unique<Cell[]>((size_t)extent);before.resize((size_t)extent);
        for(auto& snapshot:snapshots){snapshot.vars.resize((size_t)extent);snapshot.acknowledgments.resize((size_t)extent);}
    }
    void seed(const DSPJSFX_State& dsp,DSPJSFX_State& graphics)noexcept{
        for(int i=0;i<extent;++i){graphics.vars[i]=double(dsp.vars[i]);incoming[(size_t)i].version.store(0);incoming[(size_t)i].applied=0;}
        for(auto& slider:sliderIncoming){slider.version.store(0);slider.applied=0;}
        for(auto& cell:visibilityIncoming){cell.version.store(0);cell.applied=0;}queued.store(false);
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i)graphics.sliders[i]=double(dsp.sliders[i]);
        for(int i=0;i<DSPJSFX_SLIDER_MASK_WORDS;++i)graphics.sliderVisibleMask[i]=std::atomic_ref<uint64_t>(*const_cast<uint64_t*>(&dsp.sliderVisibleMask[i])).load();
        graphics.sliderVisibilityInit=dsp.sliderVisibilityInit;graphics.randIndex=dsp.randIndex;for(int i=0;i<624;++i)graphics.randMT[i]=dsp.randMT[i];
        publish(dsp);
    }
    void publish(const DSPJSFX_State& dsp)noexcept{
        // A slow graphics reader may lose an intermediate publication, but
        // it can never stall audio or read a slot while audio overwrites it.
        const int current=latest.load(std::memory_order_seq_cst);
        for(int slot=0;slot<3;++slot)if(slot!=current && readers[(size_t)slot].load(std::memory_order_seq_cst)==0){
            auto& s=snapshots[(size_t)slot];for(int i=0;i<extent;++i)if(directions[(size_t)i]&1){s.vars[(size_t)i]=double(dsp.vars[i]);s.acknowledgments[(size_t)i]=incoming[(size_t)i].applied;}
            for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i){s.sliders[(size_t)i]=double(dsp.sliders[i]);s.sliderAcknowledgments[(size_t)i]=sliderIncoming[(size_t)i].applied;s.visibilityAcknowledgments[(size_t)i]=visibilityIncoming[(size_t)i].applied;}
            for(int i=0;i<DSPJSFX_SLIDER_MASK_WORDS;++i)s.visibility[(size_t)i]=std::atomic_ref<uint64_t>(*const_cast<uint64_t*>(&dsp.sliderVisibleMask[i])).load();
            s.rate=double(dsp.srate);s.block=double(dsp.samplesblock);latest.store(slot,std::memory_order_seq_cst);return;
        }
    }
    void begin(const DSPJSFX_State& dsp,DSPJSFX_State& graphics)noexcept{
        (void)dsp;
        int slot;
        for(;;){slot=latest.load(std::memory_order_acquire);if(slot<0)return;readers[(size_t)slot].fetch_add(1,std::memory_order_seq_cst);if(slot==latest.load(std::memory_order_seq_cst))break;readers[(size_t)slot].fetch_sub(1,std::memory_order_seq_cst);}
        const auto& s=snapshots[(size_t)slot];
        for(int i=0;i<extent;++i){if(directions[(size_t)i]&1)graphics.vars[i]=incoming[(size_t)i].version.load(std::memory_order_acquire)>s.acknowledgments[(size_t)i]?incoming[(size_t)i].value.load():s.vars[(size_t)i];before[(size_t)i]=double(graphics.vars[i]);}
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i)graphics.sliders[i]=sliderIncoming[(size_t)i].version.load(std::memory_order_acquire)>s.sliderAcknowledgments[(size_t)i]?sliderIncoming[(size_t)i].value.load():s.sliders[(size_t)i];
        auto visible=s.visibility;
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i)if(visibilityIncoming[(size_t)i].version.load(std::memory_order_acquire)>s.visibilityAcknowledgments[(size_t)i]){
            const auto bit=UINT64_C(1)<<(i%64);auto& word=visible[(size_t)i/64];
            if(visibilityIncoming[(size_t)i].value.load()!=0)word|=bit;else word&=~bit;
        }
        for(int i=0;i<DSPJSFX_SLIDER_MASK_WORDS;++i)graphics.sliderVisibleMask[i]=visible[(size_t)i];visibilityBefore=visible;
        graphics.srate=s.rate;graphics.samplesblock=s.block;readers[(size_t)slot].fetch_sub(1,std::memory_order_seq_cst);
    }
    void finish(const DSPJSFX_State& graphics)noexcept{
        bool changed=false;
        for(int i=0;i<extent;++i)if(directions[(size_t)i]&2){
            const auto value=double(graphics.vars[i]);
            if(value!=before[(size_t)i]){incoming[(size_t)i].value.store(value,std::memory_order_relaxed);incoming[(size_t)i].version.fetch_add(1,std::memory_order_release);changed=true;}
        }
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i){const auto bit=UINT64_C(1)<<(i%64);if((graphics.sliderVisibleMask[i/64]^visibilityBefore[(size_t)i/64])&bit){auto& cell=visibilityIncoming[(size_t)i];cell.value.store((graphics.sliderVisibleMask[i/64]&bit)!=0);cell.version.fetch_add(1,std::memory_order_release);changed=true;}}
        if(changed)queued.store(true,std::memory_order_release);
    }
    void queueSlider(int slot,double value)noexcept{sliderIncoming[(size_t)slot].value.store(value);sliderIncoming[(size_t)slot].version.fetch_add(1,std::memory_order_release);queued.store(true,std::memory_order_release);}
    template<class ApplySlider> void apply(DSPJSFX_State& dsp,ApplySlider&& slider)noexcept{
        if(!queued.exchange(false,std::memory_order_acq_rel))return;
        for(int i=0;i<extent;++i){auto& cell=incoming[(size_t)i];const auto version=cell.version.load(std::memory_order_acquire);if(version!=cell.applied){dsp.vars[i]=cell.value.load(std::memory_order_relaxed);cell.applied=version;}}
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i){auto& cell=sliderIncoming[(size_t)i];const auto version=cell.version.load(std::memory_order_acquire);if(version!=cell.applied){slider(i,cell.value.load());cell.applied=version;}}
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i){auto& cell=visibilityIncoming[(size_t)i];const auto version=cell.version.load(std::memory_order_acquire);if(version!=cell.applied){const auto bit=UINT64_C(1)<<(i%64);auto word=std::atomic_ref<uint64_t>(dsp.sliderVisibleMask[i/64]);if(cell.value.load()!=0)word.fetch_or(bit);else word.fetch_and(~bit);cell.applied=version;}}
    }
};
}
