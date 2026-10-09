// SPDX-License-Identifier: Zlib
#pragma once
#include "DspJsfxHostTransport.h"
#include <array>
#include <cstddef>
#include <type_traits>

namespace za::jsfx {
enum class HostVariable : size_t {
    Tempo, Channels, TransportBound, TransportValid, TransportDiscontinuity,
    PlayState, PlayPosition, BeatPosition, GfxBound, GfxActive, Count
};
inline constexpr std::array<const char*,size_t(HostVariable::Count)> hostVariableNames{
    "tempo","num_ch","host_transport_bound","host_transport_valid",
    "host_transport_discontinuity","play_state","play_position","beat_position",
    "host_gfx_bound","host_gfx_active"
};
// Resolve once per compiled program, before processing. A missing variable
// remains -1; adapters still perform their normal alias/oracle-aware writes.
struct HostVariableBindings {
    std::array<int,size_t(HostVariable::Count)> indices;
    HostVariableBindings() { indices.fill(-1); }
    template<class Resolve,std::enable_if_t<std::is_invocable_r_v<int,Resolve&,const char*>,int> = 0>
    explicit HostVariableBindings(Resolve&& resolve) {
        for(size_t i=0;i<indices.size();++i)indices[i]=resolve(hostVariableNames[i]);
    }
    template<class Write> auto writer(Write&& write) const {
        return [this,write](HostVariable variable,double value){write(indices[size_t(variable)],value);};
    }
};
template<class Write> auto namedHostWriter(Write&& write) {
    return [write](HostVariable variable,double value){write(hostVariableNames[size_t(variable)],value);};
}
template<class Write> void initialiseHostDefaultsVariables(int channels,Write&& write) {
    write(HostVariable::Tempo,120.0);write(HostVariable::Channels,double(channels));
}
template<class Write> void initialiseHostDefaults(int channels,Write&& write) {
    initialiseHostDefaultsVariables(channels,namedHostWriter(write));
}
template<class Write> void initialiseHostDefaults(int channels,const HostVariableBindings& bindings,Write&& write) {
    initialiseHostDefaultsVariables(channels,bindings.writer(write));
}
// The observation is collected by the JUCE adapter. Missing host fields retain
// the last known guest values, matching the production transport contract.
struct TransportFrame {
    HostTransportObservation observation;
    bool haveBeats=false,haveBpm=false;
    double beats=0,bpm=120;
};
#if JUCE_MAJOR_VERSION >= 7
inline TransportFrame collectTransport(const juce::AudioPlayHead::PositionInfo* position) {
    TransportFrame frame;
    if(!position)return frame;
    auto& o=frame.observation;o.valid=true;o.playing=position->getIsPlaying();o.recording=position->getIsRecording();
    if(auto v=position->getTimeInSamples()){o.samples=*v;o.hasSamples=true;}
    if(auto v=position->getTimeInSeconds()){o.seconds=*v;o.hasSeconds=std::isfinite(*v);}
    if(auto v=position->getPpqPosition()){frame.beats=*v;frame.haveBeats=std::isfinite(*v);}
    if(auto v=position->getBpm()){frame.bpm=*v;frame.haveBpm=std::isfinite(*v) && *v>0;}
    return frame;
}
#endif
inline TransportFrame collectTransport(juce::AudioPlayHead* head) {
    if(!head)return {};
#if JUCE_MAJOR_VERSION >= 7
    auto position=head->getPosition();return collectTransport(position?&*position:nullptr);
#else
    TransportFrame frame;juce::AudioPlayHead::CurrentPositionInfo p;
    if(head->getCurrentPosition(p)){
        auto& o=frame.observation;o.valid=true;o.playing=p.isPlaying;o.recording=p.isRecording;
        o.samples=p.timeInSamples;o.hasSamples=true;o.seconds=p.timeInSeconds;o.hasSeconds=std::isfinite(o.seconds);
        frame.beats=p.ppqPosition;frame.haveBeats=std::isfinite(frame.beats);frame.bpm=p.bpm;frame.haveBpm=std::isfinite(frame.bpm) && frame.bpm>0;
    }
    return frame;
#endif
}
template<class Write> bool syncHostTransportVariables(const TransportFrame& frame,HostTransportTracker& tracker,int samples,double rate,int channels,Write&& write) {
    const auto& o=frame.observation;rate=juce::jmax(1.0,rate);
    write(HostVariable::Channels,double(channels));
    bool discontinuity=tracker.update(o,samples,rate);
    write(HostVariable::TransportBound,1);write(HostVariable::TransportValid,o.valid?1:0);write(HostVariable::TransportDiscontinuity,discontinuity?1:0);
    if(o.valid){
        write(HostVariable::PlayState,o.playing?(o.recording?5:1):0);
        if(o.hasSamples)write(HostVariable::PlayPosition,double(o.samples)/rate);else if(o.hasSeconds)write(HostVariable::PlayPosition,o.seconds);
        if(frame.haveBeats)write(HostVariable::BeatPosition,frame.beats);if(frame.haveBpm)write(HostVariable::Tempo,frame.bpm);
    }
    return discontinuity;
}
template<class Write> bool syncHostTransport(const TransportFrame& frame,HostTransportTracker& tracker,int samples,double rate,int channels,Write&& write) {
    return syncHostTransportVariables(frame,tracker,samples,rate,channels,namedHostWriter(write));
}
template<class Write> bool syncHostTransport(const TransportFrame& frame,HostTransportTracker& tracker,int samples,double rate,int channels,const HostVariableBindings& bindings,Write&& write) {
    return syncHostTransportVariables(frame,tracker,samples,rate,channels,bindings.writer(write));
}
template<class Write> void syncGfxActivityVariables(bool visible,bool offline,Write&& write) {
    write(HostVariable::GfxBound,1);write(HostVariable::GfxActive,visible && !offline?1:0);
}
template<class Write> void syncGfxActivity(bool visible,bool offline,Write&& write) {
    syncGfxActivityVariables(visible,offline,namedHostWriter(write));
}
template<class Write> void syncGfxActivity(bool visible,bool offline,const HostVariableBindings& bindings,Write&& write) {
    syncGfxActivityVariables(visible,offline,bindings.writer(write));
}
inline int latencyFromGuest(double raw,int oversamplingFactor) noexcept {
    const double bounded=std::isfinite(raw)?juce::jlimit(0.0,16777216.0,raw):0.0;
    return int(std::ceil(bounded/std::max(1,oversamplingFactor)));
}
}
