// SPDX-License-Identifier: Zlib
#pragma once
#include "DspJsfxHostTransport.h"

namespace za::jsfx {
template<class Write> void initialiseHostDefaults(int channels,Write&& write) {
    write("tempo",120.0);write("num_ch",double(channels));
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
template<class Write> bool syncHostTransport(const TransportFrame& frame,HostTransportTracker& tracker,int samples,double rate,int channels,Write&& write) {
    const auto& o=frame.observation;rate=juce::jmax(1.0,rate);
    write("num_ch",double(channels));
    bool discontinuity=tracker.update(o,samples,rate);
    write("host_transport_bound",1);write("host_transport_valid",o.valid?1:0);write("host_transport_discontinuity",discontinuity?1:0);
    if(o.valid){
        write("play_state",o.playing?(o.recording?5:1):0);
        if(o.hasSamples)write("play_position",double(o.samples)/rate);else if(o.hasSeconds)write("play_position",o.seconds);
        if(frame.haveBeats)write("beat_position",frame.beats);if(frame.haveBpm)write("tempo",frame.bpm);
    }
    return discontinuity;
}
template<class Write> void syncGfxActivity(bool visible,bool offline,Write&& write) {
    write("host_gfx_bound",1);write("host_gfx_active",visible && !offline?1:0);
}
inline int latencyFromGuest(double raw,int oversamplingFactor) noexcept {
    const double bounded=std::isfinite(raw)?juce::jlimit(0.0,16777216.0,raw):0.0;
    return int(std::ceil(bounded/std::max(1,oversamplingFactor)));
}
}
