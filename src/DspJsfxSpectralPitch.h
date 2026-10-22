// SPDX-License-Identifier: Zlib
#pragma once
#include <signalsmith-stretch/signalsmith-stretch.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace za::jsfx {
// Prepared on the control thread. Input is immutable source audio with lookahead,
// so spectral latency is absorbed by source reads, not by delaying MIDI output.
class SpectralSamplePitch {
    static constexpr int voices = 64, block = 4096, hop = 1024, batch = 256;
    struct Slot {
        signalsmith::stretch::SignalsmithStretch<float> noise{0}, tonal{0};
        std::array<std::array<float, block * 4>, 2> input{};
        std::array<std::array<float, batch>, 2> output{};
        std::uint64_t generation=0;
        double source=0, start=0, frames=0, pitch=0, character=0, rate=0, playbackRate=1, token=0;
        int nextInput=0, outputStart=-batch, lastAge=-1;
        bool active=false;
    };
    std::array<Slot, voices> slots;
    bool prepared=false;
public:
    void prepare() {
        if (prepared) return;
        for (auto& s : slots) {
            for (auto* engine : {&s.noise,&s.tonal}) {
                const int n=engine==&s.noise ? block/2 : block;
                engine->configure(2,n,n/4);
                // Exercise spectral branches and plans before the audio callback.
                for (int i=0; i<block*2; ++i)
                    s.input[0][i]=s.input[1][i]=float(.1*std::sin(i*.057)+.03*std::sin(i*.23));
                engine->setTransposeSemitones(24, 4000.f/48000);
                engine->outputSeek(s.input, engine->outputSeekLength(4));
                engine->process(s.input, batch*4, s.output, batch);
                engine->outputSeek(s.input, engine->outputSeekLength(0.0625));
                engine->process(s.input, batch, s.output, batch);
                engine->reset();
            }
            s.active=false;
        }
        prepared=true;
    }
    template<class Read>
    bool read(int voice, std::uint64_t generation, double source, double start,
              double frames, double nativeRate, double sampleRate, double pitch,
              double character, int age, Read&& sourceRead, double& l, double& r, double playbackRate=1, double token=0) {
        l=r=0;
        if (!prepared || voice<0 || voice>=voices || age<0 || !std::isfinite(pitch) || std::abs(pitch)>96 || !std::isfinite(character)
            || !std::isfinite(nativeRate) || nativeRate<=0 || !std::isfinite(sampleRate)
            || sampleRate<8000 || frames<1 || !std::isfinite(playbackRate) || playbackRate<0.0625 || playbackRate>4 || !std::isfinite(token)) return false;
        auto& s=slots[size_t(voice)];
        auto& engine=character>0.65 ? s.tonal : s.noise;
        auto fill=[&](int at, int count) {
            for (int i=0; i<count; ++i) {
                double position=(at+i)*nativeRate, a=0, b=0;
                if (position>=0 && position<=frames-1) sourceRead(start+position,a,b);
                s.input[0][i]=float(a);s.input[1][i]=float(b);
            }
        };
        const bool fresh=!s.active || age<s.lastAge || s.generation!=generation
            || s.source!=source || s.start!=start || s.frames!=frames || s.pitch!=pitch
            || s.character!=character || s.rate!=nativeRate || s.playbackRate!=playbackRate || s.token!=token;
        if (fresh) {
            engine.reset();
            s.generation=generation;s.source=source;s.start=start;s.frames=frames;
            s.pitch=pitch;s.character=character;s.rate=nativeRate;s.active=true;
            s.playbackRate=playbackRate;s.token=token;s.lastAge=-1;
            // Noise keeps more of its spectral envelope. Strongly periodic
            // bodies retain a wider range of conventional transposition.
            const double limit=2200+5800*std::clamp(character,0.,1.);
            engine.setTransposeSemitones(float(pitch),float(limit/sampleRate));
            s.nextInput=engine.outputSeekLength(float(playbackRate));
            if(s.nextInput>int(s.input[0].size())){s.active=false;return false;}
            fill(0,s.nextInput);engine.outputSeek(s.input,s.nextInput);
            s.outputStart=-batch;
        }
        int work=0;
        while (age>=s.outputStart+batch && ++work<=8) {
            const int at=s.outputStart+batch;
            const int consume=playbackRate==1 ? batch : int(std::floor((at+batch)*playbackRate)-std::floor(at*playbackRate));
            fill(s.nextInput,consume);
            engine.process(s.input,consume,s.output,batch);
            s.nextInput+=consume;s.outputStart+=batch;
        }
        if (age<s.outputStart || age>=s.outputStart+batch) {s.active=false;return false;}
        s.lastAge=age;
        l=s.output[0][size_t(age-s.outputStart)];r=s.output[1][size_t(age-s.outputStart)];
        if (!std::isfinite(l) || !std::isfinite(r)) {l=r=0;s.active=false;return false;}
        return true;
    }
};
}
