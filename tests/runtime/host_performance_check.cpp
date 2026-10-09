// Time the complete production audio callback, including host injection/IPC.
#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
int main(){
    juce::ScopedJuceInitialiser_GUI gui;
    for(int frames:{64,512}){
        std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
        processor->setNonRealtime(true);processor->prepareToPlay(48000,frames);
        juce::AudioBuffer<float> audio(std::max(1,std::max(processor->getTotalNumInputChannels(),processor->getTotalNumOutputChannels())),frames);
        juce::MidiBuffer midi;
        for(int trial=-1;trial<5;++trial){
            double elapsed=0;
            for(int block=0;block<256;++block){
                for(int c=0;c<audio.getNumChannels();++c)for(int f=0;f<frames;++f)
                    audio.setSample(c,f,float(.2*std::sin((block*frames+f)*.057+c)));
                midi.clear();
                if(processor->acceptsMidi() && block%64==0)midi.addEvent(juce::MidiMessage::noteOn(1,60,(juce::uint8)96),0);
                if(processor->acceptsMidi() && block%64==32)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);
                auto start=std::chrono::steady_clock::now();processor->processBlock(audio,midi);
                elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                for(int c=0;c<audio.getNumChannels();++c)for(int f=0;f<frames;++f)
                    if(!std::isfinite(audio.getSample(c,f)))return 2;
            }
            if(trial>=0)std::cout<<frames<<' '<<trial<<' '<<elapsed/256*1e6<<'\n';
        }
        processor->releaseResources();
    }
}
