// Compare real production processors, not a recreated audio callback.
#include <juce_audio_utils/juce_audio_utils.h>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#if JUCE_WINDOWS
#define NOMINMAX
#include <windows.h>
#endif
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
int main() {
#if JUCE_WINDOWS
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
#endif
    juce::ScopedJuceInitialiser_GUI gui;
    try {
        std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());
        p->setNonRealtime(true);
        uint64_t hash=14695981039346656037ull;
        auto add=[&](uint64_t value){hash^=value;hash*=1099511628211ull;};
        for(auto* parameter:p->getParameters()) {
            std::cout<<"PARAM "<<parameter->getName(512)<<' '<<parameter->getDefaultValue()<<'\n';
        }
        for(double rate:{48000.,44100.})for(int choice=0;choice<4;++choice) {
            for(auto* parameter:p->getParameters())
                if(auto* id=dynamic_cast<juce::AudioProcessorParameterWithID*>(parameter))
                    if(id->paramID=="ZA_INTERNAL_OVERSAMPLING")parameter->setValueNotifyingHost(float(choice)/3);
            p->prepareToPlay(rate,2048);
            int position=0;
            for(int repeat=0;repeat<3;++repeat)for(int count:{1,7,64,257,1024}) {
                juce::AudioBuffer<float> audio(std::max(1,std::max(p->getTotalNumInputChannels(),p->getTotalNumOutputChannels())),count);
                for(int c=0;c<audio.getNumChannels();++c)for(int f=0;f<count;++f)
                    audio.setSample(c,f,float(((position+f)*17+c*31)%251-125)/512);
                juce::MidiBuffer midi;
                if(p->acceptsMidi())midi.addEvent(juce::MidiMessage::noteOn(1,60,(juce::uint8)96),0);
                p->processBlock(audio,midi);
                for(int c=0;c<p->getTotalNumOutputChannels();++c)for(int f=0;f<count;++f) {
                    float sample=audio.getSample(c,f);
                    if(!std::isfinite(sample))throw std::runtime_error("Non-finite production output");
                    add(std::bit_cast<uint32_t>(sample));
                }
                for(auto event:midi){add(event.samplePosition);for(int n=0;n<event.numBytes;++n)add(event.data[n]);}
                position+=count;
            }
            std::cout<<"AUDIO "<<rate<<' '<<choice<<' '<<hash<<' '<<p->getLatencySamples()<<'\n';
            p->reset();p->releaseResources();
        }
        juce::MemoryBlock saved;p->getStateInformation(saved);
        p->setStateInformation(saved.getData(),int(saved.getSize()));
        std::cout<<"STATE BYTES "<<saved.getSize()<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
