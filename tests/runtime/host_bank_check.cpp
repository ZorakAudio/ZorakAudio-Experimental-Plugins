#define NOMINMAX
#include "SyntheticBank.h"
#include <iostream>
#include <windows.h>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor*,const char* const*,int);
extern "C" bool za_sample_gfx_snapshot(juce::AudioProcessor*,const char*,double*,size_t*,size_t*,int64_t*);
static void pump(){MSG m;while(PeekMessageW(&m,nullptr,0,0,PM_REMOVE)){TranslateMessage(&m);DispatchMessageW(&m);}}
int main(int argc,char** argv)try{
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
    juce::ScopedJuceInitialiser_GUI gui;const bool corpus=argc>1 && juce::String(argv[1])=="Corpus";
    auto folder=juce::File::getSpecialLocation(juce::File::tempDirectory).getNonexistentChildFile("za-aot-bank","",false);
    struct Cleanup{juce::File f;~Cleanup(){if(f.getParentDirectory()==juce::File::getSpecialLocation(juce::File::tempDirectory) && f.getFileName().startsWith("za-aot-bank"))f.deleteRecursively();}}cleanup{folder};
    auto files=makeRuntimeBank(folder);std::vector<std::string> paths;std::vector<const char*> pointers;for(auto& file:files)paths.push_back(file.toStdString());for(auto& path:paths)pointers.push_back(path.c_str());
    std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());p->setNonRealtime(true);p->prepareToPlay(48000,4096);
    std::unique_ptr<juce::AudioProcessorEditor> e(p->createEditorIfNeeded());if(!e)throw std::runtime_error("Production editor missing");
    za_sample_gfx_load_bank(p.get(),pointers.data(),3);
    auto value=[&](const char* name){double v=0;size_t copied=0,capacity=0;int64_t logical=0;za_sample_gfx_snapshot(p.get(),name,&v,&copied,&capacity,&logical);return v;};
    juce::AudioBuffer<float> audio(std::max(2,std::max(p->getTotalNumInputChannels(),p->getTotalNumOutputChannels())),256);juce::MidiBuffer midi;bool ready=false;const auto started=juce::Time::getMillisecondCounterHiRes();int block=0;
    while(juce::Time::getMillisecondCounterHiRes()-started<60000){audio.clear();midi.clear();p->processBlock(audio,midi);pump();if(++block%10==0)e->createComponentSnapshot(e->getLocalBounds());
        ready=corpus?value("loaded_count")==3 && value("ready")>0 && value("xc_ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0:value("sample_count")==3 && value("bank_loaded")>0 && value("rescan_pending")==0;
        if(ready)break;juce::Thread::sleep(2);
    }
    if(!ready){for(auto name:{"loaded_count","sample_count","ready","build_phase","cg_stage","cg_disabled","pool_state","pool_loaded","grain_count","corpus_count","xc_ready","pe_ready","rescan_pending"})std::cerr<<name<<'='<<value(name)<<'\n';throw std::runtime_error("Actual production analysis did not complete");}
    double peak=0;for(int n=0;n<400;++n){audio.clear();midi.clear();if(n==0)midi.addEvent(juce::MidiMessage::noteOn(1,60,(juce::uint8)100),0);if(n==300)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);p->processBlock(audio,midi);
        for(int c=0;c<p->getTotalNumOutputChannels();++c)for(int f=0;f<256;++f){double v=audio.getSample(c,f);if(!std::isfinite(v))throw std::runtime_error("Loaded production output not finite");peak=std::max(peak,std::abs(v));}}
    if(peak<1e-6)throw std::runtime_error("Loaded production instrument produced no output");
    if(!corpus){for(auto* parameter:p->getParameters())if(parameter->getName(200).containsIgnoreCase("Playback mode"))parameter->setValueNotifyingHost(1);
        audio.clear();midi.clear();p->processBlock(audio,midi);double granularPeak=0;
        for(int n=0;n<400;++n){audio.clear();midi.clear();if(n==0)midi.addEvent(juce::MidiMessage::noteOn(1,60,(juce::uint8)100),0);if(n==300)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);p->processBlock(audio,midi);for(int c=0;c<p->getTotalNumOutputChannels();++c)for(int f=0;f<256;++f){double v=audio.getSample(c,f);if(!std::isfinite(v))throw std::runtime_error("Loaded granular output not finite");granularPeak=std::max(granularPeak,std::abs(v));}}
        if(granularPeak<1e-6)throw std::runtime_error("Loaded granular instrument produced no output");std::cout<<"GRANULAR PASS peak="<<granularPeak<<'\n';}
    e->createComponentSnapshot(e->getLocalBounds());juce::MemoryBlock saved;p->getStateInformation(saved);e.reset();p->releaseResources();
    std::cout<<"LOADED PASS files=3 peak="<<peak<<" grains="<<value(corpus?"grain_count":"corpus_count")<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
