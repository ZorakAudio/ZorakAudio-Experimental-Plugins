#include <juce_audio_utils/juce_audio_utils.h>
#include <iostream>
#include <stdexcept>
#include <thread>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
extern "C" double za_native_gfx_heap_value(juce::AudioProcessor*,const char*,int);
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor*,const char* const*,int);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
int main() try {
    juce::ScopedJuceInitialiser_GUI gui;
    auto folder=juce::File::getCurrentWorkingDirectory().getChildFile("build/tasks/corpus/fixtures");
    folder.createDirectory();
    auto wave=folder.getChildFile("corpus.wav");
    auto output=wave.createOutputStream(); require(output!=nullptr,"Cannot create fixture");
    output->setPosition(0);output->truncate();
    juce::WavAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(output.get(),48000,2,24,{},0));
    require(writer!=nullptr,"Cannot encode fixture");output.release();
    juce::AudioBuffer<float> fixture(2,48000*8);
    for(int i=0;i<fixture.getNumSamples();++i) for(int c=0;c<2;++c)
        fixture.setSample(c,i,float(.3*std::sin(i*2*juce::MathConstants<double>::pi*(220+110*(i/48000))/48000)));
    require(writer->writeFromAudioSampleBuffer(fixture,0,fixture.getNumSamples()),"WAV write failed");writer.reset();
    std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());p->prepareToPlay(48000,256);
    auto value=[&](const char* name){double v=0;int spans=0;size_t capacity=0;
        require(za_native_gfx_snapshot(p.get(),name,&v,&spans,&capacity),name);return v;};
    auto replacement=folder.getChildFile("replacement.wav");wave.copyFileTo(replacement);
    auto path=wave.getFullPathName().toStdString();const char* paths[]={path.c_str()};
    auto nextPath=replacement.getFullPathName().toStdString();const char* nextPaths[]={nextPath.c_str()};
    za_sample_gfx_load_bank(p.get(),paths,1);
    juce::AudioBuffer<float> audio(std::max(2,p->getTotalNumOutputChannels()),256);juce::MidiBuffer midi;
    bool ready=false,reloaded=false,sawTask=false;int previousPhase=-1;double reloadEpoch=-1;
    for(int attempt=0;attempt<60000;++attempt) {
        audio.clear();midi.clear();p->processBlock(audio,midi);
        int phase=int(value("xc_phase"));
        if(phase!=previousPhase){std::cout<<"analysis phase "<<phase<<std::endl;previousPhase=phase;}
        if(value("xc_task")>0) {
            sawTask=true;
            if(!reloaded){reloadEpoch=value("index_epoch");za_sample_gfx_load_bank(p.get(),nextPaths,1);reloaded=true;}
        }
        if(value("pe_ready")>0 && value("xc_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && reloaded && value("index_epoch")!=reloadEpoch){ready=true;break;}
        if(attempt%16==0)juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    require(ready,"Corpus preprocessing did not complete after replacement");require(sawTask,"No parallel comparison task was submitted");
    require(value("xc_task_fallbacks")==0,"Worker failed and used fallback");
    const int n=int(value("xc_coarse_n"));double maxError=0;
    auto cell=[&](int i){return za_native_gfx_heap_value(p.get(),"xc_coarse",i);};
    for(int a=0;a<n;++a)for(int b=0;b<n;++b){double v=0,h=0;
        for(int i=0;i<8;++i)v+=std::abs(std::sqrt(std::max(0.,cell(a*48+i)))-std::sqrt(std::max(0.,cell(b*48+i))))*.065+std::abs(cell(a*48+20+i)-cell(b*48+20+i))*.035;
        for(int i=0;i<12;++i)h+=std::abs(cell(a*48+8+i)-cell(b*48+8+i));
        double expected=std::clamp(1-v-.40*std::min(cell(a*48+28),cell(b*48+28))*h*.5,0.,1.);
        maxError=std::max(maxError,std::abs(expected-za_native_gfx_heap_value(p.get(),"xc_matrix",a*512+b)));
    }
    require(maxError<1e-12,"Parallel matrix differs from original formula");
    double peak=0;
    for(int block=0;block<400;++block){audio.clear();midi.clear();if(block==0)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);
        p->processBlock(audio,midi);for(int i=0;i<256;++i){double sample=audio.getSample(0,i);require(std::isfinite(sample),"Nonfinite audio");peak=std::max(peak,std::abs(sample));}}
    require(peak>1e-5,"Corpus did not play after migration");
    std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditorIfNeeded());require(editor!=nullptr,"Editor failed");
    editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);
    for(int i=0;i<20;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    uint32_t dsp=0,gfx=0;za_native_gfx_faults(p.get(),&dsp,&gfx);require(dsp==0 && gfx==0,"Runtime memory fault");
    p->editorBeingDeleted(editor.get());editor.reset();p->releaseResources();
    std::cout<<"Corpus load/replacement, parallel matrix (max error "<<maxError<<"), MIDI playback and editor passed\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
