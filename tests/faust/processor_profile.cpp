#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" void za_smart_idle_mode(juce::AudioProcessor*,int);
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor*);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
int main(int argc,char** argv)try{
 if(argc<3)throw std::runtime_error("Pass the authorized input recording and report path");
 juce::ScopedJuceInitialiser_GUI gui;juce::AudioFormatManager formats;formats.registerBasicFormats();
 std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(juce::File(juce::String::fromUTF8(argv[1]))));
 if(!reader || reader->numChannels!=2)throw std::runtime_error("Expected the supplied stereo recording");
 const int frames=int(reader->lengthInSamples);const double rate=reader->sampleRate;constexpr int block=256;
 juce::AudioBuffer<float> input(2,frames);if(!reader->read(&input,0,frames,0,true,true))throw std::runtime_error("Input decode failed");reader.reset();
 std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());const int idleMode=argc>3?std::stoi(argv[3]):4;const bool offline=argc>4?std::stoi(argv[4])!=0:true;za_smart_idle_mode(processor.get(),idleMode);processor->setNonRealtime(offline);processor->setRateAndBufferSizeDetails(rate,block);processor->prepareToPlay(rate,block);
 juce::AudioBuffer<float> audio(2,block);juce::MidiBuffer midi;std::ofstream dump(std::string(argv[2])+".pcm.bin",std::ios::binary);
 double dsp=0,maxTime=0;int callbacks=0,sleepBlocks=0;float peak=0;auto start=std::chrono::steady_clock::now();
 for(int at=0;at<frames;at+=block){
  const int n=std::min(block,frames-at);audio.setSize(2,n,false,false,true);for(int ch=0;ch<2;++ch)audio.copyFrom(ch,0,input,ch,at,n);
  midi.clear();auto before=std::chrono::steady_clock::now();processor->processBlock(audio,midi);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();dsp+=seconds;maxTime=std::max(maxTime,seconds);++callbacks;sleepBlocks+=za_smart_idle_sleeping(processor.get());
  peak=std::max(peak,audio.getMagnitude(0,n));for(int ch=0;ch<2;++ch)dump.write(reinterpret_cast<const char*>(audio.getReadPointer(ch)),n*sizeof(float));
 }
 const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();dump.close();
 uint32_t dspFault=0,gfxFault=0;za_native_gfx_faults(processor.get(),&dspFault,&gfxFault);if(dspFault || gfxFault || !std::isfinite(peak) || peak<=0)throw std::runtime_error("Processing faults or invalid output");
 std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditor());if(!editor)throw std::runtime_error("Editor creation failed");
 editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);
 for(int i=0;i<5;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
 editor->createComponentSnapshot(editor->getLocalBounds());processor->editorBeingDeleted(editor.get());editor.reset();
 for(double sr:{48000.,96000.}){processor->prepareToPlay(sr,1024);audio.setSize(2,1024);audio.clear();midi.clear();processor->processBlock(audio,midi);}
 za_native_gfx_faults(processor.get(),&dspFault,&gfxFault);if(dspFault || gfxFault)throw std::runtime_error("Reset/editor faults");
 std::ofstream report(argv[2]);report<<"{\"idle_mode\":"<<idleMode<<",\"offline\":"<<(offline?"true":"false")<<",\"sleep_blocks\":"<<sleepBlocks<<",\"sample_rate\":"<<rate<<",\"block_size\":"<<block<<",\"frames\":"<<frames<<",\"wall_seconds_with_output_dump\":"<<wall<<",\"processBlock_seconds\":"<<dsp<<",\"max_callback_ms\":"<<maxTime*1000<<",\"callbacks\":"<<callbacks<<",\"peak\":"<<peak<<",\"dsp_faults\":"<<dspFault<<",\"gfx_faults\":"<<gfxFault<<",\"editor_and_rate_reset_passed\":true}";
 std::cout<<"Full JUCE processor: "<<dsp<<" processing seconds for "<<frames/rate<<" seconds of supplied audio; editor and resets passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
