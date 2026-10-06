// Exercise the real host Smart Idle path against an always-active reference.
#include <juce_audio_utils/juce_audio_utils.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" void za_smart_idle_mode(juce::AudioProcessor*,int);
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor*);
struct Result {uint64_t different=0,sleepBlocks=0,wakes=0,differentAwake=0;int first=-1,last=-1;double peak=0,squares=0;};
std::unique_ptr<juce::AudioProcessor> make(double rate,int mode,bool offline){
 std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());p->setNonRealtime(offline);za_smart_idle_mode(p.get(),mode);p->setRateAndBufferSizeDetails(rate,256);p->prepareToPlay(rate,256);return p;
}
void run(juce::AudioProcessor& p,const juce::AudioBuffer<float>& input,juce::AudioBuffer<float>& output){
 juce::AudioBuffer<float> block(2,256);juce::MidiBuffer midi;
 for(int at=0;at<input.getNumSamples();at+=256){const int n=std::min(256,input.getNumSamples()-at);block.setSize(2,n,false,false,true);for(int ch=0;ch<2;++ch)block.copyFrom(ch,0,input,ch,at,n);midi.clear();p.processBlock(block,midi);for(int ch=0;ch<2;++ch)output.copyFrom(ch,at,block,ch,0,n);}
}
Result compare(double rate,int mode,bool offline,bool toggle,const juce::AudioBuffer<float>& input,const juce::AudioBuffer<float>& reference){
 auto p=make(rate,mode,offline);juce::AudioBuffer<float> block(2,256);juce::MidiBuffer midi;Result r;bool wasSleeping=false;
 for(int at=0;at<input.getNumSamples();at+=256){
  const int n=std::min(256,input.getNumSamples()-at);block.setSize(2,n,false,false,true);for(int ch=0;ch<2;++ch)block.copyFrom(ch,0,input,ch,at,n);
  if(toggle && at==int(rate*4)/256*256)za_smart_idle_mode(p.get(),4);
  midi.clear();p->processBlock(block,midi);bool sleeping=za_smart_idle_sleeping(p.get());r.sleepBlocks+=sleeping;r.wakes+=wasSleeping&&!sleeping;wasSleeping=sleeping;
  for(int ch=0;ch<2;++ch)for(int f=0;f<n;++f){float v=block.getSample(ch,f),ref=reference.getSample(ch,at+f);double error=double(v)-ref;if(!std::isfinite(v))throw std::runtime_error("Nonfinite output");if(v!=ref){++r.different;if(!sleeping)++r.differentAwake;if(r.first<0)r.first=at+f;r.last=at+f;}r.peak=std::max(r.peak,std::abs(error));r.squares+=error*error;}
 }
 return r;
}
int main(int argc,char**argv)try{
 if(argc<3)throw std::runtime_error("Pass supplied recording and JSON report");juce::ScopedJuceInitialiser_GUI gui;juce::AudioFormatManager formats;formats.registerBasicFormats();
 std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(juce::File(juce::String::fromUTF8(argv[1]))));if(!reader||reader->numChannels!=2)throw std::runtime_error("Expected supplied stereo recording");const double rate=reader->sampleRate;const int frames=int(reader->lengthInSamples);
 juce::AudioBuffer<float> recording(2,frames);if(!reader->read(&recording,0,frames,0,true,true))throw std::runtime_error("Decode failed");reader.reset();
 juce::AudioBuffer<float> stress(2,int(rate*6));stress.clear();const float inputPeak=recording.getMagnitude(0,std::min(frames,int(rate)));const float quietScale=6e-6f/std::max(inputPeak,1e-20f);
 for(int ch=0;ch<2;++ch)for(int f=0;f<stress.getNumSamples();++f){if(f<int(rate)||f>=int(rate*4))stress.setSample(ch,f,recording.getSample(ch,f%int(rate)));else if(f>=int(rate*3))stress.setSample(ch,f,recording.getSample(ch,f%int(rate))*quietScale);}
 std::ofstream report(argv[2]);report<<"{\"sample_rate\":"<<rate<<",\"block_size\":256,\"cases\":[";bool first=true;
 for(int fixture=0;fixture<2;++fixture){const auto& input=fixture==0?recording:stress;juce::AudioBuffer<float> reference(2,input.getNumSamples());auto active=make(rate,4,false);run(*active,input,reference);active.reset();
  for(bool offline:{false,true})for(int mode:{0,1,2,3,4,5}){
   auto result=compare(rate,mode,offline,false,input,reference);
   if(result.different || result.sleepBlocks)throw std::runtime_error("Plugin without an explicit grant did not null against Active");
   if(!first)report<<',';first=false;
   report<<"{\"fixture\":\""<<(fixture==0?"supplied_recording":"quiet_and_recovery_from_supplied_recording")<<"\",\"frames\":"<<input.getNumSamples()<<",\"offline\":"<<(offline?"true":"false")<<",\"mode\":"<<mode<<",\"sleep_blocks\":"<<result.sleepBlocks<<",\"wakes\":"<<result.wakes<<",\"different_samples\":"<<result.different<<",\"different_samples_while_awake\":"<<result.differentAwake<<",\"first_different_frame\":"<<result.first<<",\"last_different_frame\":"<<result.last<<",\"max_error\":"<<result.peak<<",\"rms_error\":"<<std::sqrt(result.squares/(2.*input.getNumSamples()))<<"}";
   std::cout<<(fixture==0?"recording":"quiet/recovery")<<" offline="<<offline<<" mode="<<mode<<" sleep="<<result.sleepBlocks<<" different="<<result.different<<" max="<<result.peak<<'\n'<<std::flush;
  }
  if(fixture==1){auto r=compare(rate,1,false,true,input,reference);report<<",{\"fixture\":\"legacy_selector_change_during_playback\",\"offline\":false,\"mode\":1,\"different_samples\":"<<r.different<<",\"different_samples_while_awake\":"<<r.differentAwake<<",\"max_error\":"<<r.peak<<",\"first_different_frame\":"<<r.first<<"}";}
 }
 report<<"]}";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
