#include <juce_audio_utils/juce_audio_utils.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <chrono>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor*);
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void param(juce::AudioProcessor& p,int index,float value){auto* v=dynamic_cast<juce::RangedAudioParameter*>(p.getParameters()[index]);require(v,"Missing parameter");v->setValueNotifyingHost(v->convertTo0to1(value));}
int main(int argc,char** argv)try{
 juce::ScopedJuceInitialiser_GUI gui;const std::string key=argc>1?argv[1]:"DPT";
 const bool ddt=key=="DDT",scalar=key=="ADS"||key=="SaliencePush";
 for(int rate:{48000,96000})for(int block:{64,256,1024})for(int mode:{0,1}){
  if(scalar&&block!=(rate==48000?256:1024))continue;
  std::unique_ptr<juce::AudioProcessor> reference(createPluginFilter()),candidate(createPluginFilter());
  reference->setNonRealtime(true);candidate->setNonRealtime(false);
  for(auto* p:{reference.get(),candidate.get()}){p->setRateAndBufferSizeDetails(rate,block);p->prepareToPlay(rate,block);if(!scalar)param(*p,ddt?4:2,ddt?mode*4:mode);else if(mode)param(*p,0,key=="ADS"?90:1);}
  int slept=0;double activeTime=0,sleepTime=0;const int channels=scalar?4:2;juce::AudioBuffer<float> a(channels,block),b(channels,block);juce::MidiBuffer ma,mb;
  const int blocks=(rate*(scalar?124:12)+block-1)/block;
  for(int n=0;n<blocks;++n){
   if(n==rate*(scalar?122:10)/block){for(auto* p:{reference.get(),candidate.get()}){param(*p,0,scalar?(key=="ADS"?90:2):73);param(*p,scalar?1:ddt?8:2,scalar?75:ddt?80:1);}}
   for(int k=0;k<block;++k){double t=double(n*block+k)/rate;bool active=scalar?((t>=120&&t<121)||t>=123):(t<2||t>=11);float x=active?float(0.1*std::sin(431*t)+0.03*std::sin(2703*t)):t>=(scalar?122:10)?1e-9f:0;for(int ch=0;ch<channels;++ch)a.setSample(ch,k,x*(ch%2?0.7f:1));}
   b.makeCopyOf(a);ma.clear();mb.clear();
   auto start=std::chrono::steady_clock::now();reference->processBlock(a,ma);activeTime+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   start=std::chrono::steady_clock::now();candidate->processBlock(b,mb);sleepTime+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   slept+=za_smart_idle_sleeping(candidate.get());require(!za_smart_idle_sleeping(reference.get()),"Offline reference slept");
   for(int ch=0;ch<2;++ch)for(int k=0;k<block;++k)require(a.getSample(ch,k)==b.getSample(ch,k),"Cooperative wake output failed null test");
  }
  require(slept>0,"Fixture never slept");std::cout<<key<<" rate="<<rate<<" block="<<block<<" mode="<<mode<<" slept="<<slept<<" awake_seconds="<<activeTime<<" cooperative_seconds="<<sleepTime<<" NULL PASS\n";
 }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
