#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor*);
extern "C" void za_smart_idle_mode(juce::AudioProcessor*,int);
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
struct Playhead:juce::AudioPlayHead{bool playing=false;int64_t position=0;juce::Optional<PositionInfo> getPosition()const override{PositionInfo p;p.setIsPlaying(playing);p.setTimeInSamples(position);p.setTimeInSeconds(double(position)/48000);p.setBpm(120);return p;}};
void set(juce::AudioProcessor& p,int n,float v){for(auto* param:p.getParameters())if(auto* x=dynamic_cast<juce::RangedAudioParameter*>(param))if(x->paramID=="slider"+juce::String(n)){x->setValueNotifyingHost(x->convertTo0to1(v));return;}throw std::runtime_error("Missing slider");}
double ticks(juce::AudioProcessor& p){double v=0;int count=0;size_t cap=0;require(za_native_gfx_snapshot(&p,"samples_since_init",&v,&count,&cap),"Missing sample count");return v;}
int main()try{juce::ScopedJuceInitialiser_GUI gui;for(int model:{0,1}){
 auto p=std::unique_ptr<juce::AudioProcessor>(createPluginFilter());Playhead head;p->setPlayHead(&head);p->setNonRealtime(false);p->setRateAndBufferSizeDetails(48000,256);p->prepareToPlay(48000,256);set(*p,36,float(model));juce::AudioBuffer<float> audio(2,256);juce::MidiBuffer midi;
 auto block=[&](float level=0){audio.clear();if(level!=0)for(int ch=0;ch<2;++ch)for(int k=0;k<256;++k)audio.setSample(ch,k,level);midi.clear();p->processBlock(audio,midi);if(head.playing)head.position+=256;for(int ch=0;ch<2;++ch)for(int k=0;k<256;++k)require(std::isfinite(audio.getSample(ch,k)),"Nonfinite output");};
 for(int i=0;i<240;++i)block();require(za_smart_idle_sleeping(p.get()),"Paused silence did not sleep");double t=ticks(*p);for(int i=0;i<100;++i)block();require(ticks(*p)==t,"Paused idle DSP advanced");
 block(1e-9f);require(!za_smart_idle_sleeping(p.get())&&ticks(*p)==t+256,"Tiny input failed to wake");for(int i=0;i<240;++i)block();require(za_smart_idle_sleeping(p.get()),"Failed to sleep again");
 set(*p,6,-.4f);block();require(!za_smart_idle_sleeping(p.get()),"Parameter failed to wake");
 head.playing=true;for(int i=0;i<240;++i)block();require(za_smart_idle_sleeping(p.get()),"Playing empty track did not sleep");
 // Let a real finite tail decay before permitting sleep.
 set(*p,27,.5f);set(*p,7,.25f);for(int i=0;i<100;++i)block(.03f);require(!za_smart_idle_sleeping(p.get()),"Slept through active audio");block();require(!za_smart_idle_sleeping(p.get()),"Cut tail immediately");int drained=0;for(;drained<12000&&!za_smart_idle_sleeping(p.get());++drained)block();require(za_smart_idle_sleeping(p.get()),"Tail never settled enough to sleep");
 set(*p,27,0);for(int i=0;i<240;++i)block();
 auto editor=std::unique_ptr<juce::AudioProcessorEditor>(p->createEditor());editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);for(int i=0;i<300;++i){block();if(i%10==0)juce::MessageManager::getInstance()->runDispatchLoopUntil(5);}require(za_smart_idle_sleeping(p.get()),"Open canvas prevented sleep");t=ticks(*p);for(int i=0;i<60;++i){block();juce::MessageManager::getInstance()->runDispatchLoopUntil(5);}require(ticks(*p)==t,"Unchanged canvas kept waking DSP");p->editorBeingDeleted(editor.get());editor.reset();
 za_smart_idle_mode(p.get(),4);for(int i=0;i<10;++i)block();require(!za_smart_idle_sleeping(p.get()),"Never Sleep override ignored");
 juce::MemoryBlock state;p->getStateInformation(state);za_smart_idle_mode(p.get(),1);p->setStateInformation(state.getData(),int(state.getSize()));for(int i=0;i<240;++i)block();require(!za_smart_idle_sleeping(p.get()),"Saved Never Sleep override lost");za_smart_idle_mode(p.get(),0);for(int i=0;i<240;++i)block();require(za_smart_idle_sleeping(p.get()),"Auto reset failed");
 auto measure=[&](int mode){za_smart_idle_mode(p.get(),mode);for(int i=0;i<240;++i)block();auto start=std::chrono::steady_clock::now();for(int i=0;i<1000;++i)block();return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};double active=measure(4),idle=measure(0);require(idle<active*.3,"Idle CPU reduction too small");
 p->setNonRealtime(true);t=ticks(*p);for(int i=0;i<240;++i)block();require(!za_smart_idle_sleeping(p.get())&&ticks(*p)==t+240*256,"Offline rendering suspended DSP");
 std::cout<<"MODEL "<<model<<" PASS; 1000 idle callbacks "<<idle<<"s vs active "<<active<<"s; saved "<<100*(1-idle/active)<<"%; tail drain "<<drained*256/48000.<<"s\n";
}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
