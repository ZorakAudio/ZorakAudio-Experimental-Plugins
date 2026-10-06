// Cooperative sleep, fresh grants, exact-silence wake and task acknowledgment.
#include <juce_audio_utils/juce_audio_utils.h>
#include <memory>
#include <iostream>
#include <stdexcept>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" void za_smart_idle_mode(juce::AudioProcessor*,int);
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor*);
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
auto make(bool offline=false){std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());p->setNonRealtime(offline);p->setRateAndBufferSizeDetails(48000,256);p->prepareToPlay(48000,256);return p;}
void param(juce::AudioProcessor& p,int index,float value){auto* parameter=dynamic_cast<juce::RangedAudioParameter*>(p.getParameters()[index]);require(parameter!=nullptr,"Missing parameter");parameter->setValueNotifyingHost(parameter->convertTo0to1(value));}
float block(juce::AudioProcessor& p,float value=0,int frames=256){juce::AudioBuffer<float> audio(2,frames);for(int ch=0;ch<2;++ch)for(int f=0;f<frames;++f)audio.setSample(ch,f,value);juce::MidiBuffer midi;p.processBlock(audio,midi);return audio.getSample(0,0);}
double scalar(juce::AudioProcessor& p,const char* name){double value=0;int spans=0;size_t capacity=0;require(za_native_gfx_snapshot(&p,name,&value,&spans,&capacity),"Missing fixture scalar");return value;}
int main()try{
 juce::ScopedJuceInitialiser_GUI gui;
 // No grant: exact silence alone is insufficient.
 {auto p=make();for(int i=0;i<40;++i)block(*p);require(!za_smart_idle_sleeping(p.get()),"Slept without permission");require(scalar(*p,"ticks")==40,"DSP did not advance");}
 // Explicit heuristic modes remain available; the default still honors readiness.
 {for(int mode:{0,1,2,3,4,5}){auto p=make();za_smart_idle_mode(p.get(),mode);for(int i=0;i<100;++i)block(*p);require(za_smart_idle_sleeping(p.get())==(mode==1||mode==2),"Restored mode has wrong sleep semantics");}}
 // User overrides survive project state restoration.
 {auto p=make();za_smart_idle_mode(p.get(),2);juce::MemoryBlock state;p->getStateInformation(state);za_smart_idle_mode(p.get(),4);p->setStateInformation(state.getData(),int(state.getSize()));for(int i=0;i<100;++i)block(*p);require(za_smart_idle_sleeping(p.get()),"Saved event sleep mode lost");p->getStateInformation(state);auto xml=juce::AudioProcessor::getXmlFromBinary(state.getData(),int(state.getSize()));require(bool(xml),"Missing state XML");require(juce::ValueTree::fromXml(*xml).getChildWithName("SMART_IDLE").isValid(),"Selector not persisted");}
 // One early grant must not remain permission for subsequent blocks.
 {auto p=make();param(*p,0,2);for(int i=0;i<40;++i)block(*p);require(!za_smart_idle_sleeping(p.get()),"Stale permission caused sleep");require(scalar(*p,"stale_seen")==0,"Grant wasn't cleared before DSP");}
 // Fresh grants plus zero output permit sleep; a tiny input wakes and survives.
 {auto p=make();param(*p,0,1);for(int i=0;i<10;++i)block(*p);require(za_smart_idle_sleeping(p.get()),"Fresh grants failed to permit sleep");double ticks=scalar(*p,"ticks");block(*p);require(scalar(*p,"ticks")==ticks,"DSP ran during valid sleep");require(block(*p,1e-9f)==0.5e-9f,"Tiny input did not wake intact");require(!za_smart_idle_sleeping(p.get()),"Nonzero input stayed asleep");block(*p);require(za_smart_idle_sleeping(p.get()),"Didn't return to cooperative sleep");param(*p,0,0);block(*p);require(!za_smart_idle_sleeping(p.get()),"Parameter change didn't wake");}
 // Explicit keep-awake overrides otherwise valid grants.
 {auto p=make();param(*p,0,1);for(int i=0;i<10;++i)block(*p);require(za_smart_idle_sleeping(p.get()),"Fixture never slept");double ticks=scalar(*p,"ticks");block(*p,0,128);require(scalar(*p,"ticks")==ticks+1,"Changed buffer length failed to wake DSP");require(!za_smart_idle_sleeping(p.get()),"Old grant survived a block-size change");block(*p,0,128);require(za_smart_idle_sleeping(p.get()),"Fresh grant didn't allow sleep at new block size");}
 {auto p=make();param(*p,0,1);param(*p,2,1);for(int i=0;i<10;++i)block(*p);require(!za_smart_idle_sleeping(p.get()),"Keep-awake veto ignored");param(*p,2,0);for(int i=0;i<10;++i)block(*p);require(za_smart_idle_sleeping(p.get()),"Keep-awake release didn't allow sleep");}
 // Pending or completed-but-unreleased tasks veto sleep; release acknowledges them.
 {auto p=make();param(*p,0,1);param(*p,1,1);for(int i=0;i<40;++i)block(*p);require(scalar(*p,"t")>0,"Task not submitted");require(!za_smart_idle_sleeping(p.get()),"Unacknowledged task lost wake obligation");param(*p,1,2);for(int i=0;i<200 && !za_smart_idle_sleeping(p.get());++i){block(*p);juce::Thread::sleep(1);}require(za_smart_idle_sleeping(p.get()),"Acknowledged task prevented sleep");}
 // Offline processing remains active even with grants and explicit user mode.
 {auto p=make(true);param(*p,0,1);za_smart_idle_mode(p.get(),5);for(int i=0;i<40;++i)block(*p);require(!za_smart_idle_sleeping(p.get()),"Offline cooperative render slept");require(scalar(*p,"ticks")==40,"Offline DSP did not advance");}
 std::cout<<"Cooperative idle: permission, expiration, tiny input, parameter wake, veto, deferred completion and offline checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
