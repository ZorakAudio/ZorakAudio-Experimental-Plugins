#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
extern "C" void za_native_faust_calls(juce::AudioProcessor*,uint64_t*,uint64_t*,uint64_t*);
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void set(juce::AudioProcessor&p,int n,float v){for(auto* param:p.getParameters())if(auto*x=dynamic_cast<juce::RangedAudioParameter*>(param))if(x->paramID=="slider"+juce::String(n)){x->setValueNotifyingHost(x->convertTo0to1(v));return;}throw std::runtime_error("missing slider ID");}
double value(juce::AudioProcessor&p,const char*s){double v;int c;size_t cap;require(za_native_gfx_snapshot(&p,s,&v,&c,&cap),"snapshot");return v;}
int main(int argc,char**argv)try{require(argc>=5,"report rate block scenario");juce::ScopedJuceInitialiser_GUI gui;int rate=std::stoi(argv[2]),block=std::stoi(argv[3]),scenario=std::stoi(argv[4]);
 std::unique_ptr<juce::AudioProcessor> p[2];for(int j=0;j<2;++j){p[j].reset(createPluginFilter());set(*p[j],12,float(j+1));p[j]->setNonRealtime(true);p[j]->setRateAndBufferSizeDetails(rate,block);p[j]->prepareToPlay(rate,block);set(*p[j],12,float(j+1));set(*p[j],1,j?2:1);set(*p[j],2,scenario?85:0);set(*p[j],3,scenario?65:0);}
 juce::AudioBuffer<float>a(2,block);juce::MidiBuffer midi;std::vector<float> reference(2*block);std::ofstream binary(std::string(argv[1])+".bin",std::ios::binary);double seconds=0,peak=0,change=0;uint64_t calls=0;int frames=rate*6,callbacks=0;
 for(int at=0;at<frames;at+=block){int n=std::min(block,frames-at);if(scenario==2&&at%(std::max(1,rate/2/block)*block)==0){for(int j=0;j<2;++j){set(*p[j],1,float((at/(rate/2)+j)%8));set(*p[j],2,100);set(*p[j],3,100);set(*p[j],4,8);set(*p[j],5,100);set(*p[j],6,.95f);set(*p[j],7,-80);set(*p[j],8,0);set(*p[j],9,12);}}a.setSize(2,n,false,false,true);if(scenario==3&&at%(std::max(1,rate/2/block)*block)==0){for(int j=0;j<2;++j)set(*p[j],11,float(8+4*((at/(rate/2)+j)%5)));}if(at>=rate*3&&at<rate*3+block&&scenario){set(*p[0],1,4);set(*p[1],3,90);}
  for(int j=0;j<2;++j){for(int ch=0;ch<2;++ch)for(int k=0;k<n;++k){double t=double(at+k)/rate;float x=at<rate*5?float((j?.05:.025)*(sin(2*3.141592653589793*(j?320:310)*t)+.4*sin(2*3.141592653589793*(ch?3100:1800)*t))):0;a.setSample(ch,k,x);reference[ch*block+k]=x;}auto input=a.getSample(0,0);midi.clear();auto start=std::chrono::steady_clock::now();p[j]->processBlock(a,midi);seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();++callbacks;change=std::max(change,std::abs(double(a.getSample(0,0)-input)));for(int ch=0;ch<2;++ch){for(int k=0;k<n;++k){double x=a.getSample(ch,k);require(std::isfinite(x)&&std::abs(x)<=8.001,"bad output");peak=std::max(peak,std::abs(x));change=std::max(change,std::abs(x-reference[ch*block+k]));}binary.write((const char*)a.getReadPointer(ch),n*sizeof(float));}}
 }
 std::cout<<"IDs "<<value(*p[0],"iid")<<","<<value(*p[1],"iid")<<"; peers "<<value(*p[0],"active_peer_instances")<<","<<value(*p[1],"active_peer_instances")<<"\n";
 if(scenario==3)require(value(*p[0],"active_nb")==12&&value(*p[1],"active_nb")==16,"variable-band schedule did not execute");
 require(value(*p[0],"active_peer_instances")==1&&value(*p[1],"active_peer_instances")==1,"instances did not discover one another");
 uint64_t blocks=0,scalars=0,f=0;za_native_faust_calls(p[0].get(),&blocks,&scalars,&f);require(scalars==0,"scalar FAUST calls");if(blocks)require(blocks==uint64_t((frames+block-1)/block),"not one FAUST call per buffer");
 if(scenario)require(change>1e-5,"peer policy did not affect audio");else if(argc<6) require(change<1e-6,"neutral output does not reconstruct input");
 juce::MemoryBlock state;p[0]->getStateInformation(state);p[0]->setStateInformation(state.getData(),int(state.getSize()));
 auto editor=std::unique_ptr<juce::AudioProcessorEditor>(p[0]->createEditor());require(bool(editor),"editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);for(int i=0;i<3;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(10);p[0]->editorBeingDeleted(editor.get());editor.reset();
 p[0]->prepareToPlay(96000,128);a.setSize(2,128);a.clear();p[0]->processBlock(a,midi);require(std::isfinite(a.getMagnitude(0,128)),"rate reset output");
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p[0].get(),&dsp,&gfx);require(!dsp&&!gfx,"DSP/GFX faults");
 std::ofstream report(argv[1]);report<<"{\"seconds\":"<<seconds<<",\"peak\":"<<peak<<",\"maximum_change\":"<<change<<",\"block_calls\":"<<blocks<<",\"scalar_calls\":"<<scalars<<",\"frames_per_instance\":"<<frames<<",\"peer_count\":1,\"editor_state_passed\":true}";
 std::cout<<"PASS "<<rate<<"/"<<block<<" scenario "<<scenario<<"; two instances "<<seconds<<"s; FAUST "<<blocks<<" block / "<<scalars<<" scalar calls\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
