#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
extern "C" void za_native_faust_calls(juce::AudioProcessor*,uint64_t*,uint64_t*,uint64_t*);
using Clock=std::chrono::steady_clock;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void control(juce::AudioProcessor& p,int slider,float plain){
 for(auto* parameter:p.getParameters())if(auto* x=dynamic_cast<juce::RangedAudioParameter*>(parameter))if(x->paramID=="slider"+juce::String(slider)){x->setValueNotifyingHost(x->convertTo0to1(plain));return;}
 throw std::runtime_error("Missing slider ID "+std::to_string(slider));
}
int main(int argc,char** argv)try{
 require(argc>=6 && (argc-6)%2==0,"Pass recording, report, rate, block, scenario, then optional slider/value pairs");
 require(juce::String::fromUTF8(argv[1])=="C:\\Users\\Louis\\OneDrive\\Sound Effects\\NSFW\\Mouth Noises\\Files to Cut\\Slapping dirty slut BJ.flac","Only authorized source recording may load");
 const int rate=std::stoi(argv[3]),block=std::stoi(argv[4]),scenario=std::stoi(argv[5]);
 require(rate>=22050&&rate<=192000&&block>=1&&block<=4096&&scenario>=0&&scenario<=8,"Invalid fixture settings");
 juce::ScopedJuceInitialiser_GUI gui;juce::AudioFormatManager fm;fm.registerBasicFormats();
 auto reader=std::unique_ptr<juce::AudioFormatReader>(fm.createReaderFor(juce::File(juce::String::fromUTF8(argv[1]))));require(bool(reader),"Source decode");
 const int inputFrames=int(reader->sampleRate*6);juce::AudioBuffer<float> recording(2,inputFrames);
 require(reader->read(&recording,0,inputFrames,int64_t(reader->sampleRate*scenario*30),true,true),"Authorized excerpt decode");
 const double sourceRate=reader->sampleRate;reader.reset();
 auto p=std::unique_ptr<juce::AudioProcessor>(createPluginFilter());p->setNonRealtime(true);p->setRateAndBufferSizeDetails(rate,block);p->prepareToPlay(rate,block);
 float requestedModel=0;auto set=[&](int s,float v){control(*p,s,v);if(s==36)requestedModel=v>=.5f?1:0;};
 set(36,scenario==0?0:1);set(6,scenario==7?-1:scenario==8?1:.55f);set(8,.72f);set(7,.25f);set(14,0);
 if(scenario>=3&&scenario<=5){set(22,float(scenario-2));set(10,.65f);set(23,1);}
 if(scenario==5){set(27,.8f);set(13,.75f);set(21,.6f);set(39,1);set(7,.55f);}
 for(int arg=6;arg<argc;arg+=2)set(std::stoi(argv[arg]),std::stof(argv[arg+1]));
 const int frames=rate*8;juce::AudioBuffer<float> audio(std::max(2,p->getTotalNumOutputChannels()),block);juce::MidiBuffer midi;
 std::ofstream dump(std::string(argv[2])+".pcm.bin",std::ios::binary);double seconds=0,peak=0,energyL=0,energyR=0,tail=0,maxStep=0;float previous[2]{};int callbacks=0;
 for(int at=0;at<frames;at+=block){
  if((scenario==2||scenario==6)&&at%(std::max(1,rate/4/block)*block)==0){double t=double(at)/rate;set(6,float(.9*std::sin(1.7*t)));set(8,float(.9*std::cos(1.7*t)));set(21,float(.6*std::sin(.7*t)));set(41,float(120*std::sin(.5*t)));set(42,float(45*std::cos(.8*t)));set(43,float(70*std::sin(.3*t)));set(38,float(.5+.5*std::sin(t)));set(37,float((at/(rate/2))%3));set(44,float(2+12*(.5+.5*std::sin(t))));set(45,float(.4+1.8*(.5+.5*std::cos(t))));set(39,float((at/rate)%2));}
  if(scenario==6){if(at>=rate*2&&at<rate*4)set(36,0);else set(36,1);set(27,at>=rate*3&&at<rate*6?.8f:0);set(22,float((at/rate)%4));}
  const int n=std::min(block,frames-at);audio.setSize(std::max(2,p->getTotalNumOutputChannels()),n,false,false,true);audio.clear();
  for(int k=0;k<n;++k){double pos=double(at+k)*sourceRate/rate;int i=int(pos);double f=pos-i;if(i+1<inputFrames)for(int ch=0;ch<2;++ch){float v=float(recording.getSample(ch,i)*(1-f)+recording.getSample(ch,i+1)*f);if(scenario>=7)v=float(.05*std::sin(2*juce::MathConstants<double>::pi*700*(at+k)/rate));audio.setSample(ch,k,v);}}
  midi.clear();auto begin=Clock::now();p->processBlock(audio,midi);seconds+=std::chrono::duration<double>(Clock::now()-begin).count();++callbacks;
  for(int ch=0;ch<2;++ch){auto* data=audio.getReadPointer(ch);dump.write(reinterpret_cast<const char*>(data),n*sizeof(float));for(int k=0;k<n;++k){double x=data[k];require(std::isfinite(x),"Nonfinite output");peak=std::max(peak,std::abs(x));(ch?energyR:energyL)+=x*x;if(at+k>=rate*6)tail+=x*x;maxStep=std::max(maxStep,std::abs(x-previous[ch]));previous[ch]=float(x);}}
 }
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p.get(),&dsp,&gfx);require(!dsp&&!gfx,"Processing memory fault");require(peak>0&&peak<16,"Invalid output magnitude");
 uint64_t blocks=0,scalars=0,computed=0;za_native_faust_calls(p.get(),&blocks,&scalars,&computed);require(!scalars,"FAUST used scalar calls");
 auto value=[&](const char* name){double v=0;int spans=0;size_t capacity=0;require(za_native_gfx_snapshot(p.get(),name,&v,&spans,&capacity),"Missing native observation");return v;};
 const double model=value("p7_model");require(model==requestedModel,"Wrong render model exercised");
 if(!blocks && model!=0)require(value("p7_running")!=0,"Physical renderer never ran");
 // Exercise real editor drawing and restored controls, outside timed processing.
 juce::MemoryBlock saved;p->getStateInformation(saved);p->setStateInformation(saved.getData(),int(saved.getSize()));
 auto editor=std::unique_ptr<juce::AudioProcessorEditor>(p->createEditor());require(bool(editor),"Missing editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);
 for(int i=0;i<3;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);editor->createComponentSnapshot(editor->getLocalBounds());p->editorBeingDeleted(editor.get());editor.reset();
 for(double sr:{44100.,96000.}){p->releaseResources();p->setRateAndBufferSizeDetails(sr,1024);p->prepareToPlay(sr,1024);audio.setSize(std::max(2,p->getTotalNumOutputChannels()),1024);audio.clear();midi.clear();p->processBlock(audio,midi);}
 za_native_gfx_faults(p.get(),&dsp,&gfx);require(!dsp&&!gfx,"Editor/reset memory fault");
 std::ofstream report(argv[2]);report<<"{\"rate\":"<<rate<<",\"block\":"<<block<<",\"scenario\":"<<scenario<<",\"frames\":"<<frames<<",\"process_seconds\":"<<seconds<<",\"callbacks\":"<<callbacks<<",\"peak\":"<<peak<<",\"energy_l\":"<<energyL<<",\"energy_r\":"<<energyR<<",\"tail_energy\":"<<tail<<",\"max_adjacent_step\":"<<maxStep<<",\"faust_blocks\":"<<blocks<<",\"faust_scalars\":"<<scalars<<",\"faust_frames\":"<<computed<<",\"observed_model\":"<<model<<",\"editor_state_reset_passed\":true}";
 std::cout<<scenario<<" @ "<<rate<<"/"<<block<<": "<<seconds<<" seconds; peak "<<peak<<"; FAUST blocks "<<blocks<<std::endl;
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
