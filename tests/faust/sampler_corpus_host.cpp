#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor*,const char* const*,int);
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
extern "C" void za_native_faust_calls(juce::AudioProcessor*,uint64_t*,uint64_t*,uint64_t*);
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv)try{
 if(argc!=6 && argc!=7)throw std::runtime_error("key authorized-recording report sample-rate buffer-size");
 const bool clip=argc==7 && std::string(argv[6])=="--clip";
 const bool routes=argc==7 && std::string(argv[6])=="--character-routes";
 const bool character=routes || (argc==7 && std::string(argv[6])=="--character");
 if(argc==7 && !clip && !character)throw std::runtime_error("Unknown test mode");
 const std::string key=argv[1];if(key!="Corpus"&&key!="Sample")throw std::runtime_error("Unsupported key");
 const juce::String allowed="C:\\Users\\Louis\\OneDrive\\Sound Effects\\NSFW\\Mouth Noises\\Files to Cut\\Slapping dirty slut BJ.flac";
 if(juce::String::fromUTF8(argv[2])!=allowed)throw std::runtime_error("Only the authorized recording may load");
 juce::ScopedJuceInitialiser_GUI gui;auto p=std::unique_ptr<juce::AudioProcessor>(createPluginFilter());
 const int rate=std::stoi(argv[4]),block=std::stoi(argv[5]);p->setNonRealtime(true);p->setRateAndBufferSizeDetails(rate,block);p->prepareToPlay(rate,block);
 auto value=[&](const char* name){double v=0;int spans=0;size_t capacity=0;za_native_gfx_snapshot(p.get(),name,&v,&spans,&capacity);return v;};
 auto control=[&](int slider,float plain){auto& params=p->getParameters();if(slider>int(params.size()))throw std::runtime_error("Missing slider");auto* x=dynamic_cast<juce::RangedAudioParameter*>(params[slider-1]);if(!x)throw std::runtime_error("Missing parameter range");x->setValueNotifyingHost(x->convertTo0to1(plain));};
 const char* paths[]={argv[2]};za_sample_gfx_load_bank(p.get(),paths,1);
 juce::AudioBuffer<float> audio(std::max(2,p->getTotalNumOutputChannels()),block);juce::MidiBuffer midi;
 auto begin=Clock::now();bool ready=false;
 while(std::chrono::duration<double>(Clock::now()-begin).count()<180){audio.clear();midi.clear();p->processBlock(audio,midi);
  ready=key=="Corpus"?(value("ready")>0&&value("xc_ready")>0&&value("h_ready")>0&&value("pe_ready")>0&&value("cg_stage")==0&&value("cg_retiring")==0):(value("bank_loaded")>0 && value("rescan_pending")==0 && value("pool_state")!=1 && value("pool_state")!=2);
  if(ready)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
 if(!ready)throw std::runtime_error("Loaded-bank preparation timeout");
 std::cout<<key<<" loaded; grains="<<value("grain_count")<<" bank="<<value("bank_loaded")<<std::endl;
 std::ofstream dump(std::string(argv[3])+".pcm.bin",std::ios::binary);
 std::ofstream state(std::string(argv[3])+".state.csv");state<<"scenario,block,voices,output_smooth,td_mix,td_fast,td_slow,filter_silent,output_peak,limited_samples\n";
 uint64_t initialFaustBlocks=0,initialFaustScalars=0,initialFaustFrames=0;za_native_faust_calls(p.get(),&initialFaustBlocks,&initialFaustScalars,&initialFaustFrames);
 double dsp=0,maxPreLimitPeak=0;float peak=0;uint64_t samples=0;int callbacks=0,characterBlocks=0,harmonicBlocks=0;bool heard=false;
 for(int scenario=0;scenario<4;++scenario){
  if(key=="Corpus"){control(45,scenario==0?0:scenario==1?0.35f:1);control(7,scenario==3?(clip?12:-6):0);}
  else {control(17,scenario%2?0:1);control(38,scenario==0?20:90);control(39,scenario==0?20000:5500);control(43,scenario<2?0:scenario==2?1:4);control(44,scenario<2?0:scenario==2?1:4);control(32,scenario==0?0:9);control(33,scenario==0?0:-6);control(34,scenario==0?0:3);control(45,scenario==3?2:0.707f);control(46,scenario==3?2:0.707f);}
  if(character && key=="Sample"){control(62,scenario==0?0:scenario==1?121:242);int drive=scenario==0?0:scenario==3?48:24;control(31,drive*(1+49+49*49));control(58,drive*(1+49));control(55,scenario<2?0:scenario==2?0.35f:0.85f);}
  const int blocks=(rate*8+block-1)/block;
  for(int n=0;n<blocks;++n){audio.clear();midi.clear();
   if(n==0||n==rate*4/block){midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);midi.addEvent(juce::MidiMessage::noteOn(1,64,juce::uint8(80)),std::min(17,block-1));}
   if(n==rate/block||n==rate*5/block){midi.addEvent(juce::MidiMessage::noteOff(1,60),std::min(11,block-1));midi.addEvent(juce::MidiMessage::noteOff(1,64),std::min(29,block-1));}
   if(n==rate*2/block){if(key=="Corpus")control(45,0);else {control(38,20);control(39,20000);control(32,0);control(33,0);control(34,0);}}
   if(n==rate*3/block){if(key=="Corpus")control(45,0.8f);else{control(38,200);control(39,8000);control(32,-9);}}
   if(n==rate*6/block){if(key=="Corpus")control(45,0.5f);else{control(43,4);control(44,4);control(38,350);}}
   if(character && key=="Sample" && n==rate*2/block){control(31,0);control(58,0);control(55,0);}
   if(character && key=="Sample" && n==rate*3/block){control(31,36*(1+49+49*49));control(58,36*(1+49));control(55,scenario>=2?0.65f:0);}
   if(routes && key=="Sample"){
    if(n==rate/2/block)control(50,0.35f);
    if(n==rate/block)control(50,0);
    if(n==rate*9/2/block)control(52,0.4f);
    if(n==rate*5/block)control(52,0);
    for(int f=0;f<block;++f){double t=double(n*block+f)/rate;audio.setSample(0,f,float(0.03*sin(t*173*6.283185307179586)));audio.setSample(1,f,float(0.02*sin(t*257*6.283185307179586)));}
   }
   auto start=Clock::now();p->processBlock(audio,midi);dsp+=std::chrono::duration<double>(Clock::now()-start).count();++callbacks;maxPreLimitPeak=std::max(maxPreLimitPeak,value("output_peak"));
   characterBlocks+=value("posteq_char_active")>0;harmonicBlocks+=value("output_harmonics_amt")>0.0001;
   for(int ch=0;ch<2;++ch){for(int f=0;f<block;++f)if(!std::isfinite(audio.getSample(ch,f)))throw std::runtime_error("Nonfinite output");dump.write((char*)audio.getReadPointer(ch),block*sizeof(float));}
   samples+=uint64_t(block)*2;peak=std::max(peak,audio.getMagnitude(0,block));heard|=audio.getMagnitude(0,block)>1e-8f;
   if(n%17==0)state<<scenario<<','<<n<<','<<value(key=="Corpus"?"voice_list_n":"active_voice_seen")<<','<<value("output_smooth")<<','<<value("td_mix")<<','<<value("td_fast")<<','<<value("td_slow")<<','<<value("posteq_filter_silent")<<','<<value("output_peak")<<','<<value("limited_samples")<<'\n';
  }
 }
 uint64_t faustBlocks=0,faustScalars=0,faustFrames=0;za_native_faust_calls(p.get(),&faustBlocks,&faustScalars,&faustFrames);faustBlocks-=initialFaustBlocks;faustScalars-=initialFaustScalars;
 const uint64_t fastFrames=faustFrames-initialFaustFrames;
 if(faustScalars)throw std::runtime_error("Full-block candidate unexpectedly used scalar FAUST calls");
 const double limited=value("limited_samples");if(clip && (!(limited>0) || !(maxPreLimitPeak>1.05) || peak>0.980001))throw std::runtime_error("Safety catch audio/counter/pre-limit meter stress failed");
 if(character && (!characterBlocks||!harmonicBlocks))throw std::runtime_error("Nonlinear benchmark effects did not activate");
 if(!heard)throw std::runtime_error("Loaded voices never produced audio");uint32_t d=0,g=0;za_native_gfx_faults(p.get(),&d,&g);if(d||g)throw std::runtime_error("DSP/GFX fault");
 std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());if(!editor)throw std::runtime_error("No editor");editor.reset();p->releaseResources();p->setRateAndBufferSizeDetails(rate==48000?96000:48000,block);p->prepareToPlay(rate==48000?96000:48000,block);audio.clear();midi.clear();p->processBlock(audio,midi);za_native_gfx_faults(p.get(),&d,&g);if(d||g)throw std::runtime_error("Rate reset fault");
 std::ofstream report(argv[3]);report<<"{\"key\":\""<<key<<"\",\"sample_rate\":"<<rate<<",\"block_size\":"<<block<<",\"output_samples\":"<<samples<<",\"character_blocks\":"<<characterBlocks<<",\"harmonic_blocks\":"<<harmonicBlocks<<",\"process_seconds\":"<<dsp<<",\"faust_blocks\":"<<faustBlocks<<",\"faust_scalars\":"<<faustScalars<<",\"character_fast_frames\":"<<fastFrames<<",\"clip_stress\":"<<(clip?"true":"false")<<",\"limited_samples\":"<<limited<<",\"maximum_pre_limit_peak\":"<<maxPreLimitPeak<<",\"callbacks\":"<<callbacks<<",\"peak\":"<<peak<<",\"heard_voices\":true,\"editor_rate_reset\":true,\"faults\":0}";
 std::cout<<key<<" host active-bank PASS cpu="<<dsp<<" peak="<<peak<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
