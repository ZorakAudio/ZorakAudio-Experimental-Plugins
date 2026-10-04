// SPDX-License-Identifier: Zlib
// Actual production processor, JUCE editor and concurrent audio callback.
#include <juce_audio_utils/juce_audio_utils.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <stdexcept>
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" uint64_t za_native_gfx_frames();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
struct PlayHead : juce::AudioPlayHead {
 std::atomic<int64_t> samples{0};
 juce::Optional<PositionInfo> getPosition() const override {
  PositionInfo p;p.setIsPlaying(true);p.setTimeInSamples(samples.load());
  p.setTimeInSeconds(samples.load()/48000.);p.setPpqPosition(samples.load()/24000.);p.setBpm(120.);return p;
 }
};
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
extern "C" void za_native_gfx_files(juce::AudioProcessor*,uint64_t*,uint64_t*);
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
static void pump(int ms){juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);}
static void waitForCanvas(uint64_t before,bool expectGfx){
 for(int attempt=0;expectGfx && za_native_gfx_frames()==before && attempt<300;++attempt)pump(100);
 require(!expectGfx || za_native_gfx_frames()>before,"GFX worker did not publish a canvas");
 pump(50); // Deliver the publication's message-thread repaint before capture.
}
static juce::Component* nativeCanvas(juce::Component& c){
 if(c.getName()=="Native JSFX graphics")return &c;
 for(int i=0;i<c.getNumChildComponents();++i)if(auto* found=nativeCanvas(*c.getChildComponent(i)))return found;
 return nullptr;
}
static void save(juce::Component& editor,const juce::File& folder,const char* name){
 auto image=editor.createComponentSnapshot(editor.getLocalBounds());
 auto stream=folder.getChildFile(name).createOutputStream();require(stream!=nullptr,"screenshot open failed");
 stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;
 require(png.writeImageToStream(image,*stream),"screenshot write failed");
}
int main(int argc,char** argv)try{
 juce::ScopedJuceInitialiser_GUI gui;require(argc>=2,"Pass output directory");
 juce::File folder(argv[1]);folder.createDirectory();
 const bool expectGfx=argc<3 || std::string(argv[2])!="no-gfx";
 const bool dropWave=argc>3 && std::string(argv[3])=="drop";
 const auto wave=folder.getChildFile("kick.wav");
 if(dropWave){
  auto output=wave.createOutputStream();require(output!=nullptr,"WAV open failed");
  juce::WavAudioFormat format;std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(output.get(),48000,2,24,{},0));require(writer!=nullptr,"WAV writer failed");output.release();
  juce::AudioBuffer<float> sample(2,12000);
  for(int i=0;i<12000;++i)for(int c=0;c<2;++c)sample.setSample(c,i,float(.6*std::exp(-i/1800.)*std::sin(i*2*juce::MathConstants<double>::pi*90/48000)));
  require(writer->writeFromAudioSampleBuffer(sample,0,sample.getNumSamples()),"WAV encode failed");
 }

 std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
 processor->prepareToPlay(48000,256);
 double initialTempo=0;int spans=0;size_t capacity=0;
 if(za_native_gfx_snapshot(processor.get(),"tempo",&initialTempo,&spans,&capacity))require(initialTempo>0,"No fallback tempo");
 PlayHead playHead;processor->setPlayHead(&playHead);
 const int channels=std::max(2,std::max(processor->getTotalNumInputChannels(),processor->getTotalNumOutputChannels()));
 std::atomic<bool> stop{false};std::atomic<uint64_t> blocks{0},nonfinite{0};
 std::thread audio([&]{
  juce::AudioBuffer<float> buffer(channels,256);juce::MidiBuffer midi;double phase=0;
  while(!stop){
   for(int i=0;i<256;++i){float v=.08f*std::sin(phase);phase+=2*juce::MathConstants<double>::pi*317/48000;
    for(int c=0;c<channels;++c)buffer.setSample(c,i,v);}
   midi.clear();auto n=blocks.load();
   if(n%96==0)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(92)),0);
   if(n%96==72)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);
   processor->processBlock(buffer,midi);
   for(int c=0;c<buffer.getNumChannels();++c)for(int i=0;i<256;++i)if(!std::isfinite(buffer.getSample(c,i)))++nonfinite;
   playHead.samples+=256;++blocks;std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
 });
 struct Join{std::atomic<bool>& stop;std::thread& thread;~Join(){stop=true;if(thread.joinable())thread.join();}} join{stop,audio};
 uint64_t firstFrames=za_native_gfx_frames();
 for(int lifetime=0;lifetime<2;++lifetime){
  auto hiddenFrames=za_native_gfx_frames();
  std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditor());require(editor!=nullptr,"No editor");
  pump(100);require(za_native_gfx_frames()==hiddenFrames,"Hidden editor executed graphics before layout");
  editor->addToDesktop(juce::ComponentPeer::windowHasTitleBar);editor->setVisible(true);
  std::cerr<<"OPEN "<<lifetime<<" blocks="<<blocks<<" frames="<<za_native_gfx_frames()<<std::endl;
  auto before=za_native_gfx_frames();pump(700);
  waitForCanvas(before,expectGfx);
  std::cerr<<"PUMPED blocks="<<blocks<<" frames="<<za_native_gfx_frames()<<std::endl;

  double simulationWidth=0,simulationHeight=0;
  if(za_native_gfx_snapshot(processor.get(),"particlesim.w",&simulationWidth,&spans,&capacity) &&
     za_native_gfx_snapshot(processor.get(),"particlesim.h",&simulationHeight,&spans,&capacity))
    require(simulationWidth>1 && simulationHeight>1,"Particle simulation initialized before canvas layout");
  save(*editor,folder,lifetime?"reopened.png":"open.png");
  if(!lifetime && dropWave){
   auto* target=dynamic_cast<juce::FileDragAndDropTarget*>(editor.get());require(target!=nullptr,"No file drop target");
   auto* canvas=nativeCanvas(*editor);require(canvas!=nullptr,"No native drop canvas");
   const std::string plugin=argc>4?argv[4]:"";
   double gx=0,gy=0;auto value=[&](const char* name){double v=0;require(za_native_gfx_snapshot(processor.get(),name,&v,&spans,&capacity),"Missing drop geometry");return v;};
   const double gw=value("gfx_w"),gh=value("gfx_h");
   gx=.5*gw;gy=.3*gh;
   if(plugin=="joep_saike_bric_a_brac"){
    gx=.08*gw;gy=.65*gh;
   }else if(plugin=="joep_saike_partials"){gx=1.5*value("radius");gy=4.1*value("radius");}
   else if(plugin=="joep_saikedrums"){gx=.2*gw;gy=.1*gh;}
   else if(plugin=="joep_saike_protosynth"){gx=.155*gw;gy=.056*gh;}
   else if(plugin=="joep_saike_yutani"){
    gx=.5*(value("osc1_panel._xmin")+value("osc1_panel._xmax"));
    gy=.5*(value("osc1_panel._ymin")+value("osc1_panel._ymax"));
   }
   std::vector<juce::Point<double>> dropPoints{{gx/gw,gy/gh}};
   // Try the visible controls through the actual OS-drop entry point. The
   // source decides which pad accepts the drop; no guest heap is patched.
   for(double py:{.1,.15,.2,.3,.45,.6,.75,.85})for(double px:{.02,.04,.06,.1,.16,.25,.4,.6,.8})dropPoints.push_back({px,py});
   uint64_t dropOpens=0,dropValues=0;
   for(auto normalized:dropPoints){
    const auto point=editor->getLocalPoint(canvas,juce::Point<int>{int(normalized.x*canvas->getWidth()),int(normalized.y*canvas->getHeight())});
    target->filesDropped(juce::StringArray{wave.getFullPathName()},point.x,point.y);
    before=za_native_gfx_frames();pump(150);waitForCanvas(before,expectGfx);
    za_native_gfx_files(processor.get(),&dropOpens,&dropValues);
    if(dropOpens>0 && dropValues>0)break;
   }
   if(!dropOpens || !dropValues)std::cerr<<"DROP counts opens="<<dropOpens<<" values="<<dropValues<<std::endl;
   if(!dropOpens || !dropValues)for(auto name:{"file_dropped","radius","sample_ws","sample_hs","xs","ys","_gfx_w","_gfx_h","gfx_w","gfx_h","mouse_x","mouse_y","advanced_controls","model"}){
    double diagnostic=0;if(za_native_gfx_snapshot(processor.get(),name,&diagnostic,&spans,&capacity))std::cerr<<"DROP "<<name<<"="<<diagnostic<<std::endl;
   }
   require(dropOpens>0 && dropValues>0,"Dropped WAV did not reach native file decoding");
   save(*editor,folder,"loaded.png");
  }
  if(!lifetime){
   // Host parameter listener path, followed by restoration to the source default.
   int changed=0;for(auto* p:processor->getParameters())if(auto* ranged=dynamic_cast<juce::RangedAudioParameter*>(p)){
    float original=ranged->getValue();ranged->beginChangeGesture();ranged->setValueNotifyingHost(std::clamp(original+.08f,0.f,1.f));
    pump(30);ranged->setValueNotifyingHost(original);ranged->endChangeGesture();if(++changed==3)break;
   }
   before=za_native_gfx_frames();editor->setSize(editor->getWidth()+96,editor->getHeight()+64);pump(250);waitForCanvas(before,expectGfx);save(*editor,folder,"resized.png");
  }
  std::cerr<<"CLOSING "<<lifetime<<std::endl;editor->setVisible(false);editor.reset();pump(100);std::cerr<<"CLOSED "<<lifetime<<std::endl;
 }
 stop=true;audio.join();
 uint32_t dspFault=0,gfxFault=0;za_native_gfx_faults(processor.get(),&dspFault,&gfxFault);
 require(!dspFault && !gfxFault,"Guest memory fault");
 juce::MemoryBlock state;processor->getStateInformation(state);require(state.getSize()>0,"No saved state");
 processor->setStateInformation(state.getData(),(int)state.getSize());
 uint64_t opens=0,values=0;za_native_gfx_files(processor.get(),&opens,&values);
 processor->setPlayHead(nullptr);processor->releaseResources();require(blocks>20,"Audio callback made insufficient progress");require(nonfinite==0,"Nonfinite audio");
 std::cout<<"{\"blocks\":"<<blocks<<",\"native_frames\":"<<za_native_gfx_frames()-firstFrames
  <<",\"dsp_fault\":"<<dspFault<<",\"gfx_fault\":"<<gfxFault<<",\"initial_tempo\":"<<initialTempo<<",\"file_opens\":"<<opens<<",\"file_values\":"<<values<<",\"nonfinite\":"<<nonfinite<<",\"state_bytes\":"<<state.getSize()<<"}\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
