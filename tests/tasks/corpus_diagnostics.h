// Optional exact-recording trigger comparison and native editor captures.
#pragma once
#include <set>
#include <cstring>
#include <cmath>
extern "C" void za_test_gfx_button(juce::AudioProcessor*,int,double);

template<class Read> void corpusCompare(juce::AudioProcessor* p,
    juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,
    const std::string& report) {
  std::ofstream csv(report);csv<<"mode,trial,seed,voices,first_grain,audio_hash,grain_hash,peak,rms,correlation_to_first\n";
  int command=1;
  for(int mode=0;mode<4;++mode) {
    const char* label=mode==0 ? "MIDI127" : mode==1 ? "MIDI64" : mode==2 ? "Generate" : "MIDI127_VelocityLink0";
    za_test_gfx_button(p,9,mode==3 ? 0 : 1);
    std::set<uint64_t> audioHashes,grainHashes;std::set<double> seeds,firstGrains;
    std::vector<float> reference;
    double minCorrelation=1;
    for(int trial=0;trial<12;++trial) {
      std::vector<float> rendered;double seed=0,peak=0,power=0;uint64_t hash=1469598103934665603ULL;
      if(mode==2)za_test_gfx_button(p,24,(++command%16)*1048576+3);
      for(int block=0;block<300;++block) {
        audio.clear();midi.clear();
        if(block==0 && mode!=2)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(mode==1?64:127)),0);
        if(block==200 && mode!=2)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);
        p->processBlock(audio,midi);
        if(block==0){std::vector<double> event(64);za_native_gfx_heap_copy(p,"event_base",event.data(),64);seed=event[17];}
        for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c){float sample=audio.getSample(c,i);
          if(!std::isfinite(sample))throw std::runtime_error("Nonfinite comparison audio");
          uint32_t bits=0;std::memcpy(&bits,&sample,sizeof(bits));hash^=bits;hash*=1099511628211ULL;
          rendered.push_back(sample);peak=std::max(peak,double(std::abs(sample)));power+=sample*sample;}
      }
      const int count=int(value("record_n"));std::vector<double> record(size_t(count)*20);
      if(count<=0 || za_native_gfx_heap_copy(p,"record_base",record.data(),count*20)!=count*20)throw std::runtime_error("No trigger recording");
      uint64_t grainHash=1469598103934665603ULL;
      // Include selected excerpts and offsets, independent of absolute event clock.
      for(int i=0;i<count;++i)for(int field:{0,2,6,7,8}){uint64_t bits=0;double x=record[i*20+field];std::memcpy(&bits,&x,sizeof(bits));grainHash^=bits;grainHash*=1099511628211ULL;}
      if(reference.empty())reference=rendered;
      double dot=0,p0=0,p1=0;for(size_t i=0;i<rendered.size();++i){dot+=reference[i]*rendered[i];p0+=reference[i]*reference[i];p1+=rendered[i]*rendered[i];}
      double correlation=dot/std::sqrt(std::max(1e-30,p0*p1));minCorrelation=std::min(minCorrelation,correlation);
      csv<<label<<','<<trial<<','<<seed<<','<<count<<','<<record[0]<<','<<hash<<','<<grainHash<<','<<peak<<','<<std::sqrt(power/rendered.size())<<','<<correlation<<'\n';
      audioHashes.insert(hash);grainHashes.insert(grainHash);seeds.insert(seed);firstGrains.insert(record[0]);
      if(peak<=1e-8)throw std::runtime_error("Silent trigger comparison");
    }
    std::cout<<"COMPARE "<<label<<" unique_audio="<<audioHashes.size()<<" unique_excerpts="<<grainHashes.size()<<" unique_seeds="<<seeds.size()<<" first_grains="<<firstGrains.size()<<" minimum_correlation="<<minCorrelation<<std::endl;
  }
  za_test_gfx_button(p,9,1);
}

template<class Read> void corpusCapture(juce::AudioProcessor* p,
    juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,
    const juce::File& folder) {
  folder.createDirectory();za_test_gfx_button(p,55,2);
  std::cout<<"CAPTURE before editor epoch="<<value("index_epoch")<<" mode="<<value("source_mono")<<" ready="<<value("ready")<<" pending="<<value("auto_settings_pending")<<" rebuild="<<value("rebuild_request")<<std::endl;
  for(int i=0;i<4;++i){audio.clear();midi.clear();p->processBlock(audio,midi);}
  std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());
  if(!editor)throw std::runtime_error("Missing Corpus editor");
  editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1100,1040);
  for(int i=0;i<20;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
  auto canvas=[&](auto&& self,juce::Component& root)->juce::Component* {
    if(root.getName()=="Native JSFX graphics" || root.getName()=="EEL JSFX graphics")return &root;
    for(auto* child:root.getChildren())if(auto* result=self(self,*child))return result;return nullptr;
  };
  auto* view=canvas(canvas,*editor);if(!view)throw std::runtime_error("Missing native graphics canvas");
  const auto now=juce::Time::getCurrentTime();const juce::Point<float> point(500,400);
  juce::MouseEvent event(juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(),1,0,0,0,0,view,view,now,point,now,1,false);
  juce::MouseWheelDetails wheel;wheel.deltaY=-5;view->mouseWheelMove(event,wheel);
  auto capture=[&](const char* name){
    // Real hosts keep processing while graphics runs. Wait for any input-mode
    // rebuild already queued by the previous check, then capture without hover.
    for(int i=0;i<2000;++i){audio.clear();midi.clear();p->processBlock(audio,midi);juce::MessageManager::getInstance()->runDispatchLoopUntil(2);
      if(i>=100 && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("auto_settings_pending")==0 && value("rebuild_request")==0)break;
      if(i==1999){for(const char* name:{"clock","index_epoch","ready","build_phase","cg_stage","cg_arena","cg_t0","cg_t1","cg_t4","cg_completed","cg_disabled","cg_retiring","cg_failures","cg_last_failure","source_mono","auto_settings_pending","auto_hier_pending","rebuild_request","pending_adoption","pool_state","generation","auto_config_key","auto_last_config","h_sensitive","h_detail","slider69","slider60","slider63"})std::cerr<<name<<'='<<value(name)<<' ';std::cerr<<std::endl;throw std::runtime_error("Editor did not return to ready");}}
    auto off=point;off.x=5;off.y=5;const auto time=juce::Time::getCurrentTime();
    juce::MouseEvent leave(juce::Desktop::getInstance().getMainMouseSource(),off,juce::ModifierKeys(),1,0,0,0,0,view,view,time,off,time,1,false);view->mouseMove(leave);
    for(int i=0;i<5;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    std::cout<<"CAPTURE "<<name<<" epoch="<<value("index_epoch")<<" ready="<<value("ready")<<" graphs="<<value("cg_completed")<<" mode="<<value("source_mono")<<std::endl;
    auto stream=folder.getChildFile(name).createOutputStream();if(!stream || !stream->setPosition(0))throw std::runtime_error("Screenshot output unavailable");
    stream->truncate();juce::PNGImageFormat png;if(!png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream))throw std::runtime_error("Screenshot encoding failed");
  };
  capture("Corpus-ADSR-Linear.png");
  for(auto item:std::vector<std::pair<int,double>>{{25,1800},{26,120},{27,240},{28,.55},{29,650},{70,.4},{71,-.35},{72,-.6}})za_test_gfx_button(p,item.first,item.second);
  capture("Corpus-ADSR-Curved.png");
  editor->setSize(1480,960);capture("Corpus-ADSR-Wide.png");
  p->editorBeingDeleted(editor.get());editor.reset();
  uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);if(dsp || gfx)throw std::runtime_error("Runtime fault during editor capture");
  std::cout<<"CAPTURE native Corpus ADSR UI: Linear, Curved and Wide; runtime faults="<<dsp<<','<<gfx<<std::endl;
}

template<class Read> void corpusMono(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value) {
  for(int mode:{1,0}) {
    const auto prior=value("index_epoch");za_test_gfx_button(p,69,mode);bool ready=false;
    for(int tick=0;tick<20000;++tick){audio.clear();midi.clear();p->processBlock(audio,midi);
      ready=value("index_epoch")>prior && value("source_mono")==mode && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && value("pg_load_pending")==0;
      if(ready)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    if(!ready)throw std::runtime_error("Source mode did not rebuild");
    std::vector<double> source(8),grains(size_t(int(value("grain_count")))*64);za_native_gfx_heap_copy(p,"source_base",source.data(),8);za_native_gfx_heap_copy(p,"grain_base",grains.data(),int(grains.size()));
    if(mode==1)for(size_t i=0;i<grains.size();i+=64)if(grains[i+5]!=1 || grains[i+47]>1e-12)throw std::runtime_error("Mono model retained stereo metadata/width");
    if(value("pe_read_errors")>0 || value("invalid_audio")>0)throw std::runtime_error("Source-mode rebuild read error");
    std::cout<<"MONO model mode="<<mode<<" grains="<<value("grain_count")<<" graph_completed="<<value("cg_completed")<<" failures="<<value("cg_failures")<<std::endl;
  }
}
