#pragma once
#include <array>

template<class Read> void corpusPitchCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const std::string& path) {
  std::ofstream csv(path);csv<<"mode,pitch,tame,grains,mean_duration_ms,minimum_gain,output_peak,output_rms,active_callback_mean_ms,maximum_callback_ms,max_voices\n";
  za_test_gfx_button(p,12,1);za_test_gfx_button(p,20,0);int serial=0;
  for(int mode:{0,1})for(int pitch:{-24,-12,0,12,24}) {
    std::vector<double> firstPlan;int reduced=0;
    for(int tame:{0,1}) {
      za_test_gfx_button(p,73,mode);za_test_gfx_button(p,74,tame);za_test_gfx_button(p,8,pitch);
      za_test_gfx_button(p,24,(++serial%16)*1048576+14);
      audio.clear();midi.clear();p->processBlock(audio,midi);
      double peak=0,power=0,activeTime=0,maxTime=0;int activeBlocks=0,maxVoices=0;
      for(int block=0;block<300;++block){audio.clear();midi.clear();if(!block)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);
        if(block==200)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);
        auto start=std::chrono::steady_clock::now();p->processBlock(audio,midi);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        int voices=int(value("voice_list_n"));maxVoices=std::max(maxVoices,voices);maxTime=std::max(maxTime,elapsed);if(voices>0){activeTime+=elapsed;++activeBlocks;}
        for(int c=0;c<2;++c)for(int i=0;i<audio.getNumSamples();++i){double x=audio.getSample(c,i);if(!std::isfinite(x))throw std::runtime_error("Nonfinite pitch/tame audio");peak=std::max(peak,std::abs(x));power+=x*x;}
      }
      int n=int(value("record_n"));if(n<=0 || peak<=1e-8)throw std::runtime_error("Pitch/tame trial produced no recording/audio");
      std::vector<double> record(size_t(n)*20),render(size_t(n)*8),guard(n);
      if(za_native_gfx_heap_copy(p,"record_base",record.data(),int(record.size()))!=record.size() || za_native_gfx_heap_copy(p,"gr_record",render.data(),int(render.size()))!=render.size() || za_native_gfx_heap_copy(p,"gt_record",guard.data(),n)!=n)throw std::runtime_error("Pitch/tame record read failed");
      double duration=0,minGain=1e30;for(int i=0;i<n;++i){duration+=render[i*8+7];minGain=std::min(minGain,record[i*20+3]);if(guard[i]!=tame)throw std::runtime_error("Per-grain amount not captured");
        if(mode==1 && render[i*8+5]!=0 && std::abs(render[i*8+7]-std::ceil(record[i*20+7]))>1)throw std::runtime_error("Coherent grain duration drifted from source timing");}
      if(!tame)firstPlan=record;else {
        if(firstPlan.size()!=record.size())throw std::runtime_error("Taming changed the selected plan length");
        for(int i=0;i<n;++i){for(int field:{0,2,4,6,7})if(std::abs(record[i*20+field]-firstPlan[i*20+field])>1e-6)throw std::runtime_error("Taming changed source choices/timing");
          if(record[i*20+3]>firstPlan[i*20+3]+1e-7)throw std::runtime_error("Taming boosted a grain");reduced+=record[i*20+3]<firstPlan[i*20+3]-1e-7;}
      }
      csv<<mode<<','<<pitch<<','<<tame<<','<<n<<','<<1000*duration/n/48000<<','<<minGain<<','<<peak<<','<<std::sqrt(power/(300*audio.getNumSamples()*2.))<<','<<activeTime/std::max(1,activeBlocks)*1000<<','<<maxTime*1000<<','<<maxVoices<<'\n';
      std::cout<<"PITCH mode="<<mode<<" st="<<pitch<<" tame="<<tame<<" grains="<<n<<" mean_ms="<<1000*duration/n/48000<<" min_gain="<<minGain<<" peak="<<peak<<" callback_ms="<<activeTime/std::max(1,activeBlocks)*1000<<std::endl;
    }
    std::cout<<"TAME same plan, attenuation-only grains="<<reduced<<std::endl;
  }
  for(int mode:{0,1})for(int tame:{0,1}){
    za_test_gfx_button(p,8,24);za_test_gfx_button(p,73,mode);za_test_gfx_button(p,74,tame);int voices=0,active=0;double worst=0,total=0,first=0;
    za_test_gfx_button(p,24,(++serial%16)*1048576+14);audio.clear();midi.clear();p->processBlock(audio,midi);
    for(int block=0;block<300;++block){audio.clear();midi.clear();if(!block)for(int note=48;note<56;++note)midi.addEvent(juce::MidiMessage::noteOn(1,note,juce::uint8(127)),0);
      auto start=std::chrono::steady_clock::now();p->processBlock(audio,midi);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();worst=std::max(worst,elapsed);int n=int(value("voice_list_n"));voices=std::max(voices,n);if(n>0){total+=elapsed;++active;}if(!block)first=elapsed;}
    std::cout<<"STRESS mode="<<mode<<" tame="<<tame<<" max_voices="<<voices<<" mean_active_ms="<<total/std::max(1,active)*1000<<" first_ms="<<first*1000<<" worst_ms="<<worst*1000<<std::endl;
  }
  // End-to-end recorded replay must retain both new render settings even if
  // live pitch, behavior and taming are changed before REPEAT.
  za_test_gfx_button(p,8,24);za_test_gfx_button(p,73,1);za_test_gfx_button(p,74,1);
  za_test_gfx_button(p,24,(++serial%16)*1048576+14);audio.clear();midi.clear();p->processBlock(audio,midi);
  std::vector<float> original;
  for(int block=0;block<300;++block){audio.clear();midi.clear();if(!block)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c)original.push_back(audio.getSample(c,i));}
  if(value("record_done")<=0 || value("record_overflow")!=0)throw std::runtime_error("No completed pitch/tame replay");
  za_test_gfx_button(p,8,-24);za_test_gfx_button(p,73,0);za_test_gfx_button(p,74,0);za_test_gfx_button(p,24,(++serial%16)*1048576+4);
  double replayError=0;size_t at=0;constexpr size_t replayPadding=144*2; // Existing 3 ms stop fade before REPEAT.
  for(int block=0;block<300;++block){audio.clear();midi.clear();p->processBlock(audio,midi);for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c){double expected=at<replayPadding ? 0 : original[at-replayPadding];replayError=std::max(replayError,std::abs(double(audio.getSample(c,i))-expected));++at;}}
  if(replayError>1e-6)throw std::runtime_error("Pitch/tame replay waveform mismatch "+std::to_string(replayError));
  std::cout<<"REPLAY changed live pitch/mode/tame: waveform error="<<replayError<<std::endl;
  uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);if(dsp || gfx || value("output_faults")>0)throw std::runtime_error("Pitch/tame runtime fault");
  std::cout<<"PITCH/TAME runtime faults="<<dsp<<','<<gfx<<std::endl;
  za_test_gfx_button(p,12,0);za_test_gfx_button(p,20,1);za_test_gfx_button(p,8,0);za_test_gfx_button(p,73,0);za_test_gfx_button(p,74,1);
}

template<class Read> void corpusPitchCapture(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder) {
  folder.createDirectory();za_test_gfx_button(p,55,2);za_test_gfx_button(p,8,-24);za_test_gfx_button(p,73,1);za_test_gfx_button(p,74,1);
  for(int i=0;i<4;++i){audio.clear();midi.clear();p->processBlock(audio,midi);}
  std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());if(!editor)throw std::runtime_error("No pitch editor");
  editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1120,960);
  auto find=[&](auto&& self,juce::Component& root)->juce::Component* {if(root.getName()=="Native JSFX graphics" || root.getName()=="EEL JSFX graphics")return &root;for(auto* child:root.getChildren())if(auto* result=self(self,*child))return result;return nullptr;};
  auto* view=find(find,*editor);if(!view)throw std::runtime_error("Missing pitch canvas");
  auto event=[&](juce::Point<float> point){const auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(),1,0,0,0,0,view,view,now,point,now,1,false);};
  auto save=[&](const char* name){for(int i=0;i<3000;++i){audio.clear();midi.clear();p->processBlock(audio,midi);juce::MessageManager::getInstance()->runDispatchLoopUntil(2);if(i>100 && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("rebuild_request")==0)break;if(i==2999)throw std::runtime_error("Pitch UI not ready");}
    view->mouseMove(event({4,4}));for(int i=0;i<10;++i)juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    for(int i=0;i<60;++i){audio.clear();midi.clear();p->processBlock(audio,midi);juce::MessageManager::getInstance()->runDispatchLoopUntil(2);}
    auto stream=folder.getChildFile(name).createOutputStream();if(!stream || !stream->setPosition(0))throw std::runtime_error("Pitch screenshot failed");stream->truncate();juce::PNGImageFormat png;if(!png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream))throw std::runtime_error("Pitch screenshot encoding failed");};
  save("Corpus-Pitch-Coherent.png");juce::MouseWheelDetails wheel{};wheel.deltaY=-15;view->mouseWheelMove(event({500,400}),wheel);save("Corpus-Grain-Tame.png");
  p->editorBeingDeleted(editor.get());editor.reset();uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);if(dsp || gfx)throw std::runtime_error("Pitch UI runtime fault");
  std::cout<<"CAPTURE pitch mode and grain taming passed"<<std::endl;
}
