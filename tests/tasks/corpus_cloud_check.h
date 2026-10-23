#pragma once
#include <array>

static juce::RangedAudioParameter* corpusParam(juce::AudioProcessor* p,int id){for(auto* raw:p->getParameters())if(auto* x=dynamic_cast<juce::RangedAudioParameter*>(raw))if(x->paramID=="slider"+juce::String(id))return x;throw std::runtime_error("Missing cloud parameter");}
static void corpusSet(juce::AudioProcessor* p,int id,float value){auto* x=corpusParam(p,id);x->setValueNotifyingHost(x->convertTo0to1(value));}
static double corpusGet(juce::AudioProcessor* p,int id){auto* x=corpusParam(p,id);return x->convertFrom0to1(x->getValue());}

template<class Read> void corpusCloudCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const std::string& path){
 auto check=[](bool ok,const std::string& reason){if(!ok)throw std::runtime_error("Cloud: "+reason);};
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 auto wait=[&](){for(int i=0;i<12000;++i){process();if(i>64 && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && value("rebuild_request")==0 && value("auto_settings_pending")==0)return;std::this_thread::sleep_for(std::chrono::milliseconds(1));}throw std::runtime_error("Cloud state failed to prepare");};
 auto heap=[&](const char* symbol,int n){std::vector<double> x(n);check(za_native_gfx_heap_copy(p,symbol,x.data(),n)==n,"heap snapshot failed");return x;};
 check(corpusGet(p,76)==0 && corpusGet(p,80)==1,"fresh cloud/smear defaults");
 std::vector<std::pair<int,float>> saved{{76,7},{77,4},{78,.72f},{79,.67f},{37,.38f},{38,.91f},{53,.82f},{80,1}};
 for(auto [id,x]:saved)corpusSet(p,id,x);process();juce::MemoryBlock state;p->getStateInformation(state);
 for(auto [id,x]:saved)corpusSet(p,id,0);process();p->setStateInformation(state.getData(),int(state.getSize()));wait();
 for(auto [id,x]:saved)check(std::abs(corpusGet(p,id)-x)<.002,"preset lost slider"+std::to_string(id));
 std::cout<<"CLOUD preset round trip: count/mode/density/timing/level/similarity/width/smear saved and restored\n";
 auto xml=juce::AudioProcessor::getXmlFromBinary(state.getData(),int(state.getSize()));check(bool(xml),"state XML");auto tree=juce::ValueTree::fromXml(*xml);
 for(int id:{76,77,78,79,80}){auto node=tree.getChildWithProperty("id","slider"+juce::String(id));if(node.isValid())tree.removeChild(node,nullptr);}
 auto morphology=tree.getChildWithProperty("id","slider36");morphology.setProperty("value",3,nullptr);juce::MemoryBlock legacy;auto legacyXml=tree.createXml();juce::AudioProcessor::copyXmlToBinary(*legacyXml,legacy);
 p->setStateInformation(legacy.getData(),int(legacy.getSize()));wait();check(corpusGet(p,76)==-1 && corpusGet(p,77)==0 && corpusGet(p,80)==0,"older state did not select compatible defaults");
 audio.clear();midi.clear();midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);auto e=heap("event_base",64);check(e[45]==2 && e[46]==6,"older morphology did not restore its two Mixed children");
 for(int i=0;i<500;++i)process();corpusSet(p,76,0);audio.clear();midi.clear();midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);e=heap("event_base",64);check(e[45]==0,"explicit zero failed to disable old morphology");for(int i=0;i<500;++i)process();
 p->setStateInformation(state.getData(),int(state.getSize()));wait();
 std::cout<<"CLOUD old-state defaults: legacy morphology preserved, Classic restored, explicit Off wins\n";
 for(auto [id,x]:std::vector<std::pair<int,float>>{{12,1},{25,1800},{26,15},{27,100},{28,.8f},{29,300},{45,0},{76,4},{37,.38f},{38,.85f},{53,.45f},{78,.6f},{79,.2f}})corpusSet(p,id,x);
 std::ofstream csv(path);csv<<"mode,grains,children,observed_kinds,max_voices,peak\n";
 for(int mode=0;mode<7;++mode){corpusSet(p,77,float(mode));double peak=0;int maxVoices=0;
  for(int b=0;b<430;++b){audio.clear();midi.clear();if(!b)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);peak=std::max(peak,double(audio.getMagnitude(0,audio.getNumSamples())));maxVoices=std::max(maxVoices,int(value("voice_list_n")));}
  int n=int(value("record_n"));auto record=heap("record_base",n*20);std::set<int> kinds;for(int i=0;i<n;++i)if(record[i*20+5]>0)kinds.insert(int(record[i*20+5]));
  std::set<int> expected;if(mode==6)expected={3,1,6,2};else expected.insert(mode==0?3:mode==1?1:mode==2?2:mode==3?4:mode==4?5:6);
  check(kinds==expected && peak>1e-8 && maxVoices<=64,"mode did not emit its expected independent children");csv<<mode<<','<<n<<",4,"<<kinds.size()<<','<<maxVoices<<','<<peak<<'\n';
  std::cout<<"CLOUD mode="<<mode<<" grains="<<n<<" independent_kinds="<<kinds.size()<<" max_voices="<<maxVoices<<std::endl;
 }
 // Same recorded selection plan, real mono fold, four amounts. Only the render
 // changes; shared L/R heads cannot create stereo from this truly mono mix.
 corpusSet(p,76,0);corpusSet(p,69,1);wait();corpusSet(p,80,1);std::vector<double> plan;std::vector<float> comparison;
 std::vector<float> lastRender;
 for(float amount:{0.f,.15f,.5f,1.f}){corpusSet(p,45,amount);double peak=0,power=0,monoError=0,totalTime=0;int active=0;lastRender.clear();
  for(int b=0;b<430;++b){audio.clear();midi.clear();if(!b)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);auto start=std::chrono::steady_clock::now();p->processBlock(audio,midi);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(value("voice_list_n")>0){totalTime+=elapsed;++active;}
   for(int i=0;i<audio.getNumSamples();++i){double l=audio.getSample(0,i),r=audio.getSample(1,i);check(std::isfinite(l)&&std::isfinite(r),"nonfinite grain smear");monoError=std::max(monoError,std::abs(l-r));peak=std::max(peak,std::abs(l));power+=l*l;comparison.push_back(float(l));comparison.push_back(float(r));lastRender.push_back(float(l));lastRender.push_back(float(r));}}
  check(monoError==0 && peak>1e-8,"smear created stereo from mono");int n=int(value("record_n"));auto record=heap("record_base",n*20);if(plan.empty())plan=record;else{check(record.size()==plan.size(),"smear changed the selection count");for(int i=0;i<n;++i)for(int field:{0,2,4,5,6,7})check(record[i*20+field]==plan[i*20+field],"smear changed grain decisions");}
  std::cout<<"SMEAR amount="<<amount<<" peak="<<peak<<" rms="<<std::sqrt(power/(430*audio.getNumSamples()))<<" mono_error="<<monoError<<" active_ms="<<1000*totalTime/std::max(1,active)<<std::endl;
  comparison.resize(comparison.size()+size_t(4800)*2,0);
 }
 corpusSet(p,76,12);corpusSet(p,77,6);corpusSet(p,78,1);corpusSet(p,53,1);
 int serial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,serial*1048576+4);size_t sampleAt=0;double repeatError=0;constexpr size_t padding=144*2;
 for(int b=0;b<432;++b){process();for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c){double expected=sampleAt<padding || sampleAt-padding>=lastRender.size()?0:lastRender[sampleAt-padding];repeatError=std::max(repeatError,std::abs(double(audio.getSample(c,i))-expected));++sampleAt;}}
 check(repeatError<1e-6,"cloud/smear REPEAT changed the recorded waveform: "+std::to_string(repeatError));std::cout<<"CLOUD/SMEAR changed live cloud controls: REPEAT waveform error="<<repeatError<<std::endl;
 juce::File file(juce::String::fromUTF8((path+".smear-comparison.wav").c_str()));auto stream=file.createOutputStream();check(bool(stream),"comparison WAV open");stream->setPosition(0);stream->truncate();juce::WavAudioFormat format;
 std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),48000,2,24,{},0));check(bool(writer),"comparison WAV writer");juce::AudioBuffer<float> rendered(2,int(comparison.size()/2));for(int i=0;i<rendered.getNumSamples();++i){rendered.setSample(0,i,comparison[size_t(i)*2]);rendered.setSample(1,i,comparison[size_t(i)*2+1]);}check(writer->writeFromAudioSampleBuffer(rendered,0,rendered.getNumSamples()),"comparison WAV write");writer.reset();
 corpusSet(p,45,0);corpusSet(p,69,0);wait();corpusSet(p,76,4);corpusSet(p,77,4);corpusSet(p,80,1);
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx&&value("output_faults")==0,"runtime fault");std::cout<<"CLOUD/SMEAR actual processor modes, presets, compatibility, mono and selection isolation passed; faults="<<dsp<<','<<gfx<<std::endl;
}

template<class Read> void corpusCloudCapture(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder){
 folder.createDirectory();corpusSet(p,55,2);corpusSet(p,76,4);corpusSet(p,77,4);corpusSet(p,80,1);corpusSet(p,45,.8f);
 for(int i=0;i<16;++i){audio.clear();midi.clear();p->processBlock(audio,midi);}
 std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());if(!editor)throw std::runtime_error("No cloud editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1180,1180);
 auto find=[&](auto&& self,juce::Component& c)->juce::Component*{if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* ch:c.getChildren())if(auto* r=self(self,*ch))return r;return nullptr;};auto* view=find(find,*editor);if(!view)throw std::runtime_error("No cloud canvas");
 auto event=[&](juce::Point<float> point){auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(),1,0,0,0,0,view,view,now,point,now,1,false);};
 auto frames=[&](){for(int i=0;i<180;++i){audio.clear();midi.clear();p->processBlock(audio,midi);juce::MessageManager::getInstance()->runDispatchLoopUntil(3);}};
 auto wheel=[&](float delta){juce::MouseWheelDetails w{};w.deltaY=delta;view->mouseWheelMove(event({500,400}),w);frames();};
 auto save=[&](const char* name){view->mouseMove(event({4,4}));frames();auto stream=folder.getChildFile(name).createOutputStream();if(!stream)throw std::runtime_error("Cloud screenshot open");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;if(!png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream))throw std::runtime_error("Cloud screenshot write");};
 frames();wheel(-6);save("Corpus-Child-Cloud.png");
 // Grounded in the captured 1180-wide native layout: exercise the real count
 // and mode click handlers, then observe their normal host parameters.
 auto click=[&](juce::Point<float> root){auto local=view->getLocalPoint(editor.get(),root);auto now=juce::Time::getCurrentTime();juce::MouseEvent down(juce::Desktop::getInstance().getMainMouseSource(),local,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),1,0,0,0,0,view,view,now,local,now,1,false);view->mouseMove(down);view->mouseDown(down);frames();view->mouseUp(down);frames();};
 click({1118,325});click({100,410});
 if(corpusGet(p,76)!=5 || corpusGet(p,77)!=0)throw std::runtime_error("Native child UI click did not update host parameters");
 std::cout<<"CLOUD native count/mode clicks update saved host parameters\n";corpusSet(p,76,4);corpusSet(p,77,4);frames();
 wheel(-23);save("Corpus-Grain-Smear.png");
 editor->setSize(1120,1180);wheel(100);wheel(-6);save("Corpus-Child-Cloud-Compact.png");
 p->editorBeingDeleted(editor.get());editor.reset();std::cout<<"CAPTURE cloud wide/compact and grain smear native UI\n";
}
