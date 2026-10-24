#pragma once
template<class Read> void corpusRepeatCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder){
 auto check=[](bool ok,const std::string& why){if(!ok)throw std::runtime_error("Repeat: "+why);};
 auto process=[&](){p->processBlock(audio,midi);};
 auto idle=[&](int n=2){for(int i=0;i<n;++i){audio.clear();midi.clear();process();}};
 auto heap=[&](const char* name,int count){std::vector<double> x(count);check(za_native_gfx_heap_copy(p,name,x.data(),count)==count,"heap read");return x;};
 auto command=[&](int op,int arg=0){int serial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,serial*1048576+arg*16+op);};
 for(auto [id,x]:std::vector<std::pair<int,float>>{{10,0},{12,1},{11,12345},{25,2200},{26,80},{27,180},{28,.71f},{29,450},{70,-.4f},{71,.3f},{72,-.55f},{73,1},{8,12},{76,4},{77,6},{78,.5f},{79,.2f},{37,.3f},{249,.8f},{45,.2f},{80,1},{49,1}})corpusSet(p,id,x);idle();
 std::vector<float> reference;
 auto take=[&](int kind,int onset=0){std::vector<float> out;
  for(int b=0;b<540;++b){audio.clear();midi.clear();
   if(!b){if(kind==0)command(3);if(kind==1)command(4);if(kind==2)midi.addEvent(juce::MidiMessage::noteOn(1,84,juce::uint8(12)),onset);if(kind==3)corpusDurationPacket(midi,17,.075,1,0,100,onset);}
   if(kind==2&&b==2)midi.addEvent(juce::MidiMessage::noteOff(1,84),5);
   if(kind==3&&b==5)corpusDurationPacket(midi,18,.02,1,0,100,0,6,17);
   if(kind==3&&b==6)corpusDurationPacket(midi,18,0,1,127,127,0,7);
   process();for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c){auto x=audio.getSample(c,i);check(std::isfinite(x),"nonfinite audio");out.push_back(x);}
  }return out;
 };
 auto compare=[&](const std::vector<float>& out,int offset,const char* what){double error=0,peak=0;size_t pad=size_t(offset)*2;for(size_t i=0;i<out.size();++i){double expect=i<pad||i-pad>=reference.size()?0:reference[i-pad];error=std::max(error,std::abs(double(out[i])-expect));peak=std::max(peak,std::abs(double(out[i])));}std::cout<<"REPEAT "<<what<<" error="<<error<<" peak="<<peak<<std::endl;check(error<1e-6&&peak>1e-6,std::string(what)+" waveform mismatch");};
 auto arm=[&](){command(4,1);idle();check(heap("repeat_hold",1)[0]==1,"did not arm");};
 reference=take(0);check(value("record_done")==1&&value("record_overflow")==0&&value("record_n")>0,"generated take incomplete");auto plan=heap("record_base",int(value("record_n"))*20);
 compare(take(1),144,"immediate");arm();idle(100);check(audio.getMagnitude(0,audio.getNumSamples())==0&&heap("repeat_hold",1)[0]==1,"arming sounded or expired");
 compare(take(2,73),73,"MIDI offset73, low velocity and early NoteOff");check(heap("repeat_hold",1)[0]==0,"MIDI did not consume arm");check(heap("record_base",int(value("record_n"))*20)==plan,"MIDI changed captured plan");
 arm();compare(take(3,37),37,"Hub short duration, Continue and End");check(heap("repeat_hold",1)[0]==0&&value("record_overflow")==0,"Hub did not consume arm exactly");
 arm();command(4,1);idle();check(heap("repeat_hold",1)[0]==0,"second click did not cancel");
 arm();command(3);idle(540);check(heap("repeat_hold",1)[0]==0&&value("record_done")==1,"Generate did not cancel arm and finish");reference.clear();
 // Recreate the reference after Generate; deterministic seed and cleared smear
 // make this the same waveform, while the plan is a newly recorded event.
 reference=take(1); // immediate has144 samples of leading pad
 reference.erase(reference.begin(),reference.begin()+288);reference.resize(540*audio.getNumSamples()*2,0);
 arm();double key=value("corpus_key"),epoch=value("index_epoch");
 p->releaseResources();p->prepareToPlay(48000,audio.getNumSamples());
 std::cout<<"REPEAT prepare ready="<<value("ready")<<" pe="<<value("pe_ready")<<" pg="<<value("pg_ready")<<" arm="<<heap("repeat_hold",1)[0]<<" epoch="<<value("index_epoch")<<" key="<<value("corpus_key")<<std::endl;
 check(value("ready")==1&&value("pe_ready")==1&&value("pg_ready")==1&&value("corpus_key")==key&&value("index_epoch")==epoch&&heap("repeat_hold",1)[0]==1,"same-rate render prepare lost armed model");
 for(int i=0;i<100;++i){idle(1);juce::MessageManager::getInstance()->runDispatchLoopUntil(2);}check(value("ready")==1&&heap("repeat_hold",1)[0]==1,"render preroll lost armed take");
 compare(take(2,19),19,"same-instance render prepare");
 check(p->getParameters().size()==250,"parameter count changed");
 // Actual UI clicks, including selected state and cancellation.
 folder.createDirectory();std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());check(bool(editor),"no editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1180,920);
 auto find=[&](auto&& self,juce::Component& c)->juce::Component*{if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* ch:c.getChildren())if(auto* v=self(self,*ch))return v;return nullptr;};auto* view=find(find,*editor);check(view!=nullptr,"no canvas");
 auto frames=[&](int n=100){for(int i=0;i<n;++i){idle(1);juce::MessageManager::getInstance()->runDispatchLoopUntil(3);}};
 auto event=[&](juce::Point<float> root,bool down=false){auto local=view->getLocalPoint(editor.get(),root);auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),local,juce::ModifierKeys(down?juce::ModifierKeys::leftButtonModifier:0),1,0,0,0,0,view,view,now,local,now,1,false);};
 auto click=[&](juce::Point<float> root){view->mouseMove(event(root));view->mouseDown(event(root,true));frames(18);view->mouseUp(event(root));frames();};
 auto save=[&](const char* name){view->mouseMove(event({4,4}));frames();auto stream=folder.getChildFile(name).createOutputStream();check(bool(stream),"PNG open");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;check(png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream),"PNG save");};
 for(int i=0;i<10000;++i){frames(1);if(value("record_done")==1&&value("ready")==1&&value("pe_ready")==1&&value("pg_ready")==1&&value("cg_stage")==0)break;}check(value("record_done")==1,"render re-commit lost completed take");
 frames();save("Corpus-Repeat-Options.png");click({350,154});check(heap("repeat_hold",1)[0]==1,"UI next-note arm failed");save("Corpus-Repeat-Armed.png");click({350,154});check(heap("repeat_hold",1)[0]==0,"UI next-note cancel failed");click({180,154});check(value("replaying")>0,"UI immediate failed");frames(550);
 editor->setSize(1120,920);save("Corpus-Repeat-Compact.png");p->editorBeingDeleted(editor.get());editor.reset();
 // Rate change cannot reuse sample-indexed timings; the first note must fail
 // explicitly instead of substituting a fresh random take.
 arm();p->prepareToPlay(44100,audio.getNumSamples());check(heap("repeat_hold",1)[0]==0,"different rate retained invalid arm");
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx&&value("output_faults")==0,"runtime fault");
 std::cout<<"REPEAT passed: two actions, silent one-use arm, exact MIDI/Hub/prepare audio, cancel/Generate, native UI, rate invalidation; faults0/0\n";
}
