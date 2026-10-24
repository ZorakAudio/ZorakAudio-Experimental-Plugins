#pragma once
#include "../../src/JsfxSliderDeclarations.h"

template<class Read> void corpusTrajectoryCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const std::string& path){
 auto check=[](bool ok,const std::string& reason){if(!ok)throw std::runtime_error("Trajectories: "+reason);};
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 auto messages=[&](){juce::MessageManager::getInstance()->runDispatchLoopUntil(2);};
 auto wait=[&](){for(int i=0;i<16000;++i){process();if(i%32==0)messages();if(i>64 && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && value("rebuild_request")==0 && value("auto_settings_pending")==0 && value("cg_stage")==0)return;std::this_thread::sleep_for(std::chrono::milliseconds(1));}throw std::runtime_error("Trajectory state failed to prepare");};
 auto heap=[&](const char* symbol,int n){std::vector<double> x(n);check(za_native_gfx_heap_copy(p,symbol,x.data(),n)==n,"heap snapshot failed");return x;};
 auto command=[&](int op,int arg){int serial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,serial*1048576+arg*16+op);process();messages();process();};
 auto decls=parseJsfxSliderDecls("slider1:0.2<0,1,.001>Existing\nslider99:1<0,3,1>Old bound\nslider100:7<0,10,1>Hundred\nslider241:8<0,10,1>Curve\nslider256:9<0,10,1>Last valid\nslider257:1<0,2,1>Invalid\nslider999:1<0,2,1>Invalid\nslider1000:1<0,2,1>Invalid\n");
 check(decls.size()==5 && decls[0].index0==0 && decls[1].index0==98 && decls[2].index0==99 && decls[3].index0==240 && decls[4].index0==255,"wrapper declaration parser boundaries");
 check(p->getParameters().size()==250,"expected 249 slots + oversampling; actual "+std::to_string(p->getParameters().size()));
 check(corpusGet(p,81)==1 && corpusGet(p,85)==-1,"fresh trajectory defaults");
 std::vector<std::pair<int,float>> saved{{81,1.37f},{82,3},{83,3.42f},{84,.17f},{85,2}};
 for(int l=0;l<12;++l){int base=86+l*13;int meta=3072+(l*4)%53+(l%3)*64+(l%3)*256;saved.emplace_back(base,float(meta));saved.emplace_back(base+1,float(l+1)/12);
  for(int j=0;j<11;++j){int packed=0;for(int k=0;k<3;++k)if(j*3+k<32)packed+=((l*11+(j*3+k)*7)%127)<<(k*7);saved.emplace_back(base+2+j,float(packed));}}
 for(auto [id,x]:saved)corpusSet(p,id,x);process();auto before=heap("traj_base",480);juce::MemoryBlock state;p->getStateInformation(state);
 for(auto [id,x]:saved)corpusSet(p,id,0);process();p->setStateInformation(state.getData(),int(state.getSize()));wait();
 for(auto [id,x]:saved)check(std::abs(corpusGet(p,id)-x)<.002,"host state lost slider"+std::to_string(id));
 auto restored=heap("traj_base",480);for(int j=0;j<480;++j)check(std::abs(before[j]-restored[j])<1e-5,"restored lane table differs");
 command(6,0);wait();auto rebuilt=heap("traj_base",480);for(int j=0;j<480;++j)check(std::abs(before[j]-rebuilt[j])<1e-5,"analysis reload lost a curve");
 std::cout<<"TRAJECTORY twelve-lane host state round trip, float-exact packed shapes and analysis reload preserved\n";
 // Exercise the actual acknowledged shape protocol and DSP -> host publication.
 command(14,30003);for(int i=0;i<8;++i){messages();process();}
 check(int(corpusGet(p,86))==3073,"preset failed to persist lane configuration");
 command(13,60000);for(int i=0;i<16;++i)command(15,255*256+255);command(13,59999);
 for(int i=0;i<12;++i){messages();process();}check(corpusGet(p,88)==2080638 && corpusGet(p,98)==16254,"brush commit not published to saved host state");
 auto painted=heap("traj_base",40);for(int j=8;j<40;++j)check(painted[j]==1,"committed stroke incomplete");
 juce::MemoryBlock paintedState;p->getStateInformation(paintedState);command(14,30000);p->setStateInformation(paintedState.getData(),int(paintedState.getSize()));wait();painted=heap("traj_base",40);for(int j=8;j<40;++j)check(painted[j]==1,"painted curve lost on restore");
 std::cout<<"TRAJECTORY actual brush transaction and preset command persist through host state\n";
 auto xml=juce::AudioProcessor::getXmlFromBinary(state.getData(),int(state.getSize()));check(bool(xml),"state XML");auto tree=juce::ValueTree::fromXml(*xml);
 for(int id=81;id<=241;++id){auto node=tree.getChildWithProperty("id","slider"+juce::String(id));if(node.isValid())tree.removeChild(node,nullptr);}
 juce::MemoryBlock legacy;auto oldXml=tree.createXml();juce::AudioProcessor::copyXmlToBinary(*oldXml,legacy);p->setStateInformation(legacy.getData(),int(legacy.getSize()));wait();restored=heap("traj_base",480);
 for(int l=0;l<12;++l)check(restored[l*40]==0 && restored[l*40+4]==0,"older preset inherited a previous lane");check(corpusGet(p,85)==-1,"older preset inherited Solo");
 std::cout<<"TRAJECTORY old states reset missing new settings without inheriting active lanes\n";
 for(auto [id,x]:std::vector<std::pair<int,float>>{{12,1},{11,12345},{25,2400},{26,20},{27,100},{28,.8f},{29,250},{45,0},{76,0},{56,0},{3,0},{35,0},{34,0},{2,0},{9,0},{46,0},{81,1},{82,0},{83,2},{84,0},{85,-1},{31,.5f},{8,0}})corpusSet(p,id,x);
 auto grains=heap("grain_base",int(value("grain_count"))*64);std::ofstream csv(path);csv<<"amount,target,grains,mean_brightness,peak,average_active_ms\n";std::vector<double> means;std::vector<std::vector<int>> plans;std::vector<float> comparison;
 for(auto [amount,target]:std::vector<std::pair<float,float>>{{0,.1f},{0,.9f},{1,.1f},{1,.9f}}){corpusSet(p,81,amount);corpusSet(p,86,3073);corpusSet(p,87,1);int q=int(std::round(target*126));for(int j=0;j<11;++j)corpusSet(p,88+j,float(q+q*128+(j<10?q*16384:0)));process();double peak=0,cpuMs=0;int active=0;
  for(int b=0;b<540;++b){audio.clear();midi.clear();if(!b)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);auto start=std::chrono::steady_clock::now();p->processBlock(audio,midi);if(value("voice_list_n")>0){cpuMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();++active;}for(int i=0;i<audio.getNumSamples();++i){double l=audio.getSample(0,i),r=audio.getSample(1,i);check(std::isfinite(l)&&std::isfinite(r),"nonfinite audio");peak=std::max(peak,std::max(std::abs(l),std::abs(r)));comparison.push_back(float(l));comparison.push_back(float(r));}}
  int n=int(value("record_n"));auto records=heap("record_base",n*20);double sum=0;std::vector<int> plan;for(int i=0;i<n;++i){int id=int(records[i*20]);check(id>=0 && id<int(value("grain_count")),"invalid selected grain");plan.push_back(id);sum+=grains[id*64+17];}check(n>0 && peak>1e-8,"silent plan");means.push_back(sum/n);plans.push_back(plan);csv<<amount<<','<<target<<','<<n<<','<<sum/n<<','<<peak<<','<<cpuMs/std::max(1,active)<<'\n';std::cout<<"TRAJECTORY amount="<<amount<<" target="<<target<<" mean_brightness="<<sum/n<<" grains="<<n<<" active_ms="<<cpuMs/std::max(1,active)<<std::endl;comparison.resize(comparison.size()+4800*2,0);
 }
 check(plans[0]==plans[1],"zero influence changed selection when only shape changed");check(means[3]>means[2]+.1,"high/low requests didn't meaningfully change selected brightness");
 auto lastBegin=comparison.end()-9600-540*256*2;std::vector<float> lastRender(lastBegin,comparison.end()-9600);
 corpusSet(p,81,2);corpusSet(p,85,11);corpusSet(p,86,3074);for(int j=0;j<11;++j)corpusSet(p,88+j,0);process();
 int replaySerial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,replaySerial*1048576+4);size_t sampleAt=0;double repeatError=0;constexpr size_t padding=144*2;
 for(int b=0;b<542;++b){process();for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c){double expected=sampleAt<padding || sampleAt-padding>=lastRender.size()?0:lastRender[sampleAt-padding];repeatError=std::max(repeatError,std::abs(double(audio.getSample(c,i))-expected));++sampleAt;}}
 check(repeatError<1e-6,"live trajectory edits changed recorded REPEAT waveform");std::cout<<"TRAJECTORY edited shapes/solo/influence: recorded REPEAT waveform error="<<repeatError<<std::endl;
 corpusSet(p,81,1);corpusSet(p,85,-1);const int axes[]{1,2,3,4,5,20,16,18,24,27,51,52};for(int l=0;l<12;++l){corpusSet(p,86+l*13,float(3072+axes[l]));corpusSet(p,87+l*13,.6f);for(int j=0;j<11;++j)corpusSet(p,88+l*13+j,float(j==10?8127:1040319));}process();
 std::vector<double> busyTimes;double busyPeak=0;for(int b=0;b<540;++b){audio.clear();midi.clear();if(!b)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);auto start=std::chrono::steady_clock::now();p->processBlock(audio,midi);if(value("voice_list_n")>0)busyTimes.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());busyPeak=std::max(busyPeak,double(audio.getMagnitude(0,audio.getNumSamples())));}
 check(busyPeak>1e-8 && !busyTimes.empty() && value("record_n")>0,"twelve-lane plan failed");std::sort(busyTimes.begin(),busyTimes.end());std::cout<<"TRAJECTORY twelve active Core/Spectral/Temporal/PE lanes: p95_ms="<<busyTimes[size_t(.95*(busyTimes.size()-1))]<<" max_ms="<<busyTimes.back()<<" peak="<<busyPeak<<std::endl;
 juce::File file(juce::String::fromUTF8((path+".comparison.wav").c_str()));auto stream=file.createOutputStream();check(bool(stream),"comparison WAV open");stream->setPosition(0);stream->truncate();juce::WavAudioFormat format;std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),48000,2,24,{},0));juce::AudioBuffer<float> render(2,int(comparison.size()/2));for(int i=0;i<render.getNumSamples();++i)for(int c=0;c<2;++c)render.setSample(c,i,comparison[size_t(i)*2+c]);check(writer->writeFromAudioSampleBuffer(render,0,render.getNumSamples()),"comparison WAV write");writer.reset();
 corpusSet(p,81,1);command(14,30003);uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx&&value("output_faults")==0,"runtime faults");std::cout<<"TRAJECTORY real-source choices move with target, zero bypass exact, no runtime faults\n";
}

template<class Read> void corpusTrajectoryCapture(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder){
 folder.createDirectory();corpusSet(p,55,6);corpusSet(p,81,1);corpusSet(p,82,0);corpusSet(p,83,2);corpusSet(p,84,0);corpusSet(p,85,-1);corpusSet(p,25,2400);
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 auto command=[&](int op,int arg){int serial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,serial*1048576+arg*16+op);process();};command(14,30003);for(int i=0;i<16;++i)process();
 std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());if(!editor)throw std::runtime_error("No trajectory editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1180,1180);
 auto find=[&](auto&& self,juce::Component& c)->juce::Component*{if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* ch:c.getChildren())if(auto* r=self(self,*ch))return r;return nullptr;};auto* view=find(find,*editor);if(!view)throw std::runtime_error("No trajectory canvas");
 auto event=[&](juce::Point<float> point,bool down=false){auto local=view->getLocalPoint(editor.get(),point);auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),local,juce::ModifierKeys(down?juce::ModifierKeys::leftButtonModifier:0),1,0,0,0,0,view,view,now,local,now,1,false);};
 auto frames=[&](int n=100){for(int i=0;i<n;++i){process();juce::MessageManager::getInstance()->runDispatchLoopUntil(3);}};
 auto save=[&](const char* name){view->mouseMove(event({4,4}));frames();auto stream=folder.getChildFile(name).createOutputStream();if(!stream)throw std::runtime_error("Trajectory screenshot open");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;if(!png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream))throw std::runtime_error("Trajectory screenshot write");};
 frames();audio.clear();midi.clear();midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);frames(450);save("Corpus-Trajectories.png");
 // Coordinates are grounded in the captured 1180-wide native editor. Exercise
 // the real mouse handlers and inspect the resulting DSP table / host state.
 auto check=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
 auto lane=[&](){std::vector<double> x(80);check(za_native_gfx_heap_copy(p,"traj_base",x.data(),80)==80,"UI lane read");return x;};
 auto click=[&](juce::Point<float> point,int settle=90){view->mouseMove(event(point));view->mouseDown(event(point,true));frames(18);view->mouseUp(event(point));frames(settle);};
 auto baseline=lane();click({420,578},320);auto rise=lane();check(rise[8]<.02 && rise[39]>.98,"Native Rise button didn't publish shape");
 click({1100,607},320);auto undone=lane();for(int j=8;j<40;++j)check(std::abs(undone[j]-baseline[j])<.005,"Native Undo didn't restore shape");
 view->mouseMove(event({650,709}));view->mouseDown(event({650,709},true));frames(18);view->mouseDrag(event({650,650},true));frames(24);view->mouseUp(event({650,650}));frames(320);auto points=lane();check(points[21]>baseline[21]+.2,"Native point drag didn't reshape neighboring samples");
 click({1100,607},320);click({1030,607});view->mouseMove(event({470,700}));view->mouseDown(event({470,700},true));frames(18);view->mouseDrag(event({760,660},true));frames(24);view->mouseUp(event({760,660}));frames(320);auto painted=lane();check(painted[25]>.7,"Native freehand stroke didn't commit");save("Corpus-Trajectories-Draw.png");
 click({1100,607},320);click({960,607});click({930,445});check((int(corpusGet(p,86))/1024)%2==0 && lane()[4]==1,"Native mute lost allocation or didn't persist");click({930,445});check((int(corpusGet(p,86))/1024)%2==1,"Native unmute failed");
 click({1010,445});check(corpusGet(p,85)==0,"Native Solo wasn't a saved parameter");click({1010,445});check(corpusGet(p,85)==-1,"Native Solo toggle failed");
 click({1000,486});auto duplicated=lane();check(duplicated[44]==1,"Native Duplicate didn't allocate a lane");for(int j=0;j<40;++j)check(duplicated[j]==duplicated[40+j],"Native Duplicate changed the shape");click({130,530});click({1090,445});check(lane()[44]==0,"Native Remove didn't free the selected slot");
 std::cout<<"TRAJECTORY native UI: Rise, Undo, point drag, Draw, mute, unmute, Solo, Duplicate and Remove passed\n";
 command(14,30001);frames();save("Corpus-Trajectories-Multiple-Lanes.png");
 editor->setSize(1120,1180);frames();save("Corpus-Trajectories-Compact.png");
 p->editorBeingDeleted(editor.get());editor.reset();std::cout<<"CAPTURE trajectory dedicated workspace, multiple lanes and compact layout\n";
}
