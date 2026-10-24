#pragma once
template<class Read> void corpusLiveCapture(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder){
 auto check=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};folder.createDirectory();
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 auto frames=[&](int n=100){for(int i=0;i<n;++i){process();juce::MessageManager::getInstance()->runDispatchLoopUntil(3);}};
 auto wait=[&](){for(int i=0;i<180000;++i){process();if(i%32==0)juce::MessageManager::getInstance()->runDispatchLoopUntil(1);if(i>100 && value("ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && value("auto_settings_pending")==0 && value("cg_stage")==0)return;std::this_thread::sleep_for(std::chrono::milliseconds(1));}throw std::runtime_error("UI prepare timeout");};
 corpusSet(p,242,2);corpusSet(p,55,6);corpusSet(p,76,0);corpusSet(p,25,5000);wait();
 std::vector<double> range(162);check(za_native_gfx_heap_copy(p,"desc_value_range",range.data(),162)==162,"UI range copy");
 auto lane=[&](int desc){for(int i=0;i<12;++i)corpusSet(p,86+i*13,0);corpusSet(p,86,float(3072+desc));corpusSet(p,87,1);double lo=range[desc*3],hi=range[desc*3+1];for(int j=0;j<11;++j){int packed=0;for(int k=0;k<3;++k)if(j*3+k<32){double u=double(j*3+k)/31;int q=int(std::round(std::clamp(lo+(hi-lo)*(.15+.7*u),0.,1.)*126));packed|=q<<(k*7);}corpusSet(p,88+j,float(packed));}process();};
 lane(16);std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());check(bool(editor),"No UI editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1180,1180);
 auto find=[&](auto&& self,juce::Component& c)->juce::Component*{if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* ch:c.getChildren())if(auto* r=self(self,*ch))return r;return nullptr;};auto* view=find(find,*editor);check(view!=nullptr,"No UI canvas");
 auto event=[&](juce::Point<float> point,bool down=false){auto local=view->getLocalPoint(editor.get(),point);auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),local,juce::ModifierKeys(down?juce::ModifierKeys::leftButtonModifier:0),1,0,0,0,0,view,view,now,local,now,1,false);};
 auto save=[&](const char* name){view->mouseMove(event({4,4}));frames();auto stream=folder.getChildFile(name).createOutputStream();check(bool(stream),"UI image open");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;check(png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream),"UI image save");};
 auto note=[&](){audio.clear();midi.clear();midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);p->processBlock(audio,midi);frames(450);};
 frames();note();save("Corpus-Trajectory-Corpus-Range-Overview.png");
 check(value("ui_tr_hi")-value("ui_tr_lo")<.8 && value("ui_tr_hi")>=range[49] && value("ui_tr_lo")<=range[48],"Actual UI range did not follow centroid evidence");
 std::cout<<"UI centroid scale="<<value("ui_tr_lo")<<".."<<value("ui_tr_hi")<<" actual="<<range[48]<<".."<<range[49]<<std::endl;
 // The first captured native image grounds the Rise button at this position.
 view->mouseMove(event({420,696}));view->mouseDown(event({420,696},true));frames(18);view->mouseUp(event({420,696}));frames(320);
 std::vector<double> edited(40);check(za_native_gfx_heap_copy(p,"traj_base",edited.data(),40)==40,"UI edited shape copy");check(std::abs(edited[8]-value("ui_tr_lo"))<.01 && std::abs(edited[39]-value("ui_tr_hi"))<.01,"UI Rise did not use corpus range");
 juce::MouseWheelDetails wheel{};wheel.deltaY=-4;view->mouseWheelMove(event({850,700}),wheel);frames();save("Corpus-Trajectory-Corpus-Range.png");
 std::cout<<"UI real Rise button uses observed range; packed endpoints="<<edited[8]<<".."<<edited[39]<<std::endl;
 frames(600);lane(53);note();save("Corpus-Trajectory-Length.png");
 corpusSet(p,246,1);corpusSet(p,55,1);frames();save("Corpus-Length-Map.png");
 frames(600);corpusSet(p,244,1.2f);wait();save("Corpus-Length-Limit-ms.png");
 editor->setSize(1120,920);save("Corpus-Compact-Header.png");
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx,"UI faults");
 p->editorBeingDeleted(editor.get());editor.reset();std::cout<<"UI corpus scale, Length map/trajectory, ms limit and compact header captured; faults0/0\n";
}
