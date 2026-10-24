#pragma once
template<class Read> void corpusPickerCheck(juce::AudioProcessor* p,juce::AudioBuffer<float>& audio,juce::MidiBuffer& midi,Read value,const juce::File& folder){
 auto check=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};folder.createDirectory();
 auto process=[&](){audio.clear();midi.clear();p->processBlock(audio,midi);};
 auto frames=[&](int n=100){for(int i=0;i<n;++i){process();juce::MessageManager::getInstance()->runDispatchLoopUntil(3);}};
 auto lane=[&](){std::vector<double> x(40);check(za_native_gfx_heap_copy(p,"traj_base",x.data(),40)==40,"Picker lane read");return x;};
 auto command=[&](int op,int arg){int serial=(int(value("cmd_seen")/1048576)+1)%16;za_test_gfx_button(p,24,serial*1048576+arg*16+op);process();};
 corpusSet(p,55,6);command(14,30003);frames();check(lane()[1]==1,"Picker baseline Brightness");
 std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());check(bool(editor),"No picker editor");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);editor->setSize(1180,920);
 auto find=[&](auto&& self,juce::Component& c)->juce::Component*{if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* ch:c.getChildren())if(auto* v=self(self,*ch))return v;return nullptr;};auto* view=find(find,*editor);check(view!=nullptr,"No picker canvas");
 auto event=[&](juce::Point<float> root,bool down=false){auto local=view->getLocalPoint(editor.get(),root);auto now=juce::Time::getCurrentTime();return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),local,juce::ModifierKeys(down?juce::ModifierKeys::leftButtonModifier:0),1,0,0,0,0,view,view,now,local,now,1,false);};
 auto click=[&](juce::Point<float> root){view->mouseMove(event(root));view->mouseDown(event(root,true));frames(18);view->mouseUp(event(root));frames();};
 auto save=[&](const char* name){view->mouseMove(event({4,4}));frames();auto stream=folder.getChildFile(name).createOutputStream();check(bool(stream),"Picker PNG open");stream->setPosition(0);stream->truncate();juce::PNGImageFormat png;check(png.writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream),"Picker PNG save");};
 frames();save("Corpus-Length-Picker-Before.png");
 // Feature control position is grounded in the previous native 1180-wide
 // Trajectories layout (Auto header) and in the initial capture above.
 click({575,445});check(value("ui_desc_popup")==100&&value("ui_desc_popup_scroll")==0,"Trajectory picker did not open at Core");save("Corpus-Length-Core-Menu.png");
 // Follow the popup's own layout, whose input coordinates are published by
 // its real opening click; menu order deliberately differs from descriptor ID.
 auto rowPoint=[&](int row){float pw=float(std::max(190,std::min(360,view->getWidth()-16))),ph=float(std::max(190,std::min(590,view->getHeight()-16)));float px=std::clamp(float(value("ui_desc_popup_x"))-pw*.18f,8.f,std::max(8.f,float(view->getWidth())-pw-8)),py=std::clamp(float(value("ui_desc_popup_y"))-36.f,8.f,std::max(8.f,float(view->getHeight())-ph-8));return editor->getLocalPoint(view,juce::Point<float>{px+80,py+42+28*row+14});};
 click(rowPoint(2));check(value("ui_desc_popup")==0&&lane()[1]==53&&int(corpusGet(p,86))%64==53,"Real Length menu click did not reach the saved trajectory");save("Corpus-Length-Selected.png");
 // Closing, reopening and switching to another descriptor must use display
 // order, while saved state continues storing the old descriptor IDs.
 click({575,445});check(value("ui_desc_popup_scroll")==0,"Reopened Length picker did not start at Core");click(rowPoint(3));check(lane()[1]==1,"Brightness display row changed its stable ID");click({575,445});click(rowPoint(2));check(lane()[1]==53,"Length second selection failed");
 juce::MemoryBlock state;p->getStateInformation(state);corpusSet(p,86,3073);process();p->setStateInformation(state.getData(),int(state.getSize()));frames();check(lane()[1]==53&&int(corpusGet(p,86))%64==53,"Length selection lost after preset recall");
 editor->setSize(1120,920);frames();click({575,445});check(value("ui_desc_popup")==100,"Compact picker open");save("Corpus-Length-Core-Menu-Compact.png");click(rowPoint(2));check(lane()[1]==53,"Compact Length click failed");
 uint32_t dsp=0,gfx=0;za_native_gfx_faults(p,&dsp,&gfx);check(!dsp&&!gfx,"Picker runtime fault");p->editorBeingDeleted(editor.get());editor.reset();
 std::cout<<"PICKER actual UI: Core Energy/Length/Brightness; Length click selects ID53, Brightness ID1; reopen, compact click and preset recall passed; faults0/0\n";
}
