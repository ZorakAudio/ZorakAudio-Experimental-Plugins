#pragma once
#include "JitControlDefaults.h"
#include "JitDeclarations.h"

namespace {
template<class T> std::vector<T*> descendants(juce::Component& root) {
    std::vector<T*> result;
    for(int i=0;i<root.getNumChildComponents();++i){auto* child=root.getChildComponent(i);if(auto* item=dynamic_cast<T*>(child))result.push_back(item);auto nested=descendants<T>(*child);result.insert(result.end(),nested.begin(),nested.end());}
    return result;
}
void clickButton(juce::Component& root,const juce::String& name) {
    for(auto* b:descendants<juce::TextButton>(root))if(b->getButtonText()==name){b->onClick();return;}
    throw std::runtime_error("Missing button: "+name.toStdString());
}
void pumpEditor(){
    const auto start=juce::Time::getMillisecondCounter();
    while(juce::Time::getMillisecondCounter()-start<100){MSG message;while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}juce::Thread::sleep(1);}
}
struct ParameterEvents final : juce::AudioProcessorParameter::Listener {
    std::vector<int> events;
    void parameterValueChanged(int,float)override{events.push_back(2);}
    void parameterGestureChanged(int,bool starting)override{events.push_back(starting?1:3);}
};
void checkControls(bool cpp) {
    using namespace jit_controls;
    JitProcessor p;if(cpp)p.setFrontend("cpp-frontend");p.prepareToPlay(48000,256);
    const juce::String source="slider1:gain=.25<0,1,.01>Gain\nslider2:freq=120<20,20000,1:log>Frequency\nslider3:rev=7<10,0,1>Reverse\nslider4:square=.25<0,1,.01:sqr=2>Square\nslider5:mode=3.5<4,3,.5{High,Mid,Low}>Mode\nslider6:hidden=.5<0,1,.01>-Hidden\nslider7:#text=\"initial\"<string>Text\n@init\nframes=0;\n@sample\nframes+=1;spl0*=gain;spl1*=gain;";
    runCode(p,source,"jsfx",.25f);
    std::unique_ptr<juce::AudioProcessorEditor> editor(p.createEditor());
    auto sliders=descendants<DefaultSlider>(*editor);auto choices=descendants<DefaultChoice>(*editor);auto texts=descendants<DefaultTextEditor>(*editor);auto labels=descendants<DefaultLabel>(*editor);auto sources=descendants<juce::CodeEditorComponent>(*editor);
    require(sliders.size()==5 && choices.size()==1 && texts.size()==1,"Generated numeric/choice/string controls");
    require(!p.engine.graphicsVisible.load() && editor->getHeight()==270,"No-GFX compact view: visible controls only");
    require(!sliders[4]->isVisible() && !labels[5]->isVisible(),"Hidden control remains hidden");
    require(sources.size()==1 && !sources[0]->isVisible(),"No-GFX source initially hidden");
    clickButton(*editor,"Edit");require(sources[0]->isVisible() && editor->getHeight()>=600 && sources[0]->getWidth()>editor->getWidth()-50,"No-GFX editing uses full width");
    clickButton(*editor,"Show plugin");require(!sources[0]->isVisible() && editor->getHeight()==270,"Show plugin restores compact size");
    for(int slot:{0,1,2,3,4}){
        auto* param=p.parameters[(size_t)slot];ParameterEvents observer;param->addListener(&observer);
        p.setSliderValue(slot,slot==0?.8:slot==1?900:slot==2?2:slot==3?.81:3);
        observer.events.clear();labels[(size_t)slot]->performReset();
        require(observer.events==std::vector<int>({1,2,3}),"Reset notifies host with one balanced parameter gesture");
        require(std::abs(param->getValue()-param->getDefaultValue())<1e-6,"Reset uses host mapping for curves/reversed/fractional choices");
        param->removeListener(&observer);
    }
    auto menu=choices[0]->resetMenu();juce::PopupMenu::MenuItemIterator iterator(menu);require(iterator.next() && iterator.getItem().text.contains("Mid") && iterator.getItem().itemID==1,"Choice reset menu shows declared default label");
    p.setSliderValue(4,4);choices[0]->performReset();require(p.sliderValue(4)==3.5 && choices[0]->getSelectedId()==2,"Choice reset updates widget and nonzero fractional DSP value");
    p.engine.setStringSlider(6,"edited");texts[0]->setText("edited",false);juce::PopupMenu textMenu;texts[0]->addPopupMenuItems(textMenu,nullptr);require(textMenu.getNumItems()>2,"String menu retains normal editing commands");texts[0]->performPopupMenuAction(DefaultTextEditor::resetItem);require(p.engine.stringSlider(6)=="initial" && texts[0]->getText()=="initial","String reset restores declaration");
    p.setSliderValue(0,.8);ParameterEvents observer;p.parameters[0]->addListener(&observer);
    const auto now=juce::Time::getCurrentTime();juce::MouseEvent doubleClick(juce::Desktop::getInstance().getMainMouseSource(),{20,12},juce::ModifierKeys::leftButtonModifier,1,0,0,0,0,sliders[0],sliders[0],now,{20,12},now,2,false);
    sliders[0]->mouseDown(doubleClick);sliders[0]->mouseDoubleClick(doubleClick);sliders[0]->mouseUp(doubleClick);
    require(observer.events==std::vector<int>({1,2,3}) && std::abs(p.sliderValue(0)-.25)<1e-6,"Double-click resets without nested/duplicate host gestures");p.parameters[0]->removeListener(&observer);
    juce::AudioBuffer<float> audio(2,256);juce::MidiBuffer midi;const auto frames=p.engine.inspectVariable("frames");fill(audio);p.processBlock(audio,midi);checkGain(audio,.25f);require(p.engine.inspectVariable("frames")==frames+256,"Reset reaches DSP without clearing its history");
    const auto oldReset=sliders[0]->onReset;
    runCode(p,"slider1:g=.75<0,1,.01>New gain\n@sample\nspl0*=g;spl1*=g;\n@gfx 100 100\ngfx_set(1,0,0,1);gfx_rect(0,0,gfx_w,gfx_h);","jsfx",.75f);
    p.setSliderValue(0,.4);oldReset();require(std::abs(p.sliderValue(0)-.4)<1e-6,"Stale menu cannot reset replacement schema");pumpEditor();require(p.engine.graphicsVisible.load() && editor->getHeight()==800,"Adding GFX restores graphics view");
    runCode(p,"slider1:g=.3<0,1,.01>New gain\n@sample\nspl0*=g;spl1*=g;","jsfx",.3f);pumpEditor();require(!p.engine.graphicsVisible.load() && editor->getHeight()==100,"Removing GFX restores compact view");
    editor.reset();oldReset(); // SafePointer survives editor destruction without dereferencing it.
    runCode(p,"slider1:g=.3<0,1,.01>Gain\nslider2:other=0<0,1,1>-Conditional\n@slider\nslider_show(slider2,g>.5);\n@sample\nspl0*=g;spl1*=g;","jsfx",.3f);
    require(!p.sliderVisible[1].load(),"@slider can hide controls without GFX");p.setSliderValue(0,.8);fill(audio);p.processBlock(audio,midi);require(p.sliderVisible[1].load(),"@slider can reveal controls without GFX");
    runCode(p,"slider1:g=.3<0,1,.01>Gain\nslider2:other=0<0,1,1>-Conditional\n@slider\nslider_show(slider2,g>.5);\n@sample\nspl0*=g;spl1*=g;\n@gfx\nslider_show(slider2,mouse_cap&1 ? 1 : g>.5);","jsfx",.3f);
    JitEngine::GraphicsInput visibility;p.engine.renderGraphics(100,100,0,0,1,0,0,0,&visibility);require(visibility.visible[0]&2,"GFX reveals conditional control");fill(audio);p.processBlock(audio,midi);p.setSliderValue(0,.8);fill(audio);p.processBlock(audio,midi);p.setSliderValue(0,.1);fill(audio);p.processBlock(audio,midi);p.engine.renderGraphics(100,100,0,0,0,0,0,0,&visibility);require(!(visibility.visible[0]&2),"DSP visibility changes reach isolated GFX state");
    runCode(p,"process=_,_:*(hslider(\"Gain\",0.5,0,1,0.01)),*(hslider(\"Gain\",0.5,0,1,0.01));","faust",.5f);
    editor.reset(p.createEditor());auto faust=descendants<DefaultSlider>(*editor);require(faust.size()==1 && !p.engine.graphicsVisible.load(),"Pure Faust uses compact generated controls");p.setSliderValue(0,.1);faust[0]->performReset();fill(audio);p.processBlock(audio,midi);checkGain(audio,.5f);
    editor.reset();
    runCode(p,"g=hslider(\"Gain\",.5,0,1,.01); e=checkbox(\"Enable\"); b=hslider(\"Binary\",0,0,1,1); n=nentry(\"Binary entry\",0,0,1,1); t=button(\"Trigger\"); process=*(g*(1-e)+b+n+t),*(g*(1-e)+b+n+t);","faust",.5f);
    editor.reset(p.createEditor());
    auto toggles=descendants<DefaultToggle>(*editor);auto buttons=descendants<DefaultButton>(*editor);faust=descendants<DefaultSlider>(*editor);
    require(toggles.size()==3 && buttons.size()==1 && faust.size()==1,"Faust checkbox/integer binary controls use toggles; button stays momentary; continuous gain stays a slider");
    for(auto* toggle:toggles)require(toggle->isVisible() && toggle->getWidth()>0,"Binary control is visible and laid out");
    require(buttons[0]->isVisible() && buttons[0]->getWidth()>0,"Momentary control is visible and laid out");
    int trigger=-1,enable=-1;
    for(auto& d:p.declarations){if(d.name=="Trigger")trigger=d.index0;if(d.name=="Enable")enable=d.index0;if(d.name!="Gain")require(dynamic_cast<juce::AudioParameterBool*>(p.parameters[(size_t)d.index0])!=nullptr,"Binary Faust host parameter is boolean");}
    require(trigger>=0 && enable>=0,"Faust binary declarations retained");
    int firstToggle=-1;for(auto& d:p.declarations)if(jitControlKind(d.index0,p.engine.compiledDescriptor())==JitControlKind::toggle){firstToggle=d.index0;break;}
    auto* toggle=toggles[0];toggle->setToggleState(true,juce::dontSendNotification);toggle->onClick();
    require(firstToggle>=0 && p.sliderValue(firstToggle)==1,"Toggle click sets its boolean DSP control");
    toggle->performReset();require(!toggle->getToggleState(),"Binary reset restores off default and widget");
    ParameterEvents binaryObserver;p.parameters[(size_t)trigger]->addListener(&binaryObserver);
    buttons[0]->setState(juce::Button::buttonDown);require(p.sliderValue(trigger)==1,"Faust button presses at mouse-down");
    fill(audio);p.processBlock(audio,midi);checkGain(audio,1.5f);
    buttons[0]->setState(juce::Button::buttonNormal);require(p.sliderValue(trigger)==0,"Faust button releases at mouse-up");
    require(binaryObserver.events==std::vector<int>({1,2,3,1,2,3}),"Momentary transitions use balanced host gestures");
    p.parameters[(size_t)trigger]->removeListener(&binaryObserver);
    buttons[0]->setState(juce::Button::buttonDown);editor.reset();require(p.sliderValue(trigger)==0,"Closing editor releases a held Faust button");
    require(!p.resetSliderValue(-1) && !p.resetSliderValue(256) && !p.resetSliderValue(200),"Missing/out-of-range control reset is ignored");
    std::cout<<"CONTROL DEFAULTS PASS "<<(cpp?"cpp":"standard")<<": reset menus, curves/reversed/choice/string/Faust defaults, host gestures, binary toggle/momentary/continuous controls, held-button close, hidden/dynamic visibility, double-click, DSP history, stale callbacks, compact/edit/GFX transitions\n";
}
void checkGraphicsConcurrency(){
    JitProcessor p;p.prepareToPlay(48000,256);
    runCode(p,"@block\ni=1000000;loop(1000,i=1000000;);\n@sample\nspl0*=1;spl1*=1;\n@gfx 64 32\ngfx_set(0,0,0,1);gfx_rect(0,0,64,32);gfx_set(1,0,0,1);i=0;loop(64,loop(256,scratch=0;);gfx_rect(i,0,1,32);i+=1;);","jsfx",1);
    std::atomic<bool> running{true};std::thread audio([&]{juce::AudioBuffer<float> buffer(2,256);juce::MidiBuffer midi;while(running.load()){fill(buffer);p.processBlock(buffer,midi);}});
    int bad=0,minimum=64;for(int frame=0;frame<60;++frame){auto image=p.engine.renderGraphics(64,32,0,0,0);int columns=0;for(int x=0;x<64;++x)columns+=image.getPixelAt(x,16).getRed()>240;minimum=std::min(minimum,columns);bad+=columns!=64;}
    running.store(false);audio.join();std::cout<<"GFX CONCURRENCY: bad frames="<<bad<<"/60 minimum columns="<<minimum<<"/64\n";
    require(bad==0,"Audio processing corrupts the graphics loop's working variables");
}
}
