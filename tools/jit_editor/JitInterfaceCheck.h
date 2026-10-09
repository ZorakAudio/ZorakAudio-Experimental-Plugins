#pragma once

namespace {
juce::String utf8(const char8_t* text){return juce::String::fromUTF8(reinterpret_cast<const char*>(text));}
bool visibleLabel(juce::Component& root,const juce::String& prefix){for(auto* label:descendants<juce::Label>(root))if(label->isVisible() && label->getText().startsWith(prefix))return true;return false;}
void waitForRun(JitProcessor& p,uint64_t before,bool failure=false){
    juce::AudioBuffer<float> audio(2,256);juce::MidiBuffer midi;const auto started=juce::Time::getMillisecondCounterHiRes();
    while(juce::Time::getMillisecondCounterHiRes()-started<45000){p.pollCompiledSchema();fill(audio);p.processBlock(audio,midi);pumpEditor();
        if(failure && p.engine.status().startsWith("Run failed"))return;
        require(failure || !p.engine.status().startsWith("Run failed"),p.engine.status());
        if(!failure && p.engine.activeRevision()!=before)return;
    }
    throw std::runtime_error("Interface Run timed out: "+p.engine.status().toStdString());
}
void checkInterfaceUnicode(const juce::File& output){
    JitProcessor p;p.prepareToPlay(48000,256);
    std::unique_ptr<juce::AudioProcessorEditor> editor(p.createEditor());
    require(editor->getHeight()==56 && !p.engine.graphicsVisible.load(),"Empty program collapses GFX, controls and routine footer");
    for(auto* view:descendants<juce::Viewport>(*editor))require(!view->isVisible() && view->getBounds().isEmpty(),"Empty controls viewport collapses");
    require(!visibleLabel(*editor,"Factory:"),"Factory information removed from UI");
    clickButton(*editor,"Edit");
    require(descendants<juce::ToggleButton>(*editor).empty(),"Experimental frontend checkbox removed");
    require(visibleLabel(*editor,utf8(u8"Source folder: Not set — use Open source")),"UI UTF-8 dash is decoded correctly");
    auto* code=descendants<juce::CodeEditorComponent>(*editor).at(0);
    require(std::abs(code->getFont().getStringWidthFloat("iiii")-code->getFont().getStringWidthFloat("WWWW"))<.01,"Source editor uses a fixed-width primary font");
    // Pixel evidence, actual edit operations, tabs and horizontal scrolling.
    // Each rendered glyph must occupy the cell returned by JUCE hit-testing.
    const auto line=utf8(u8"WWii0123456789 abcdefghijklmnopqrstuvwxyz éΩ漢🙂XYZ");
    code->getDocument().replaceAllContent(line+"\n\tABC\n");
    code->setColour(juce::CodeEditorComponent::backgroundColourId,juce::Colours::black);
    code->setColour(juce::CodeEditorComponent::defaultTextColourId,juce::Colours::white);
    pumpEditor();
    auto snapshot=code->createComponentSnapshot(code->getLocalBounds());
    for(int i=0;i<line.length();++i){
        auto cell=code->getCharacterBounds({code->getDocument(),0,i});
        auto hit=code->getPositionAt(cell.getX()+1,cell.getCentreY());require(hit.getIndexInLine()==i,"Caret/hit-testing agree at every ASCII/Unicode codepoint");
        if(line[i]!=' '){int lit=0;for(int y=cell.getY();y<cell.getBottom();++y)for(int x=cell.getX();x<cell.getRight();++x)lit+=snapshot.getPixelAt(x,y).getBrightness()>.15f;require(lit>0,"Rendered glyph is present in its caret cell at index "+juce::String(i));}
    }
    const int unicode=line.indexOf(utf8(u8"漢"));
    code->getDocument().newTransaction();code->moveCaretTo({code->getDocument(),0,unicode+1},false);code->keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey));
    require(code->getDocument().getLine(0).trimEnd()==line.substring(0,unicode)+line.substring(unicode+1),"Backspace erases the Unicode codepoint immediately before the visible caret");
    require(code->keyPressed(juce::KeyPress('Z',juce::ModifierKeys::ctrlModifier,0)) && code->getDocument().getLine(0).trimEnd()==line,"Ctrl+Z restores Unicode edit");
    auto tab=code->getCharacterBounds({code->getDocument(),1,1});auto tabStart=code->getCharacterBounds({code->getDocument(),1,0});
    require(std::abs((tab.getX()-tabStart.getX())-code->getTabSize()*code->getFont().getStringWidthFloat("0"))<=1,"Tabs use caret column width");
    code->scrollToColumn(12);pumpEditor();auto scrolled=code->getCharacterBounds({code->getDocument(),0,20});require(code->getPositionAt(scrolled.getX()+1,scrolled.getCentreY()).getIndexInLine()==20,"Horizontal scroll keeps mouse/caret aligned");code->scrollToColumn(0);
    const auto clipboard=juce::SystemClipboard::getTextFromClipboard();
    juce::SystemClipboard::copyTextToClipboard(utf8(u8"éΩ漢🙂"));code->selectAll();code->keyPressed(juce::KeyPress('V',juce::ModifierKeys::ctrlModifier,0));
    require(code->getDocument().getAllContent()==utf8(u8"éΩ漢🙂"),"Unicode clipboard paste preserves all codepoints");
    juce::SystemClipboard::copyTextToClipboard(clipboard);

    const auto folder=juce::File::getSpecialLocation(juce::File::tempDirectory).getNonexistentChildFile("za-unicode-"+utf8(u8"é漢"),"",false);
    struct Cleanup{juce::File folder;~Cleanup(){if(folder.getParentDirectory()==juce::File::getSpecialLocation(juce::File::tempDirectory) && folder.getFileName().startsWith("za-unicode-"))folder.deleteRecursively();}} cleanup{folder};
    require(folder.createDirectory().wasOk(),"Unicode source folder");
    auto dependency=folder.getChildFile(utf8(u8"係数.jsfx-inc"));require(dependency.replaceWithText("@init\nfunction imported_gain()(1;);\n"),"Unicode import filename");
    auto imageFile=folder.getChildFile(utf8(u8"画像.png"));juce::Image image(juce::Image::RGB,5,5,true);image.clear(image.getBounds(),juce::Colours::red);
    {auto stream=imageFile.createOutputStream();require(stream && juce::PNGImageFormat().writeImageToStream(image,*stream),"Unicode image filename");}
    const auto title=utf8(u8"Café Ω 漢字 🙂");const auto source=utf8(u8"desc:Unicode é Ω 漢字\nimport 係数.jsfx-inc\nslider1:gain=.25<0,1,.01>Gain é Ω 漢字\nslider2:mode=0<0,1,1{閉,開}>Mode Ω\nslider3:hidden=0<0,1,1>-Hidden\nslider4:#title=\"Café Ω 漢字 🙂\"<string>Title 漢字\n@init\nframes=0;\n@sample\nframes+=1;spl0*=gain*imported_gain();spl1*=gain*imported_gain();\n@gfx 320 100\ngfx_setfont(1,\"Arial\",20);gfx_set(1,1,1,1);gfx_x=5;gfx_y=5;gfx_drawstr(#title);gfx_measurestr(#title,measured,h);loaded=gfx_loadimg(5,\"画像.png\");gfx_x=5;gfx_y=50;gfx_blit(5,1,0);\n");
    const auto sourceFile=folder.getChildFile(utf8(u8"楽器.jsfx"));require(sourceFile.replaceWithText(source),"UTF-8 source file");p.setSourceFile(sourceFile.getFullPathName());code->getDocument().replaceAllContent(sourceFile.loadFileAsString());
    // Old projects can retain the developer backend; pressing the public Run
    // button always selects the standard pipeline, with no hidden UI option.
    p.setFrontend("cpp-frontend");auto before=p.engine.activeRevision();clickButton(*editor,"Run");
    require(p.frontend()=="python-reference","User Run selects standard compiler");
    require(!visibleLabel(*editor,"Compiling"),"Compilation uses Run button instead of footer");waitForRun(p,before);pumpEditor();
    require(p.engine.appliedFrontend()=="python-reference","Standard compiler actually applied");
    require(!code->isVisible() && !visibleLabel(*editor,"Running") && !visibleLabel(*editor,"Ready"),"Successful Run hides source and routine footer");
    require(p.parameters[0]->getName(128)==utf8(u8"Gain é Ω 漢字") && p.declarations[1].choices==juce::StringArray{utf8(u8"閉"),utf8(u8"開")},"Unicode host labels and choices");
    require(p.engine.stringSlider(3)==title,"Unicode string default");
    auto rendered=p.engine.renderGraphics(320,100,0,0,0);require(rendered.isValid() && p.engine.inspectGraphicsVariable("loaded")==5 && p.engine.inspectGraphicsVariable("measured")>20,"Unicode GFX text and image resource");
    int ink=0;for(int y=0;y<40;++y)for(int x=0;x<300;++x)ink+=rendered.getPixelAt(x,y).getBrightness()>.15f;require(ink>30 && rendered.getPixelAt(6,51).getRed()>200,"Text and source-relative image actually render");
    if(output.isDirectory()){auto stream=output.getChildFile("unicode-gfx.png").createOutputStream();require(stream && juce::PNGImageFormat().writeImageToStream(rendered,*stream),"GFX screenshot");}

    p.setSliderValue(0,.75);p.engine.setStringSlider(3,title+" changed");const auto applied=p.engine.appliedSource();const auto old=p.engine.activeRevision();const auto frames=p.engine.inspectVariable("frames");
    clickButton(*editor,"Edit");code->getDocument().replaceAllContent(source+utf8(u8"// 未適用の編集\n"));
    require(p.engine.activeRevision()==old && std::abs(p.sliderValue(0)-.75)<1e-6 && p.engine.inspectVariable("frames")==frames,"Draft edits preserve active program, parameters and DSP history");
    juce::MemoryBlock saved;p.getStateInformation(saved);JitProcessor restored;restored.prepareToPlay(48000,256);restored.setStateInformation(saved.getData(),(int)saved.getSize());waitForRun(restored,0);
    require(restored.sourcePath()==sourceFile.getFullPathName() && restored.draft()==code->getDocument().getAllContent() && restored.engine.appliedSource()==applied,"Saved project preserves Unicode path, applied source and separate draft");
    require(restored.sourceFile()==sourceFile.getFullPathName(),"Saved project preserves explicit source save target");
    require(std::abs(restored.sliderValue(0)-.75)<1e-6 && restored.engine.stringSlider(3)==title+" changed","Project reload preserves numeric/string controls");
    require(restored.engine.renderGraphics(320,100,0,0,0).isValid() && restored.engine.inspectGraphicsVariable("loaded")==5,"Restored Unicode resource path");
    code->getDocument().replaceAllContent(source+"@sample\nspl0*=;\n");clickButton(*editor,"Run");waitForRun(p,old,true);pumpEditor();
    require(visibleLabel(*editor,"Run failed") && code->isVisible() && p.engine.activeRevision()==old && std::abs(p.sliderValue(0)-.75)<1e-6 && p.engine.stringSlider(3)==title+" changed","Failed Run displays diagnostic and preserves program/controls");
    code->getDocument().replaceAllContent(source);clickButton(*editor,"Run");waitForRun(p,old);pumpEditor();
    require(std::abs(p.sliderValue(0)-.25)<1e-6 && p.engine.stringSlider(3)==title && p.engine.inspectVariable("frames")<=256,"Successful Run resets numeric/string defaults and DSP history");
    require(!visibleLabel(*editor,"Run failed") && !code->isVisible(),"Successful correction removes error footer");
    clickButton(*editor,"Edit");code->getDocument().replaceAllContent(source+utf8(u8"// Ctrl+S 保存\n"));p.setSliderValue(0,.8);before=p.engine.activeRevision();
    require(code->keyPressed(juce::KeyPress('S',juce::ModifierKeys::ctrlModifier,0)),"Ctrl+S handled by source editor");
    const auto disk=sourceFile.loadFileAsString();require(disk==code->getDocument().getAllContent(),"Ctrl+S saves the complete UTF-8 draft to the opened source file");waitForRun(p,before);pumpEditor();
    require(p.engine.appliedSource()==disk && std::abs(p.sliderValue(0)-.25)<1e-6,"Ctrl+S compiles/runs the saved source with fresh defaults");
    clickButton(*editor,"Edit");auto blocked=folder.getChildFile("blocked");require(blocked.createDirectory().wasOk(),"Save failure fixture");p.setSourceFile(blocked.getFullPathName());before=p.engine.activeRevision();code->keyPressed(juce::KeyPress('S',juce::ModifierKeys::ctrlModifier,0));pumpEditor();
    require(visibleLabel(*editor,"Save failed") && p.engine.activeRevision()==before && sourceFile.loadFileAsString()==disk && blocked.isDirectory(),"Failed save retains running program, original disk source and target directory");p.setSourceFile(sourceFile.getFullPathName());
    // Normal Run is still available after a failed save and clears its error.
    clickButton(*editor,"Run");waitForRun(p,before);pumpEditor();
    clickButton(*editor,"Edit");code->getDocument().replaceAllContent(line+"\n\tABC\n");pumpEditor();
    if(output.isDirectory()){auto screenshot=editor->createComponentSnapshot(editor->getLocalBounds());auto stream=output.getChildFile("unicode-editor.png").createOutputStream();require(stream && juce::PNGImageFormat().writeImageToStream(screenshot,*stream),"Editor screenshot");}
    editor.reset();
    runCode(p,utf8(u8"g=hslider(\"Gain é Ω 漢字\",.5,0,1,.01);b=checkbox(\"開閉 Ω\");process=*(g+b),*(g+b);"),"faust",.5f);
    bool gain=false,binary=false;for(auto& d:p.declarations){gain|=d.name==utf8(u8"Gain é Ω 漢字");binary|=d.name==utf8(u8"開閉 Ω");}require(gain && binary,"Pure Faust Unicode control labels");
    editor.reset(p.createEditor());require(!p.engine.graphicsVisible.load() && editor->getHeight()==134,"Pure Faust controls-only layout has no empty GFX/footer");
    runCode(p,"slider1:g=.5<0,1,.01>-Hidden\n@sample\nspl0*=g;spl1*=g;","jsfx",.5f);pumpEditor();
    require(editor->getHeight()==56,"Replacing controls-only schema with all-hidden controls collapses its window");
    runCode(p,"slider1:g=.5<0,1,.01>Gain\n@sample\nspl0*=g;spl1*=g;","jsfx",.5f);pumpEditor();require(editor->getHeight()==100,"Replacing controls-only schema reflows visible rows");
    std::cout<<"INTERFACE UNICODE PASS: frontend UI removed, standard Run, compact empty view, conditional error footer, monospace/fallback glyph pixels, caret/backspace/undo/clipboard/tabs/scroll, Unicode source/import/image paths, host labels/choices/strings, GFX pixels, saved draft/applied source and control restore, failed Run retention, successful Run defaults/fresh DSP, pure Faust controls\n";
}
}
