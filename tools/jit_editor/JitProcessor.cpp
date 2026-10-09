#include "JsfxGfxInput.h"
#include "JsfxGfxMenus.h"
#include <deque>
#include <map>
#include "JitDeclarations.h"
#include "JitControlDefaults.h"
#include "JitProcessor.h"
#include "JitExamples.h"
#include "JsfxSliderValues.h"
#include <juce_gui_extra/juce_gui_extra.h>
#define NOMINMAX
#if JUCE_WINDOWS
#include <windows.h>
#endif
using namespace juce;
namespace {
NormalisableRange<float> range(const JsfxSliderDecl&);
class SourceEditor final : public CodeEditorComponent {
public:
    std::function<void()> onSave;
    SourceEditor(CodeDocument& d):CodeEditorComponent(d,nullptr){setMouseClickGrabsKeyboardFocus(true);
#if JUCE_WINDOWS
        editors.push_back(this);if(!hook)hook=SetWindowsHookExW(WH_GETMESSAGE,messageHook,nullptr,GetCurrentThreadId());
#endif
    }
    ~SourceEditor() override {
#if JUCE_WINDOWS
        editors.erase(std::remove(editors.begin(),editors.end(),this),editors.end());if(editors.empty() && hook){UnhookWindowsHookEx(hook);hook=nullptr;}
#endif
    }
    void paint(Graphics& g) override {
        // CodeEditorComponent's caret/hit-testing uses fixed columns. Drawing
        // shaped runs with proportional fallback glyphs breaks that contract,
        // even with a monospace primary font. Use the same grid for Unicode.
        g.fillAll(findColour(backgroundColourId));
        int gutter=0,right=getWidth(),bottom=getHeight();
        for(int i=0;i<getNumChildComponents();++i){auto* child=getChildComponent(i);
            if(auto* scroll=dynamic_cast<ScrollBar*>(child)){if(scroll->isVisible()){if(scroll->isVertical())right=scroll->getX();else bottom=scroll->getY();}}
            else if(child->getX()==0 && child->getHeight()==getHeight())gutter=child->getRight()+2;
        }
        g.reduceClipRegion(gutter,0,std::max(0,right-gutter),bottom);
        g.setColour(findColour(highlightColourId));g.fillRectList(getTextBounds(getHighlightedRegion()));
        g.setColour(findColour(defaultTextColourId));g.setFont(getFont());
        const auto clip=g.getClipBounds();const float cell=getFont().getStringWidthFloat("0");
        auto& doc=getDocument();
        for(int line=getFirstLineOnScreen();line<std::min(doc.getNumLines(),getFirstLineOnScreen()+getNumLinesOnScreen()+1);++line){
            const auto origin=getCharacterBounds({doc,line,0});if(origin.getBottom()<clip.getY() || origin.getY()>clip.getBottom())continue;
            int column=0;auto text=doc.getLine(line);auto chars=text.getCharPointer();
            while(!chars.isEmpty()){
                auto ch=chars.getAndAdvance();if(ch=='\r' || ch=='\n')break;
                const float x=origin.getX()+column*cell;
                if(ch=='\t'){column+=getTabSize()-column%getTabSize();continue;}
                if(x>clip.getRight())break;
                if(x+cell>=clip.getX() && !CharacterFunctions::isWhitespace(ch)){
                    Graphics::ScopedSaveState saved(g);g.reduceClipRegion({roundToInt(x),origin.getY(),std::max(1,(int)std::ceil(cell)),getLineHeight()});
                    // Fit wide fallback glyphs into their column; do not let
                    // shaping/ligatures shift later text away from its caret.
                    g.drawFittedText(String::charToString(ch),juce::Rectangle<int>(roundToInt(x),origin.getY(),std::max(1,(int)std::ceil(cell)),getLineHeight()),Justification::centred,1,0.1f);
                }
                ++column;
            }
        }
    }
    // REAPER can consume Ctrl shortcuts in its accelerator before dispatching
    // a queued Windows key message to the plugin's child window. Intercept only
    // editing shortcuts while this source editor actually owns keyboard focus.
#if JUCE_WINDOWS
    static LRESULT CALLBACK messageHook(int code,WPARAM removed,LPARAM value){
        if(code>=0 && removed==PM_REMOVE){auto* msg=reinterpret_cast<MSG*>(value);
            if(msg->message==WM_KEYDOWN && (GetKeyState(VK_CONTROL)&0x8000) && !(GetKeyState(VK_MENU)&0x8000))
                for(auto* editor:editors)if(editor->isShowing() && editor->hasKeyboardFocus(true)){
                    auto key=(int)msg->wParam;if(key=='A'||key=='C'||key=='X'||key=='V'||key=='Z'||key=='Y'||key=='S'){
                        int mods=ModifierKeys::ctrlModifier;if(GetKeyState(VK_SHIFT)&0x8000)mods|=ModifierKeys::shiftModifier;
                        if(editor->keyPressed(KeyPress(key,mods,0)))msg->message=WM_NULL;
                    }break;
                }
        }return CallNextHookEx(hook,code,removed,value);
    }
    inline static thread_local HHOOK hook=nullptr;
    inline static thread_local std::vector<SourceEditor*> editors;
#endif
    bool keyStateChanged(bool) override {return hasKeyboardFocus(true);}
    bool keyPressed(const KeyPress& key) override {
        if(key.getModifiers().isCtrlDown() || key.getModifiers().isCommandDown()) {
            auto raw=key.getKeyCode();if(raw>=1 && raw<=26)raw+='A'-1;auto ch=CharacterFunctions::toLowerCase((juce_wchar)raw);
            if(ch=='a')return selectAll(); if(ch=='c')return copyToClipboard();
            if(ch=='s' && onSave){onSave();return true;}
            if(ch=='v'){for(int attempt=0;attempt<5;++attempt){if(SystemClipboard::getTextFromClipboard().isNotEmpty())return pasteFromClipboard();Thread::sleep(2);}return true;} if(ch=='x')return cutToClipboard();
            if(ch=='y' || (ch=='z' && key.getModifiers().isShiftDown()))return redo();
            if(ch=='z')return undo();
        }
        return CodeEditorComponent::keyPressed(key);
    }
};
class GraphicsPanel final : public Component, public FileDragAndDropTarget, private Timer,private Thread,private AsyncUpdater {
public:
    explicit GraphicsPanel(JitProcessor& p):Thread("JIT native GFX"),processor(p){
        processor.engine.graphicsVisible.store(true);setWantsKeyboardFocus(true);addChildComponent(menuOverlay);
        menus.setCallbacks([this]{triggerAsyncUpdate();},[this]{const ScopedLock lock(inputLock);cap&=~67;edges.clear();},[this]{notify();});
        menuOverlay.onFinished=[this](int result){menus.completeMenu(result);};
        startThread();startTimerHz(30);
    }
    ~GraphicsPanel()override{processor.engine.graphicsVisible.store(false);stopTimer();menus.close();signalThreadShouldExit();notify();stopThread(10000);processor.endGfxGestures();cancelPendingUpdate();menuOverlay.forceCloseSilently();}
    void visibilityChanged()override{renderingEnabled.store(isVisible());processor.engine.graphicsVisible.store(isVisible());if(isVisible())startTimerHz(30);else{stopTimer();menus.cancelAll();menuOverlay.forceCloseSilently();}notify();}
    void paint(Graphics& g)override{g.fillAll(Colours::black);const ScopedLock lock(outputLock);if(image.isValid())g.drawImageAt(image,0,0);else{g.setColour(Colours::grey);g.drawText("Plugin graphics",getLocalBounds(),Justification::centred);}}
    void resized()override{menuOverlay.setBounds(getLocalBounds());const ScopedLock lock(inputLock);width=getWidth();height=getHeight();notify();}
    void mouseMove(const MouseEvent& e)override{const ScopedLock lock(inputLock);position=e.position;cap=flags(e.mods);notify();}
    void mouseDown(const MouseEvent& e)override{grabKeyboardFocus();mouseEdge(e);}
    void mouseDrag(const MouseEvent& e)override{mouseMove(e);}
    void mouseUp(const MouseEvent& e)override{mouseEdge(e);}
    void mouseWheelMove(const MouseEvent& e,const MouseWheelDetails& wheel)override{mouseMove(e);const ScopedLock lock(inputLock);wheelX+=wheel.deltaX*120;wheelY+=wheel.deltaY*120;notify();}
    bool keyPressed(const KeyPress& e)override{const ScopedLock lock(inputLock);int code=keyCode(e);keys.push_back(code);tracked[e.getKeyCode()]=code;cap=flags(e.getModifiers());notify();return true;}
    bool keyStateChanged(bool)override{const ScopedLock lock(inputLock);cap=flags(ModifierKeys::getCurrentModifiersRealtime());for(auto it=tracked.begin();it!=tracked.end();)if(!KeyPress::isKeyCurrentlyDown(it->first))it=tracked.erase(it);else ++it;notify();return true;}
    void focusLost(FocusChangeType)override{const ScopedLock lock(inputLock);tracked.clear();edges.clear();cap&=~67;notify();}
    bool isInterestedInFileDrag(const StringArray&)override{return true;}
    void filesDropped(const StringArray& files,int,int)override{const ScopedLock lock(inputLock);for(auto& file:files)drops.push_back(file);notify();}
private:
    void mouseEdge(const MouseEvent& e){const ScopedLock lock(inputLock);position=e.position;cap=flags(e.mods);edges.push_back({position,cap});notify();}
    static int flags(ModifierKeys m){return (m.isLeftButtonDown()?1:0)|(m.isRightButtonDown()?2:0)|(m.isCtrlDown()?4:0)|(m.isShiftDown()?8:0)|(m.isAltDown()?16:0)|(m.isMiddleButtonDown()?64:0);}
    static int keyCode(const KeyPress& e){return jsfx_input::keyPressToJsfx(e);}
    void handleAsyncUpdate()override{
        if(menus.takePendingCancel()){menuOverlay.forceCloseSilently();menus.completeMenu(0);}
        String description;int x,y;if(menus.takePendingOpen(description,x,y)){menuOverlay.openMenu(description,{x,y});if(!menuOverlay.isMenuShowing())menus.completeMenu(0);}
    }
    void timerCallback()override{
        if(seen!=processor.schemaRevision.load()){seen=processor.schemaRevision.load();menus.cancelAll();menuOverlay.forceCloseSilently();}
        {const ScopedLock lock(inputLock);focused=hasKeyboardFocus(true);}
        notify();repaint();
        switch(cursor.load()){case 32513:setMouseCursor(MouseCursor::IBeamCursor);break;case 32514:setMouseCursor(MouseCursor::WaitCursor);break;case 32515:setMouseCursor(MouseCursor::CrosshairCursor);break;case 32649:setMouseCursor(MouseCursor::PointingHandCursor);break;default:setMouseCursor(MouseCursor::NormalCursor);}
    }
    void run()override{
        while(!threadShouldExit()){
            if(!renderingEnabled.load()){wait(500);continue;}
            int w,h,c,k=0;double x,y,wx,wy;JitEngine::GraphicsInput input;
            {const ScopedLock lock(inputLock);w=width;h=height;x=position.x;y=position.y;c=cap;if(!edges.empty()){x=edges.front().position.x;y=edges.front().position.y;c=edges.front().cap;edges.pop_front();}wx=wheelX;wy=wheelY;wheelX=wheelY=0;if(!keys.empty()){k=keys.front();keys.pop_front();}input.menu=&menus;input.focused=focused;for(auto& key:tracked){input.keysDown.push_back(key.second);if(key.first>0 && key.first<128){input.keysDown.push_back(key.first);if(key.first>='A' && key.first<='Z')input.keysDown.push_back(key.first+32);}}input.drops.swap(drops);}
            if(w>0 && h>0){auto next=processor.engine.renderGraphics(w,h,x,y,c,k,wx,wy,&input).createCopy();cursor.store(input.cursor);if(next.isValid())for(int n=0;n<256;++n)processor.sliderVisible[(size_t)n].store((input.visible[n/64]&(UINT64_C(1)<<(n%64)))!=0);const ScopedLock lock(outputLock);image=next;}
            wait(500);
        }
    }
    JitProcessor& processor;CriticalSection inputLock,outputLock;Image image;Point<float> position;
    int width=0,height=0,cap=0;double wheelX=0,wheelY=0;bool focused=false;uint64_t seen=0;
    struct MouseEdge{Point<float> position;int cap;};std::deque<MouseEdge> edges;
    std::deque<int> keys;std::map<int,int> tracked;std::vector<String> drops;std::atomic<int> cursor{32512};std::atomic<bool> renderingEnabled{false};
    JsfxGfxMenuBridge menus;GfxMenuOverlay menuOverlay;
};
class Editor final : public AudioProcessorEditor, private Timer, private CodeDocument::Listener {
public:
    explicit Editor(JitProcessor& p):AudioProcessorEditor(p),processor(p),code(document),graphics(p) {
        for(auto* c:std::initializer_list<Component*>{&code,&graphics,&run,&reset,&edit,&oversampling,&idleMode,&open,&sourceFolder,&sourceLocation,&language,&examples,&status,&controlsViewport})addAndMakeVisible(c);
        controlsViewport.setViewedComponent(&controlsBody,false);
        run.setButtonText("Run");reset.setButtonText("Default");edit.setButtonText("Edit");open.setButtonText("Open source...");sourceFolder.setButtonText("Source folder...");
        edit.onClick=[this]{setEditing(!editing);if(editing)code.grabKeyboardFocus();};
        open.onClick=[this]{chooser=std::make_unique<FileChooser>("Open JSFX or Faust source",File{},"*.jsfx;*.dsp;*.txt");chooser->launchAsync(FileBrowserComponent::openMode|FileBrowserComponent::canSelectFiles,[safe=Component::SafePointer<Editor>(this)](const FileChooser& c){if(!safe)return;auto file=c.getResult();if(!file.existsAsFile())return;safe->processor.setSourceFile(file.getFullPathName());safe->document.replaceAllContent(file.loadFileAsString());safe->language.setSelectedId(file.hasFileExtension("dsp")?2:1);safe->setEditing(true);safe->code.grabKeyboardFocus();});};
        sourceFolder.setTooltip("Choose the original source directory for pasted code's imports and image resources. Press Run to apply.");
        sourceFolder.onClick=[this]{auto path=processor.sourcePath();auto initial=path.isEmpty()?File{}:File(path).getParentDirectory();chooser=std::make_unique<FileChooser>("Choose source folder containing dependencies and resources",initial);chooser->launchAsync(FileBrowserComponent::openMode|FileBrowserComponent::canSelectDirectories,[safe=Component::SafePointer<Editor>(this)](const FileChooser& c){if(!safe)return;auto folder=c.getResult();if(!folder.isDirectory())return;safe->processor.setSourcePath(folder.getChildFile("editor.jsfx").getFullPathName());safe->resized();});};
        oversampling.addItemList(StringArray{"OS: Off","OS: 2x","OS: 4x","OS: 8x"},1);oversampling.setSelectedId(processor.engine.oversamplingChoice.load()+1,dontSendNotification);oversampling.onChange=[this]{int choice=oversampling.getSelectedId()-1;processor.engine.oversamplingChoice.store(choice);if(auto* p=processor.oversamplingParameter){p->beginChangeGesture();p->setValueNotifyingHost(float(choice)/3);p->endChangeGesture();}};
        idleMode.addItemList(StringArray{"Sleep: Auto","Sleep: Input","Sleep: Events","Sleep: Free run","Sleep: Never","Sleep: Explicit"},1);idleMode.setSelectedId(processor.engine.idleOverride.load()+1,dontSendNotification);idleMode.onChange=[this]{processor.engine.idleOverride.store(idleMode.getSelectedId()-1);};idleMode.setTooltip("Production idle policy. Auto respects source options and readiness; Explicit requires za_sleep_ready. Never keeps audio processing awake.");
        language.addItem("JSFX / mixed",1);language.addItem("FAUST",2);
        language.setSelectedId(p.language()=="faust"?2:1,dontSendNotification);
        document.replaceAllContent(p.draft());document.addListener(this);document.setSavePoint();
        code.setFont(Font(FontOptions(Font::getDefaultMonospacedFontName(),15.0f,Font::plain)));code.setWantsKeyboardFocus(true);
        examples.setTextWhenNothingSelected("Load example...");
        for(int n=0;n<(int)std::size(jitExamples);++n)examples.addItem(String::fromUTF8(jitExamples[n].name),n+1);
        run.setTooltip("Compile and run the draft. Ctrl+S in the source editor saves the source file, then compiles and runs.");
        run.onClick=[this]{runDraft();};code.onSave=[this]{saveAndRun();};
        reset.onClick=[this]{processor.factory();document.replaceAllContent(processor.draft());language.setSelectedId(1);};
        language.onChange=[this]{saveDraft();};
        examples.onChange=[this]{int n=examples.getSelectedId()-1;if(n<0 || n>=(int)std::size(jitExamples))return;processor.setSourcePath({});document.replaceAllContent(String::fromUTF8(jitExamples[n].source));language.setSelectedId(String(jitExamples[n].mode)=="faust"?2:1);saveDraft();resized();};
        status.setJustificationType(Justification::topLeft);
        status.setVisible(false);
        setResizable(true,true);setSize(1150,800);
        refreshFeedback();buildControls();updateViewSize(true);startTimerHz(20);
    }
    ~Editor()override{for(auto& w:widgets)if(w->momentary && w->momentaryDown && seen==processor.schemaRevision.load()){if(auto* p=processor.parameters[(size_t)w->slot]){p->beginChangeGesture();processor.setSliderValue(w->slot,0);p->endChangeGesture();}}document.removeListener(this);saveDraft();}
    void paint(Graphics& g)override{g.fillAll(Colour(0xff202328));}
    void resized()override{
        auto a=getLocalBounds().reduced(12);auto bar=a.removeFromTop(32);
        edit.setBounds(bar.removeFromRight(110));bar.removeFromRight(8);oversampling.setBounds(bar.removeFromRight(100));bar.removeFromRight(8);
        if(!editing)idleMode.setBounds(bar.removeFromLeft(160));
        if(editing){
            run.setBounds(bar.removeFromLeft(85));bar.removeFromLeft(8);reset.setBounds(bar.removeFromLeft(85));bar.removeFromLeft(8);
            open.setBounds(bar.removeFromLeft(130));bar.removeFromLeft(8);language.setBounds(bar.removeFromLeft(std::min(165,bar.getWidth())));
            a.removeFromTop(6);auto next=a.removeFromTop(32);examples.setBounds(next.removeFromLeft(220));next.removeFromLeft(8);sourceFolder.setBounds(next.removeFromLeft(130));next.removeFromLeft(8);idleMode.setBounds(next.removeFromLeft(140));a.removeFromTop(4);sourceLocation.setBounds(a.removeFromTop(22));auto path=processor.sourcePath();auto location=path.isEmpty()?String::fromUTF8("Not set \xe2\x80\x94 use Open source or Source folder for dependencies/resources"):File(path).getParentDirectory().getFullPathName();sourceLocation.setText("Source folder: "+location,dontSendNotification);sourceLocation.setTooltip(location);
        }
        code.setVisible(editing);graphics.setVisible(hasGraphics);if(!hasGraphics)graphics.setBounds(0,0,0,0);run.setVisible(editing);reset.setVisible(editing);open.setVisible(editing);language.setVisible(editing);examples.setVisible(editing);sourceFolder.setVisible(editing);sourceLocation.setVisible(editing);
        int visibleCount=visibleControlCount();
        if(editing || hasGraphics || visibleCount || status.isVisible())a.removeFromTop(10);
        if(status.isVisible()){status.setBounds(a.removeFromBottom(feedbackHeight()));a.removeFromBottom(8);}else status.setBounds(0,0,0,0);
        int rows=std::min(hasGraphics||editing?5:8,visibleCount);controlsViewport.setVisible(visibleCount>0);controlsViewport.setBounds(visibleCount?(!hasGraphics&&!editing?a:a.removeFromBottom(rows*34)):juce::Rectangle<int>{});
        controlsBody.setSize(std::max(1,controlsViewport.getWidth()-18),std::max(1,visibleCount*34));
        int rowIndex=0;
        for(auto& w:widgets){
            const bool visible=processor.sliderVisible[(size_t)w->slot].load();
            w->label.setVisible(visible);
            w->slider.setVisible(visible&&!w->choice&&!w->text&&!w->toggle&&!w->momentary);
            if(w->text)w->text->setVisible(visible);
            if(w->choice)w->choice->setVisible(visible);
            if(w->toggle)w->toggle->setVisible(visible);
            if(w->momentary)w->momentary->setVisible(visible);
            if(!visible)continue;
            auto row=juce::Rectangle<int>(0,rowIndex++*34,controlsBody.getWidth(),32);
            w->label.setBounds(row.removeFromLeft(std::min(260,row.getWidth()/3)));
            if(w->toggle)w->toggle->setBounds(row);
            else if(w->momentary)w->momentary->setBounds(row);
            else if(w->choice)w->choice->setBounds(row);
            else if(w->text)w->text->setBounds(row);
            else w->slider.setBounds(row);
        }
        for(auto& button:fileButtons)button->setBounds(0,rowIndex++*34,controlsBody.getWidth(),32);
        if(visibleCount && (editing || hasGraphics))a.removeFromBottom(8);if(editing){if(hasGraphics){graphics.setBounds(a.removeFromRight(std::max(240,a.getWidth()/3)));a.removeFromRight(8);}code.setBounds(a);}else if(hasGraphics)graphics.setBounds(a);
    }
private:
    void runDraft(){fileError.clear();processor.setFrontend("python-reference");processor.runSource(document.getAllContent(),selectedLanguage());awaitingRun=true;refreshFeedback();resized();}
    void saveAndRun(){
        auto path=processor.sourceFile();
        if(path.isNotEmpty()){saveToFile(File(path));return;}
        auto origin=processor.sourcePath();auto initial=origin.isEmpty()?File{}:File(origin).getParentDirectory();
        if(initial.isDirectory())initial=initial.getChildFile(selectedLanguage()=="faust"?"program.dsp":"program.jsfx");
        chooser=std::make_unique<FileChooser>("Save source and run",initial,selectedLanguage()=="faust"?"*.dsp":"*.jsfx");
        chooser->launchAsync(FileBrowserComponent::saveMode|FileBrowserComponent::canSelectFiles|FileBrowserComponent::warnAboutOverwriting,[safe=Component::SafePointer<Editor>(this)](const FileChooser& c){if(safe && c.getResult()!=File{})safe->saveToFile(c.getResult());});
    }
    void saveToFile(const File& file){
        // On POSIX, JUCE's replacement helper can remove an empty directory.
        // A stale/preset save target must never replace a directory with source.
        if(file.isDirectory()){fileError="Save failed: source target is a directory: "+file.getFullPathName()+"\nThe running program was retained.";refreshFeedback();resized();return;}
        TemporaryFile temporary(file);bool wrote=false;
        {auto stream=temporary.getFile().createOutputStream();if(stream){auto text=document.getAllContent();wrote=stream->write(text.toRawUTF8(),text.getNumBytesAsUTF8());stream->flush();wrote=wrote && stream->getStatus().wasOk();}}
        if(!wrote || !temporary.overwriteTargetFileWithTemporary()){fileError="Save failed: could not replace "+file.getFullPathName()+"\nThe previous source file and running program were retained.";refreshFeedback();resized();return;}
        processor.setSourceFile(file.getFullPathName());document.setSavePoint();runDraft();
    }
    int visibleControlCount()const{int count=(int)fileButtons.size();for(auto& w:widgets)if(processor.sliderVisible[(size_t)w->slot].load())++count;return count;}
    bool refreshFeedback(){
        const auto message=processor.engine.status();run.setButtonText(message.startsWith("Compiling")?"Compiling...":"Run");
        const auto error=fileError.isNotEmpty()?fileError:(message.startsWith("Run failed")?message:String{});
        const bool changed=status.getText()!=error || status.isVisible()!=error.isNotEmpty();
        status.setText(error,dontSendNotification);status.setTooltip(error);status.setVisible(error.isNotEmpty());
        if(error.isNotEmpty())awaitingRun=false;
        return changed;
    }
    int feedbackHeight()const{return status.isVisible()?jlimit(52,140,24+18*(StringArray::fromLines(status.getText()).size()-1)):0;}
    int compactHeight()const{const int count=visibleControlCount();return (count || status.isVisible()?66:56)+std::min(8,count)*34+(status.isVisible()?feedbackHeight()+8:0);}
    void updateViewSize(bool newProgram=false){
        const auto minimum=editing||hasGraphics?520:compactHeight();setResizeLimits(780,minimum,2400,1800);
        if(editing)setSize(getWidth(),std::max(600,getHeight()));
        else if(newProgram)setSize(getWidth(),hasGraphics?800:compactHeight());
        else setSize(pluginViewSize.x,std::max(minimum,pluginViewSize.y));
        if(newProgram&&!editing)pluginViewSize={getWidth(),getHeight()};
        resized();
    }
    void setEditing(bool value){if(value&&!editing)pluginViewSize={getWidth(),getHeight()};editing=value;edit.setButtonText(editing?"Show plugin":"Edit");updateViewSize();}
    struct Widget{jit_controls::DefaultLabel label;jit_controls::DefaultSlider slider;std::unique_ptr<jit_controls::DefaultChoice> choice;std::unique_ptr<jit_controls::DefaultTextEditor> text;std::unique_ptr<jit_controls::DefaultToggle> toggle;std::unique_ptr<jit_controls::DefaultButton> momentary;bool momentaryDown=false;int slot=0;};
    void resetControl(int slot,uint64_t revision){
        // An open menu from the previous program must not reset a new program.
        if(revision!=processor.schemaRevision.load() || !processor.resetSliderValue(slot))return;
        for(auto& w:widgets)if(w->slot==slot){
            if(w->text)w->text->setText(processor.engine.stringSlider(slot),false);
            else if(w->toggle)w->toggle->setToggleState(processor.sliderValue(slot)>=.5,dontSendNotification);
            else if(w->momentary)w->momentary->setToggleState(processor.sliderValue(slot)>=.5,dontSendNotification);
            else if(w->choice){auto* p=processor.parameters[(size_t)slot];if(p)w->choice->setSelectedId(1+roundToInt(p->getValue()*(w->choice->getNumItems()-1)),dontSendNotification);}
            else w->slider.setValue(processor.sliderValue(slot),dontSendNotification);
            break;
        }
    }
    String selectedLanguage()const{return language.getSelectedId()==2?"faust":"jsfx";}
    void saveDraft(){processor.updateDraft(document.getAllContent(),selectedLanguage());}
    void codeDocumentTextInserted(const String&,int)override{saveDraft();}
    void codeDocumentTextDeleted(int,int)override{saveDraft();}
    void buildControls(){widgets.clear();fileButtons.clear();for(auto& file:parseJsfxFilenameDecls(processor.engine.compiledDescriptor().getProperty("expandedSource",processor.engine.appliedSource()).toString().toRawUTF8()))if(file.enhancedImport){auto button=std::make_unique<TextButton>("Load "+file.name+"...");button->onClick=[this,slot=file.index0]{chooser=std::make_unique<FileChooser>("Select audio files",File{},"*.wav;*.flac;*.aiff;*.ogg;*.mp3");chooser->launchAsync(FileBrowserComponent::openMode|FileBrowserComponent::canSelectFiles|FileBrowserComponent::canSelectMultipleItems,[safe=Component::SafePointer<Editor>(this),slot](const FileChooser& c){if(!safe)return;StringArray paths;for(auto& f:c.getResults())if(f.existsAsFile())paths.add(f.getFullPathName());if(!paths.isEmpty())safe->processor.engine.setFileSlot(slot,paths);});};controlsBody.addAndMakeVisible(*button);fileButtons.push_back(std::move(button));}for(auto& d:processor.declarations){auto w=std::make_unique<Widget>();w->slot=d.index0;controlsBody.addAndMakeVisible(w->label);w->label.setText(d.name,dontSendNotification);
        const auto kind=jitControlKind(d.index0,processor.engine.compiledDescriptor());
        if(kind==JitControlKind::toggle){w->toggle=std::make_unique<jit_controls::DefaultToggle>();controlsBody.addAndMakeVisible(*w->toggle);w->toggle->setButtonText("Enabled");w->toggle->setToggleState(processor.sliderValue(d.index0)>=.5,dontSendNotification);auto* item=w.get();w->toggle->onClick=[this,item]{auto* p=processor.parameters[(size_t)item->slot];if(p)p->beginChangeGesture();processor.setSliderValue(item->slot,item->toggle->getToggleState()?1:0);if(p)p->endChangeGesture();};}
        else if(kind==JitControlKind::momentary){w->momentary=std::make_unique<jit_controls::DefaultButton>();controlsBody.addAndMakeVisible(*w->momentary);w->momentary->setButtonText(d.name);auto* item=w.get();w->momentary->onStateChange=[this,item]{const bool down=item->momentary->isDown();if(down==item->momentaryDown)return;item->momentaryDown=down;auto* p=processor.parameters[(size_t)item->slot];if(p)p->beginChangeGesture();processor.setSliderValue(item->slot,down?1:0);if(p)p->endChangeGesture();};}
        else if(d.isString){w->text=std::make_unique<jit_controls::DefaultTextEditor>();controlsBody.addAndMakeVisible(*w->text);w->text->setText(processor.engine.stringSlider(d.index0),false);auto* item=w.get();w->text->onTextChange=[this,item]{processor.engine.setStringSlider(item->slot,item->text->getText());};}
        else if(d.isChoice){w->choice=std::make_unique<jit_controls::DefaultChoice>();controlsBody.addAndMakeVisible(*w->choice);for(int n=0;n<d.choices.size();++n)w->choice->addItem(d.choices[n],n+1);if(auto* parameter=processor.parameters[(size_t)d.index0])w->choice->setSelectedId(1+roundToInt(parameter->getValue()*(d.choices.size()-1)),dontSendNotification);auto* item=w.get();w->choice->onChange=[this,item,d]{auto* p=processor.parameters[(size_t)item->slot];if(p)p->beginChangeGesture();processor.setSliderValue(item->slot,d.rangeStart+(item->choice->getSelectedId()-1)*(d.reversed?-d.step:d.step));if(p)p->endChangeGesture();};}
        else{controlsBody.addAndMakeVisible(w->slider);auto r=range(d);w->slider.setNormalisableRange(NormalisableRange<double>(d.min,d.max,[r](double,double,double v){return (double)r.convertFrom0to1((float)v);},[r](double,double,double v){return (double)r.convertTo0to1((float)v);},[r](double,double,double v){return (double)r.snapToLegalValue((float)v);}));w->slider.setTextBoxStyle(Slider::TextBoxRight,false,90,24);int digits=4;if(d.step>0){digits=0;double scaled=d.step;while(digits<8 && (std::round(scaled)==0 || std::abs(scaled-std::round(scaled))>1e-6*std::max(1.0,std::abs(scaled)))){++digits;scaled*=10;}}w->slider.setNumDecimalPlacesToDisplay(digits);w->slider.setValue(processor.sliderValue(d.index0),dontSendNotification);auto* item=w.get();w->slider.onDragStart=[this,item]{if(auto* p=processor.parameters[(size_t)item->slot])p->beginChangeGesture();};w->slider.onDragEnd=[this,item]{if(auto* p=processor.parameters[(size_t)item->slot])p->endChangeGesture();};w->slider.onValueChange=[this,item]{processor.setSliderValue(item->slot,item->slider.getValue());};}
        auto resetAction=[safe=Component::SafePointer<Editor>(this),slot=d.index0,revision=processor.schemaRevision.load()]{if(safe)safe->resetControl(slot,revision);};
        String defaultText=d.isString?(d.stringDefault.isEmpty()?String("empty"):d.stringDefault):w->slider.getTextFromValue(d.def);
        if(d.isChoice || kind!=JitControlKind::continuous)if(auto* p=processor.parameters[(size_t)d.index0])defaultText=p->getText(p->getDefaultValue(),128);
        auto tip=d.tooltip+(d.tooltip.isEmpty()?String{}:String("\n"))+"Right-click for Reset to default ("+defaultText+").";
        w->label.defaultDescription=defaultText;w->label.onReset=resetAction;w->label.setTooltip(tip);
        if(w->toggle){w->toggle->defaultDescription=defaultText;w->toggle->onReset=resetAction;w->toggle->setTooltip(tip);}
        else if(w->momentary){w->momentary->defaultDescription=defaultText;w->momentary->onReset=resetAction;w->momentary->setTooltip(tip+" Hold to press; release returns to zero.");}
        else if(w->text){w->text->defaultDescription=defaultText;w->text->onReset=resetAction;w->text->setTooltip(tip);}
        else if(w->choice){w->choice->defaultDescription=defaultText;w->choice->onReset=resetAction;w->choice->setTooltip(tip);}
        else{w->slider.defaultDescription=defaultText;w->slider.onReset=resetAction;w->slider.setTooltip(tip+" Double-click the slider to reset.");}
        widgets.push_back(std::move(w));}seen=processor.schemaRevision.load();hasGraphics=(bool)processor.engine.compiledDescriptor().getProperty("metadata",var{}).getProperty("sections_present",var{}).getProperty("gfx",false);resized();}
    void timerCallback()override{
        idleMode.setSelectedId(processor.engine.idleOverride.load()+1,dontSendNotification);oversampling.setSelectedId(processor.engine.oversamplingChoice.load()+1,dontSendNotification);const bool feedbackChanged=refreshFeedback();
        if(seen!=processor.schemaRevision.load()){
            const bool hadGraphics=hasGraphics;buildControls();
            if(awaitingRun){awaitingRun=false;editing=false;edit.setButtonText("Edit");updateViewSize(true);}
            else if(hadGraphics!=hasGraphics || (!editing&&!hasGraphics))updateViewSize(true);
        }
        bool relayout=feedbackChanged;for(auto& w:widgets)if(w->label.isVisible()!=processor.sliderVisible[(size_t)w->slot].load())relayout=true;if(relayout){if(!editing&&!hasGraphics)updateViewSize(true);else resized();}
        for(auto& w:widgets){if(w->text)continue;auto v=processor.sliderValue(w->slot);if(w->toggle)w->toggle->setToggleState(v>=.5,dontSendNotification);else if(w->momentary)w->momentary->setToggleState(v>=.5,dontSendNotification);else if(w->choice){auto* p=processor.parameters[(size_t)w->slot];if(p)w->choice->setSelectedId(1+roundToInt(p->getValue()*(w->choice->getNumItems()-1)),dontSendNotification);}else if(!w->slider.isMouseButtonDown())w->slider.setValue(v,dontSendNotification);}
    }
    JitProcessor& processor;CodeDocument document;SourceEditor code;GraphicsPanel graphics;TextButton run,reset,edit,open,sourceFolder;Label sourceLocation;String fileError;bool editing=false,awaitingRun=false,hasGraphics=false;Point<int> pluginViewSize{1150,800};std::unique_ptr<FileChooser> chooser;ComboBox language,examples,oversampling,idleMode;Label status;
    Viewport controlsViewport;Component controlsBody;std::vector<std::unique_ptr<TextButton>> fileButtons;std::vector<std::unique_ptr<Widget>> widgets;uint64_t seen=0;
};
NormalisableRange<float> range(const JsfxSliderDecl& s){
    auto from=[s](float,float,float t){if(s.shape==JsfxSliderDecl::Shape::Sqr)return curveFrom01_sqr(t,s.rangeStart,s.rangeEnd,s.shapeModifier>0?s.shapeModifier:2);if(s.shape==JsfxSliderDecl::Shape::Log)return curveFrom01_log(t,s.rangeStart,s.rangeEnd,s.shapeModifier);return s.rangeStart+t*(s.rangeEnd-s.rangeStart);};
    auto to=[s](float,float,float v){if(s.shape==JsfxSliderDecl::Shape::Sqr)return curveTo01_sqr(v,s.rangeStart,s.rangeEnd,s.shapeModifier>0?s.shapeModifier:2);if(s.shape==JsfxSliderDecl::Shape::Log)return curveTo01_log(v,s.rangeStart,s.rangeEnd,s.shapeModifier);return clamp01f((v-s.rangeStart)/(s.rangeEnd-s.rangeStart));};
    auto snap=[s](float,float,float v){if(s.step>0)v=s.rangeStart+std::round((v-s.rangeStart)/s.step)*s.step;return jlimit(s.min,s.max,v);};return {s.min,s.max,from,to,snap};
}
}
JitProcessor::JitProcessor():AudioProcessor(BusesProperties().withInput("Input",AudioChannelSet::stereo(),true).withOutput("Output",AudioChannelSet::stereo(),true)){
    engine.hostProcessor=this;
    slotIndex.fill(-1);
    engine.compiledCallback=[this]{triggerAsyncUpdate();};
    engine.sliderVisibilityChanged=[this](uint64_t revision,const std::array<uint64_t,4>& visible){if(revision!=configured.load())return;for(int i=0;i<256;++i)sliderVisible[(size_t)i].store((visible[(size_t)i/64]&(UINT64_C(1)<<(i%64)))!=0);};
    engine.graphicsSliderChanged=[this](uint64_t revision,int slot,double value,int events){if(slot<0 || slot>=256 || revision!=configured.load())return;if(MessageManager::getInstance()->isThisTheMessageThread())applyGfxSlider(slot,value,events);else{gfxValues[(size_t)slot].store(value);gfxRevision[(size_t)slot].store(revision);gfxEvents[(size_t)slot].fetch_or(events);gfxPending[(size_t)slot].store(true);triggerAsyncUpdate();}};
}
JitProcessor::~JitProcessor(){engine.shutdown();cancelPendingUpdate();engine.compiledCallback={};engine.graphicsSliderChanged={};engine.sliderVisibilityChanged={};}
void JitProcessor::pollCompiledSchema(){auto descriptor=engine.compiledDescriptor();auto rev=(uint64_t)(int64)descriptor.getProperty("revision",0);if(rev<=configured)return;pendingSchema=descriptor;
    if(requestHostReconfigure)requestHostReconfigure();else{const ScopedLock lock(getCallbackLock());commitHostConfiguration();}}
bool JitProcessor::commitHostConfiguration(){if(!pendingSchema.isObject())return false;
    auto newDecls=jitDeclarations(engine.appliedSource(),pendingSchema);
    endGfxGestures();
    AudioProcessorParameterGroup group("program","Program","|");parameters.fill(nullptr);declarations=std::move(newDecls);slotIndex.fill(-1);for(size_t i=0;i<declarations.size();++i)slotIndex[(size_t)declarations[i].index0]=(int)i;
    for(auto& d:declarations){if(d.isString)continue;std::unique_ptr<RangedAudioParameter> p;
        if(jitControlKind(d.index0,pendingSchema)!=JitControlKind::continuous)p=std::make_unique<AudioParameterBool>(ParameterID(d.id,1),d.name,d.def>=.5f);
        else if(d.isChoice)p=std::make_unique<AudioParameterChoice>(ParameterID(d.id,1),d.name,d.choices,jlimit(0,d.choices.size()-1,roundToInt((d.def-d.rangeStart)/(d.reversed?-d.step:d.step))));
        else p=std::make_unique<AudioParameterFloat>(ParameterID(d.id,1),d.name,range(d),d.def);
        parameters[(size_t)d.index0]=p.get();group.addChild(std::move(p));}
    for(int i=0;i<256;++i){gfxPending[(size_t)i].store(false);gfxEvents[(size_t)i].store(0);}
    auto oversampling=std::make_unique<AudioParameterChoice>(ParameterID("ZA_INTERNAL_OVERSAMPLING",1),"Oversampling",StringArray{"Off","2x","4x","8x"},engine.oversamplingChoice.load());oversamplingParameter=oversampling.get();group.addChild(std::move(oversampling));
    setParameterTree(std::move(group));
    for(auto& d:declarations)sliderVisible[(size_t)d.index0].store(!d.hidden);
    auto io=pendingSchema.getProperty("metadata",var()).getProperty("io_channels",var());
    int in=(int)io.getProperty("inputs",2),out=(int)io.getProperty("outputs",2);
    auto layout=getBusesLayout();layout.inputBuses.set(0,AudioChannelSet::canonicalChannelSet(in));layout.outputBuses.set(0,AudioChannelSet::canonicalChannelSet(out));setBusesLayout(layout);
    for(auto& d:declarations)if(parameters[(size_t)d.index0]){if(restoring){double value=restoredValues[(size_t)d.index0];if(restoredNormalized)value=d.isChoice?d.rangeStart+roundToInt(value*(d.choices.size()-1))*(d.reversed?-d.step:d.step):parameters[(size_t)d.index0]->convertFrom0to1((float)value);setSliderValue(d.index0,value);}else engine.controls[(size_t)d.index0].store(sliderValue(d.index0));}
    restoring=false;configured=(uint64_t)(int64)pendingSchema.getProperty("revision",0);pendingSchema=var();schemaRevision.fetch_add(1);engine.refreshSliderVisibility();engine.allowAdoption();return true;
}
void JitProcessor::prepareToPlay(double rate,int block){
    Array<var> current;for(int i=0;i<256;++i)current.add(sliderValue(i));
    if(engine.prepare(rate,block,current)){restoring=true;restoredNormalized=false;for(int i=0;i<256;++i)restoredValues[(size_t)i]=(double)current[i];}
}
void JitProcessor::processBlock(AudioBuffer<float>& b,MidiBuffer& midi){ScopedNoDenormals guard;if(oversamplingParameter)engine.oversamplingChoice.store(oversamplingParameter->getIndex());for(auto& d:declarations)if(parameters[(size_t)d.index0])engine.controls[(size_t)d.index0].store(sliderValue(d.index0),std::memory_order_relaxed);auto pos=getPlayHead()?getPlayHead()->getPosition():Optional<AudioPlayHead::PositionInfo>{};engine.process(b,&midi,pos?&*pos:nullptr,isNonRealtime());if(getLatencySamples()!=engine.latencySamples.load())triggerAsyncUpdate();}
double JitProcessor::sliderValue(int slot)const{if(slot<0 || slot>=256)return 0;auto* p=parameters[(size_t)slot];int index=slotIndex[(size_t)slot];if(!p || index<0)return 0;auto& d=declarations[(size_t)index];if(auto* choice=dynamic_cast<AudioParameterChoice*>(p))return za::jsfx::hostSliderValue(d,choice->getIndex());if(auto* floating=dynamic_cast<AudioParameterFloat*>(p))return za::jsfx::hostSliderValue(d,floating->get());return za::jsfx::hostSliderValue(d,p->convertFrom0to1(p->getValue()));}
bool JitProcessor::resetSliderValue(int slot){
    if(slot<0 || slot>=256)return false;
    const auto index=slotIndex[(size_t)slot];if(index<0)return false;
    const auto& d=declarations[(size_t)index];
    if(d.isString){engine.setStringSlider(slot,d.stringDefault);return true;}
    auto* p=parameters[(size_t)slot];if(!p)return false;
    // The host owns the normalized mapping, including reversed/curved ranges
    // and nonzero/fractional choice indices. Reset without touching DSP history.
    p->beginChangeGesture();p->setValueNotifyingHost(p->getDefaultValue());p->endChangeGesture();
    engine.controls[(size_t)slot].store(sliderValue(slot));return true;
}

void JitProcessor::endGfxGestures(){for(int i=0;i<256;++i)if(gfxGestures[(size_t)i]){if(parameters[(size_t)i])parameters[(size_t)i]->endChangeGesture();gfxGestures[(size_t)i]=false;}}
void JitProcessor::applyGfxSlider(int slot,double value,int events){auto* p=parameters[(size_t)slot];if(!p)return;if((events&2) && !gfxGestures[(size_t)slot]){p->beginChangeGesture();gfxGestures[(size_t)slot]=true;}setSliderValue(slot,value);if((events&4) && gfxGestures[(size_t)slot]){p->endChangeGesture();gfxGestures[(size_t)slot]=false;}}
void JitProcessor::setSliderValue(int slot,double value){if(slot<0 || slot>=256)return;auto* p=parameters[(size_t)slot];int index=slotIndex[(size_t)slot];if(!p || index<0)return;auto& d=declarations[(size_t)index];float norm=d.isChoice?(float)((value-d.rangeStart)/(d.reversed?-d.step:d.step))/std::max(1,d.choices.size()-1):p->convertTo0to1((float)value);p->setValueNotifyingHost(jlimit(0.0f,1.0f,norm));engine.controls[(size_t)slot].store(sliderValue(slot));}
AudioProcessorEditor* JitProcessor::createEditor(){return new Editor(*this);}
String JitProcessor::draft()const{const ScopedLock lock(sourceLock);return source;}String JitProcessor::language()const{const ScopedLock lock(sourceLock);return mode;}
void JitProcessor::updateDraft(const String& text,const String& kind){const ScopedLock lock(sourceLock);source=text;mode=kind;}
void JitProcessor::runSource(const String& text,const String& kind){updateDraft(text,kind);engine.submit(text,kind,sourcePath(),{},frontend());}
void JitProcessor::setFrontend(const String& value){const ScopedLock lock(sourceLock);selectedFrontend=value=="cpp-frontend"?value:String("python-reference");}
String JitProcessor::frontend()const{const ScopedLock lock(sourceLock);return selectedFrontend;}
void JitProcessor::setSourcePath(const String& path){const ScopedLock lock(sourceLock);origin=path;savedSourceFile.clear();}
String JitProcessor::sourcePath()const{const ScopedLock lock(sourceLock);return origin;}
void JitProcessor::setSourceFile(const String& path){const ScopedLock lock(sourceLock);origin=path;savedSourceFile=path;}
String JitProcessor::sourceFile()const{const ScopedLock lock(sourceLock);return savedSourceFile;}
void JitProcessor::factory(){engine.restoreFactory();setSourcePath({});updateDraft("desc:Factory passthrough\n@sample\nspl0=spl0;spl1=spl1;\n","jsfx");}
void JitProcessor::getStateInformation(MemoryBlock& b) {
    const ScopedLock lock(getCallbackLock());
    auto serialized=engine.serializeState();auto running=serialized.getProperty("program",var{});
    DynamicObject::Ptr o=new DynamicObject;
    o->setProperty("version",3);o->setProperty("sourcePath",sourcePath());o->setProperty("sourceFile",sourceFile());o->setProperty("serialized",serialized);
    o->setProperty("draft",draft());o->setProperty("mode",language());o->setProperty("frontend",frontend());
    o->setProperty("appliedFrontend",running.getProperty("frontend",engine.appliedFrontend()));
    o->setProperty("appliedSource",running.getProperty("source",engine.appliedSource()));
    o->setProperty("appliedMode",running.getProperty("mode",engine.appliedMode()));
    o->setProperty("appliedSourcePath",running.getProperty("sourcePath",engine.appliedSourcePath()));
    o->setProperty("oversamplingChoice",engine.oversamplingChoice.load());o->setProperty("idleOverride",engine.idleOverride.load());
    o->setProperty("activeRevision",(int64)engine.activeRevision());o->setProperty("status",engine.status());
    Array<var> values;for(int i=0;i<256;++i)values.add(sliderValue(i));o->setProperty("controls",values);
    auto text=JSON::toString(var(o.get()));b.replaceAll(text.toRawUTF8(),text.getNumBytesAsUTF8());
}
void JitProcessor::setStateInformation(const void* data,int size){
    auto v=JSON::parse(String::fromUTF8((const char*)data,size));if(!v.isObject())return;
    engine.idleOverride.store(jlimit(0,5,(int)v.getProperty("idleOverride",0)));engine.oversamplingChoice.store(jlimit(0,3,(int)v.getProperty("oversamplingChoice",0)));
    setFrontend(v.getProperty("frontend",v.getProperty("appliedFrontend","python-reference")).toString());setSourcePath(v.getProperty("sourcePath","").toString());
    auto file=v.getProperty("sourceFile","").toString();if(file.isNotEmpty() && file==sourcePath())setSourceFile(file);
    auto text=v.getProperty("appliedSource",{}).toString();auto kind=v.getProperty("appliedMode","jsfx").toString();
    updateDraft(v.getProperty("draft",text).toString(),v.getProperty("mode",kind).toString());auto controls=v.getProperty("controls",{});
    if(auto* a=controls.getArray()){restoring=true;restoredNormalized=(int)v.getProperty("version",1)<2;for(int i=0;i<std::min(256,a->size());++i)restoredValues[(size_t)i]=(double)(*a)[i];}
    if(text.isEmpty())engine.restoreFactory();else engine.submit(text,kind,v.getProperty("appliedSourcePath",sourcePath()).toString(),v.getProperty("serialized",{}),v.getProperty("appliedFrontend","python-reference").toString(),controls,restoredNormalized);
}
AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new JitProcessor();}
