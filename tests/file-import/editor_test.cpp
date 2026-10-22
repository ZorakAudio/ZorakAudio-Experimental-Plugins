#include <juce_audio_utils/juce_audio_utils.h>
#include "ZAAudioImportRecipe.h"
#include <iostream>
#include <stdexcept>
using namespace za::fileimport;
static void check(bool ok,const char*text){if(!ok)throw std::runtime_error(text);}
static void pump(int ms){juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);}
static void save(juce::Component& c,juce::File f){auto stream=f.createOutputStream();if(stream){stream->setPosition(0);stream->truncate();}check(stream!=nullptr&&juce::PNGImageFormat().writeImageToStream(c.createComponentSnapshot(c.getLocalBounds()),*stream),"Screenshot failed");}
static juce::MouseEvent event(juce::Component& c,juce::Point<float> p,int flags){const auto t=juce::Time::getCurrentTime();return{juce::Desktop::getInstance().getMainMouseSource(),p,juce::ModifierKeys(flags),1.f,0.f,0.f,0.f,0.f,&c,&c,t,p,t,1,false};}
namespace za::fileimport {
struct ImportPreviewTestAccess {
    static void ready(ImportPreviewComponent& c){for(int n=0;n<800&&(c.loading||c.pendingPreview);++n)pump(10);check(!c.loading&&c.source!=nullptr,c.previewError.toRawUTF8());}
    static bool key(ImportPreviewComponent& c,int code,int mods=0){return c.keyPressed(juce::KeyPress(code,juce::ModifierKeys(mods),0));}
    static void run(ImportPreviewComponent& c,juce::File out){
        ready(c);check(c.rules.boundaryDetectorVersion==3,"New detector is not default");check(c.previewSegments.size()>=3,"Fixture detection failed");
        save(c,out.getChildFile("01-extractor-wide.png"));
        const int ctrl=juce::ModifierKeys::ctrlModifier,shift=juce::ModifierKeys::shiftModifier;
        c.waveform.grabKeyboardFocus();check(key(c,'F'),"Fit selection shortcut failed");auto range=c.waveform.visibleRange();check(range.second-range.first<c.source->audio.buffer.getNumSamples(),"Fit did not zoom");
        const int oldSpan=range.second-range.first;int chosen=c.selected;check(key(c,juce::KeyPress::rightKey),"Next shortcut failed");check(c.selected!=chosen,"Next did not select");range=c.waveform.visibleRange();check(range.second-range.first==oldSpan,"Navigation changed zoom");check(c.previewSegments[size_t(c.selected)].startSample>=range.first&&c.previewSegments[size_t(c.selected)].startSample<=range.second,"Selection was not revealed");
        check(key(c,juce::KeyPress::leftKey),"Previous shortcut failed");check(c.selected==chosen,"Previous did not restore selection");
        auto start=c.previewSegments[size_t(c.selected)].startSample;key(c,'.');check(c.previewSegments[size_t(c.selected)].startSample==start+48,"1 ms start nudge failed");
        key(c,'Z',ctrl);ready(c);check(c.previewSegments[size_t(c.selected)].startSample==start,"Undo failed");key(c,'Y',ctrl);ready(c);check(c.previewSegments[size_t(c.selected)].startSample==start+48,"Redo failed");key(c,'Z',ctrl);ready(c);
        key(c,juce::KeyPress::homeKey);auto cursor=c.waveform.getCursorSample();key(c,juce::KeyPress::rightKey,ctrl|shift);check(c.waveform.getCursorSample()==cursor+48,"Fine cursor nudge failed");key(c,'[');check(c.previewSegments[size_t(c.selected)].startSample==cursor+48,"Set start at cursor failed");key(c,'Z',ctrl);ready(c);
        int count=int(c.previewSegments.size());key(c,'S');check(int(c.previewSegments.size())==count+1,"Split shortcut failed");key(c,'M');check(int(c.previewSegments.size())<=count+1,"Merge shortcut failed");key(c,'Z',ctrl);ready(c);key(c,'Z',ctrl);ready(c);
        c.waveform.grabKeyboardFocus();chosen=c.selected;key(c,juce::KeyPress::returnKey);check(c.previewSegments[size_t(chosen)].locked&&c.selected!=chosen,"Confirm + advance failed");key(c,'Z',ctrl);ready(c);
        c.waveform.grabKeyboardFocus();chosen=c.selected;key(c,juce::KeyPress::deleteKey);check(!c.previewSegments[size_t(chosen)].enabled&&c.selected!=chosen,"Delete + advance failed");key(c,'Z',ctrl);ready(c);
        // Numeric editing must keep arrow/space/letter shortcuts away from DSP edits.
        c.section=1;c.resized();c.segmentStart.showTextBox();pump(30);chosen=c.selected;check(!key(c,juce::KeyPress::rightKey),"Arrow stolen from numeric input");check(c.selected==chosen,"Typing changed selection");c.segmentStart.hideTextBox(true);c.waveform.grabKeyboardFocus();
        c.section=0;c.advancedOpen=true;c.resized();c.detector.grabKeyboardFocus();check(!key(c,'S'),"Dropdown typing was intercepted");c.waveform.grabKeyboardFocus();
        key(c,'/',shift);check(c.keyHelp.isVisible(),"Physical Shift+/ shortcut guide failed");save(c,out.getChildFile("02-shortcuts.png"));key(c,juce::KeyPress::escapeKey);check(!c.keyHelp.isVisible(),"Guide does not close with Esc");key(c,'?');check(c.keyHelp.isVisible(),"Character shortcut guide failed");key(c,'?');check(!c.keyHelp.isVisible(),"Guide does not toggle closed");
        key(c,'A');check(c.autoplay.getToggleState(),"Auto-play shortcut failed");key(c,juce::KeyPress::rightKey);for(int n=0;n<300&&c.auditionPending;++n)pump(5);check(c.auditionActive,"Auto-play navigation did not audition");key(c,juce::KeyPress::spaceKey);check(c.auditionPaused,"Space did not pause");key(c,juce::KeyPress::spaceKey);check(!c.auditionPaused,"Space did not resume");key(c,juce::KeyPress::escapeKey);check(!c.auditionActive,"Esc did not stop");key(c,'A');
        // Test real edge mouse handlers and timer ticks with a held, stationary pointer.
        c.selected=0;c.updateControls();c.waveform.fitSelection(c.previewSegments[0].startSample,c.previewSegments[0].endSample);auto& w=c.waveform;
        auto bounds=w.getSourceWaveBounds();auto before=w.visibleRange();auto point=juce::Point<float>(float(w.xFromSample(c.previewSegments[0].endSample)),float(bounds.getCentreY()));
        w.mouseDown(event(w,point,juce::ModifierKeys::leftButtonModifier));check(w.draggingSegment==0&&!w.draggingStart,"End handle did not grab");
        auto edge=juce::Point<float>(float(bounds.getRight()+20),point.y);w.mouseDrag(event(w,edge,juce::ModifierKeys::leftButtonModifier));pump(400);auto after=w.visibleRange();check(after.first>before.first&&after.second-after.first==before.second-before.first,"Right edge did not continuously pan at same zoom");
        w.mouseUp(event(w,edge,0));before=w.visibleRange();pump(80);check(w.visibleRange()==before,"Scrolling continued after release");save(c,out.getChildFile("03-follow-right-edge.png"));
        // Fractional scroll steps must accumulate even at single-sample zoom.
        w.visibleStartSample=c.previewSegments[0].endSample-4;w.visibleEndSample=w.visibleStartSample+8;bounds=w.getSourceWaveBounds();before=w.visibleRange();point={float(w.xFromSample(c.previewSegments[0].endSample)),float(bounds.getCentreY())};w.mouseDown(event(w,point,juce::ModifierKeys::leftButtonModifier));w.mouseDrag(event(w,{float(bounds.getRight()+20),point.y},juce::ModifierKeys::leftButtonModifier));pump(400);check(w.visibleRange().first>before.first,"Sample-level zoom stopped edge scrolling");w.mouseUp(event(w,point,0));
        // Use a middle region to leave room to pan left.
        c.selected=1;c.updateControls();w.fitSelection(c.previewSegments[1].startSample,c.previewSegments[1].endSample);bounds=w.getSourceWaveBounds();before=w.visibleRange();point={float(w.xFromSample(c.previewSegments[1].startSample)),float(bounds.getCentreY())};w.mouseDown(event(w,point,juce::ModifierKeys::leftButtonModifier));check(w.draggingSegment==1&&w.draggingStart,"Start handle did not grab");edge={float(bounds.getX()-20),point.y};w.mouseDrag(event(w,edge,juce::ModifierKeys::leftButtonModifier));pump(350);check(w.visibleRange().first<before.first,"Left edge did not pan");w.mouseUp(event(w,edge,0));
        // Creating a region also follows the edge; all changes remain undoable.
        before=w.visibleRange();point={float(bounds.getCentreX()),float(bounds.getCentreY())};w.mouseDown(event(w,point,ctrl|juce::ModifierKeys::leftButtonModifier));w.mouseDrag(event(w,{float(bounds.getRight()+12),point.y},ctrl|juce::ModifierKeys::leftButtonModifier));pump(200);check(w.visibleRange().first>before.first,"Create-region edge pan failed");w.mouseUp(event(w,{float(bounds.getRight()+12),point.y},0));
        c.waveform.grabKeyboardFocus();key(c,'0');check(w.visibleRange().first==0&&w.visibleRange().second==c.source->audio.buffer.getNumSamples(),"Fit whole source failed");
        for(const auto size:{juce::Point<int>(960,640),juce::Point<int>(720,520)}){c.setSize(size.x,size.y);pump(50);save(c,out.getChildFile("04-extractor-"+juce::String(size.x)+"x"+juce::String(size.y)+".png"));check(c.next.getBounds().getBottom()<w.getY(),"Transport not visible");}
        c.setSize(1180,760);key(c,juce::KeyPress::pageDownKey);ready(c);check(c.fileIndex==1,"Next source failed");key(c,juce::KeyPress::pageUpKey);ready(c);check(c.fileIndex==0,"Previous source failed");
        // Persisted detector version is explicit; a missing field is v1.
        auto tree=rulesToValueTree(c.rules);auto roundtrip=rulesFromValueTree(tree);check(roundtrip.boundaryDetectorVersion==3,"New detector version did not serialize");tree.removeProperty("boundaryDetectorVersion",nullptr);auto old=rulesFromValueTree(tree);check(old.boundaryDetectorVersion==1,"Old recipe changed detector");
        auto snapshot=c.previewSegments;storeSegmentSnapshot(old,0,snapshot,48000);auto replay=segmentsForInput(old,0,c.source->audio.buffer,48000);check(replay.size()==snapshot.size(),"Saved region count changed");for(size_t i=0;i<replay.size();++i)check(replay[i].startSample==snapshot[i].startSample&&replay[i].endSample==snapshot[i].endSample&&replay[i].enabled==snapshot[i].enabled,"Saved boundaries changed");
        auto locked=c.previewSegments.front();locked.locked=true;locked.enabled=true;old.assistedExtraction=true;old.boundaryDetectorVersion=2;storeSegmentSnapshot(old,0,{locked},48000);auto protectedCuts=segmentsForInput(old,0,c.source->audio.buffer,48000,nullptr,{},nullptr,true);bool found=false;for(const auto&r:protectedCuts)if(r.locked&&r.startSample==locked.startSample&&r.endSample==locked.endSample)found=true;check(found,"Re-analysis overwrote protected edit");
        std::cout<<"PASS navigation, text focus, transport, shortcuts, both held-edge scrolls, region creation, undo/redo, source switching, serialization, saved cuts and protected re-analysis\n";
    }
};}
int main(int argc,char**argv)try{check(argc==2,"Pass output folder");juce::ScopedJuceInitialiser_GUI juce;juce::File out(argv[1]);out.createDirectory();
    juce::AudioBuffer<float> b(2,48000*5);b.clear();for(int k=0;k<5;++k)for(int i=0;i<18000;++i){float v=float(.5*std::pow(1-i/18000.,2)*std::sin(2*juce::MathConstants<double>::pi*235*i/48000));for(int ch=0;ch<2;++ch)b.setSample(ch,k*40000+12000+i,v);}
    auto input=out.getChildFile("editing-fixture.wav");input.deleteFile();auto stream=input.createOutputStream();juce::WavAudioFormat fmt;std::unique_ptr<juce::AudioFormatWriter> writer(fmt.createWriterFor(stream.release(),48000,2,24,{},0));check(writer&&writer->writeFromAudioSampleBuffer(b,0,b.getNumSamples()),"Fixture WAV failed");writer.reset();
    auto second=out.getChildFile("editing-fixture-2.wav");input.copyFileTo(second);
    auto rules=makeDefaultRulesForAction(ImportAction::SegmentLongFile);rules.joinSegmentationInputs=false;ImportPreviewComponent c({input,second},ImportAction::SegmentLongFile,rules,[](ImportRules){},[](juce::AudioBuffer<float>,double){},[]{},[](bool){});c.addToDesktop(juce::ComponentPeer::windowHasTitleBar);c.setTopLeftPosition(-16000,-16000);c.setVisible(true);pump(50);ImportPreviewTestAccess::run(c,out);return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
