#include <juce_audio_utils/juce_audio_utils.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "JsfxNativeDefaults.h"
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static void pump(int ms){juce::MessageManager::getInstance()->runDispatchLoopUntil(ms);}
static juce::Component* canvas(juce::Component& c){if(c.getName()=="Native JSFX graphics"||c.getName()=="EEL JSFX graphics")return &c;for(auto* child:c.getChildren())if(auto* found=canvas(*child))return found;return nullptr;}
static juce::MouseEvent event(juce::Component& c,juce::Point<float> p,int flags){auto now=juce::Time::getCurrentTime();return {juce::Desktop::getInstance().getMainMouseSource(),p,juce::ModifierKeys(flags),1,0,0,0,0,&c,&c,now,p,now,1,false};}
static void click(juce::Component& c,juce::Point<float> p,int flags=juce::ModifierKeys::leftButtonModifier){c.mouseDown(event(c,p,flags));pump(120);c.mouseUp(event(c,p,flags));pump(150);}
static juce::RangedAudioParameter* parameter(juce::AudioProcessor& p,int id){for(auto* raw:p.getParameters())if(auto* x=dynamic_cast<juce::RangedAudioParameter*>(raw))if(x->paramID=="slider"+juce::String(id))return x;throw std::runtime_error("Missing slider "+std::to_string(id));}
static void set(juce::AudioProcessor& p,int id,float value){auto* x=parameter(p,id);x->setValueNotifyingHost(x->convertTo0to1(value));}
static float get(juce::AudioProcessor& p,int id){auto* x=parameter(p,id);return x->convertFrom0to1(x->getValue());}
static void save(juce::Component& c,const juce::File& dir,const char* name){auto stream=dir.getChildFile(name).createOutputStream();require(stream&&stream->setPosition(0),"Screenshot output");stream->truncate();juce::PNGImageFormat png;require(png.writeImageToStream(c.createComponentSnapshot(c.getLocalBounds()),*stream),"Screenshot encoding");}

static juce::MemoryBlock withoutNewControls(const juce::MemoryBlock& saved){
 auto xml=juce::AudioProcessor::getXmlFromBinary(saved.getData(),int(saved.getSize()));require(bool(xml),"State XML");auto tree=juce::ValueTree::fromXml(*xml);
 for(int id:{46,47,48,49,50,51,52}){auto node=tree.getChildWithProperty("id","slider"+juce::String(id));if(node.isValid())tree.removeChild(node,nullptr);}
 juce::MemoryBlock legacy;auto result=tree.createXml();juce::AudioProcessor::copyXmlToBinary(*result,legacy);return legacy;
}
static std::unique_ptr<juce::AudioProcessor> makeProcessor(){auto p=std::unique_ptr<juce::AudioProcessor>(createPluginFilter());p->setNonRealtime(true);p->setRateAndBufferSizeDetails(48000,256);set(*p,7,1);set(*p,22,2);return p;}
static void prepare(juce::AudioProcessor& p){p.prepareToPlay(48000,256);}
static void render(juce::AudioProcessor& p,juce::AudioBuffer<float>& audio){
 for(int i=0;i<audio.getNumSamples();++i)for(int c=0;c<2;++c)audio.setSample(c,i,float(.7*std::sin(i*.17+c*.31)));
 juce::MidiBuffer midi;p.processBlock(audio,midi);
}
static void requireSame(const juce::AudioBuffer<float>& a,const juce::AudioBuffer<float>& b,const char* message){
 for(int c=0;c<2;++c)for(int i=0;i<a.getNumSamples();++i)require(a.getSample(c,i)==b.getSample(c,i),message);
}

static void displayUnits(){
 auto feet=makeProcessor(),metric=makeProcessor();require(get(*feet,48)==0,"Display must default to feet");set(*metric,48,1);prepare(*feet);prepare(*metric);
 juce::AudioBuffer<float> a(2,256),b(2,256);
 for(int block=0;block<96;++block){
  if(block%12==0)set(*metric,48,get(*metric,48)<.5?1:0);
  if(block%24==0){set(*feet,22,float((block/24)%4));set(*metric,22,float((block/24)%4));}
  if(block%16==0){float distance=float((block/16)%5)*.25f;set(*feet,7,distance);set(*metric,7,distance);}
  render(*feet,a);render(*metric,b);requireSame(a,b,"Unit switches must leave audio bit-identical");
 }
 set(*metric,48,1);set(*metric,46,.42);juce::MemoryBlock saved;metric->getStateInformation(saved);
 auto reopened=makeProcessor();reopened->setStateInformation(saved.getData(),int(saved.getSize()));require(get(*reopened,48)==1,"Metric preference did not persist");
 auto xml=juce::AudioProcessor::getXmlFromBinary(saved.getData(),int(saved.getSize()));auto tree=juce::ValueTree::fromXml(*xml);
 tree.removeChild(tree.getChildWithProperty("id","slider48"),nullptr);auto olderXml=tree.createXml();juce::MemoryBlock older;juce::AudioProcessor::copyXmlToBinary(*olderXml,older);
 reopened->setStateInformation(older.getData(),int(older.getSize()));
 require(get(*reopened,48)==0&&std::abs(get(*reopened,46)-.42)<.001,"Old units state must default to feet while preserving Proximity");
 auto oldTree=juce::ValueTree::fromXml(*juce::AudioProcessor::getXmlFromBinary(older.getData(),int(older.getSize())));
 juce::MemoryBlock restored;reopened->getStateInformation(restored);auto restoredTree=juce::ValueTree::fromXml(*juce::AudioProcessor::getXmlFromBinary(restored.getData(),int(restored.getSize())));
 for(const auto& child:oldTree){if(!child.hasType("PARAM"))continue;auto node=restoredTree.getChildWithProperty("id",child.getProperty("id"));require(node.isValid(),"Old parameter lost");require(std::abs(double(node.getProperty("value"))-double(child.getProperty("value")))<1e-6,"Unit preference changed a saved parameter");}
 std::cout<<"DISPLAY UNITS PASS: feet default, metric save/restore, old-state feet fallback, all prior saved parameter values preserved, bit-identical audio across live unit changes and all four source modes\n";
}

static void perspectiveCompatibility(){
 auto factory=makeProcessor();require(get(*factory,49)==0&&get(*factory,50)==0&&get(*factory,51)==0&&get(*factory,52)==.5,"Neutral perspective defaults");
 for(int oldProximity:{0,1}){
  auto reference=makeProcessor(),live=makeProcessor();float amount=oldProximity?.75f:0.f;
  for(auto* p:{reference.get(),live.get()}){set(*p,7,oldProximity?.25f:1.f);set(*p,46,amount);}
  juce::MemoryBlock saved;reference->getStateInformation(saved);
  auto xml=juce::AudioProcessor::getXmlFromBinary(saved.getData(),int(saved.getSize()));auto tree=juce::ValueTree::fromXml(*xml);
  for(int id:{49,50,51,52})tree.removeChild(tree.getChildWithProperty("id","slider"+juce::String(id)),nullptr);
  juce::MemoryBlock older;auto result=tree.createXml();juce::AudioProcessor::copyXmlToBinary(*result,older);
  auto cold=makeProcessor();cold->setStateInformation(older.getData(),int(older.getSize()));
  require(get(*cold,50)==0&&get(*cold,51)==0&&get(*cold,52)==.5,"Old perspective state before prepare");
  prepare(*cold);prepare(*reference);juce::AudioBuffer<float> a(2,256),b(2,256);
  for(int n=0;n<12;++n){render(*cold,a);render(*reference,b);requireSame(a,b,"Old perspective state cold audio differs");}
  // Advance a matching reference while the other instance uses both new paths.
  reference->releaseResources();reference=makeProcessor();set(*reference,7,oldProximity?.25f:1.f);set(*reference,46,amount);prepare(*reference);
  set(*live,49,180);set(*live,50,1);set(*live,51,1);set(*live,52,.9);prepare(*live);
  for(int n=0;n<50;++n){render(*live,a);render(*reference,b);}
  live->setStateInformation(older.getData(),int(older.getSize()));
  require(get(*live,49)==0&&get(*live,50)==0&&get(*live,51)==0&&get(*live,52)==.5,"Older state inherited active perspective controls");
  render(*live,a);render(*reference,b);requireSame(a,b,"Older default/legacy perspective must bypass from first block");
 }
 set(*factory,49,145);set(*factory,50,.8);set(*factory,51,.7);set(*factory,52,.9);juce::MemoryBlock custom;factory->getStateInformation(custom);
 auto restored=makeProcessor();restored->setStateInformation(custom.getData(),int(custom.getSize()));
 require(std::abs(get(*restored,49)-145)<.01&&std::abs(get(*restored,50)-.8)<.001&&std::abs(get(*restored,51)-.7)<.001&&std::abs(get(*restored,52)-.9)<.001,"Custom perspectives did not persist");
 std::cout<<"PERSPECTIVE STATE PASS: neutral defaults, older cold/default/legacy load, active-instance reset from first block, custom values preserved\n";
}

static void compatibility(){
 auto factory=makeProcessor();require(std::abs(get(*factory,46)-.75)<.001,"Fresh instance must enable Proximity");
 auto* parameter46=parameter(*factory,46);require(std::abs(parameter46->convertFrom0to1(parameter46->getDefaultValue())-.75)<.001,"Host factory default");
 set(*factory,46,0);juce::MemoryBlock savedOff;factory->getStateInformation(savedOff);auto legacy=withoutNewControls(savedOff);
 auto reference=makeProcessor();set(*reference,46,0);prepare(*reference);
 auto restored=makeProcessor();restored->setStateInformation(legacy.getData(),int(legacy.getSize()));
 require(get(*restored,46)==0&&std::abs(get(*restored,47)-.65)<.001,"Legacy load before prepare");prepare(*restored);
 juce::AudioBuffer<float> a(2,256),b(2,256);
 for(int n=0;n<12;++n){render(*reference,a);render(*restored,b);requireSame(a,b,"Legacy restore must render exact legacy audio from first block");}
 auto live=makeProcessor();prepare(*live);auto liveReference=makeProcessor();set(*liveReference,46,0);prepare(*liveReference);
 bool activeDifference=false;for(int n=0;n<50;++n){render(*live,a);render(*liveReference,b);for(int c=0;c<2;++c)for(int i=0;i<256;++i)activeDifference=activeDifference||std::abs(a.getSample(c,i)-b.getSample(c,i))>1e-4;}require(activeDifference,"Legacy restore fixture did not exercise active proximity processing");
 set(*live,47,.12);live->setStateInformation(legacy.getData(),int(legacy.getSize()));
 require(get(*live,46)==0&&std::abs(get(*live,47)-.65)<.001,"Legacy must override active current values");
 render(*live,a);render(*liveReference,b);requireSame(a,b,"Legacy load over active instance must bypass from first block");
 juce::MemoryBlock upgraded;live->getStateInformation(upgraded);auto reopened=makeProcessor();reopened->setStateInformation(upgraded.getData(),int(upgraded.getSize()));require(get(*reopened,46)==0,"Legacy resave must persist zero");
 set(*live,46,.75);activeDifference=false;for(int n=0;n<50;++n){render(*live,a);render(*liveReference,b);for(int c=0;c<2;++c)for(int i=0;i<256;++i)activeDifference=activeDifference||std::abs(a.getSample(c,i)-b.getSample(c,i))>1e-4;}require(activeDifference,"Legacy restore fixture did not exercise active proximity processing");
 live->setStateInformation(upgraded.getData(),int(upgraded.getSize()));render(*live,a);render(*liveReference,b);
 requireSame(a,b,"Resaved legacy Off must bypass from first block over active instance");
 set(*reopened,46,.9);reopened->setStateInformation(savedOff.getData(),int(savedOff.getSize()));require(get(*reopened,46)==0,"Explicit saved Off must survive factory change");
 set(*factory,46,.42);set(*factory,47,.31);juce::MemoryBlock savedCustom;factory->getStateInformation(savedCustom);
 reopened->setStateInformation(savedCustom.getData(),int(savedCustom.getSize()));require(std::abs(get(*reopened,46)-.42)<.001&&std::abs(get(*reopened,47)-.31)<.001,"Explicit saved values must remain exact");
 // Metadata is opt-in: unrelated JSFX declarations and state remain untouched.
 const char* plain="slider1:0.2<0,1,0.001>Amount\n";auto declarations=parseJsfxSliderDecls(plain,nullptr);auto rules=za::jsfx::parseNativeSliderDefaults(plain,declarations);
 require(rules.empty(),"Unmarked source opted in");za::jsfx::applyNativeSliderDefaults(rules,declarations);require(std::abs(declarations[0].def-.2)<.001,"Unmarked default changed");
 juce::ValueTree unrelated("PARAMS");require(!za::jsfx::restoreMissingNativeSliderDefaults(rules,unrelated)&&unrelated.getNumChildren()==0,"Unmarked state changed");
 std::cout<<"STATE COMPATIBILITY PASS: factory 75%, pre-feature load before prepare and over active instance, exact legacy audio from first block, resave/reopen, explicit Off/custom values, unmarked plugins unchanged\n";
}

int main(int argc,char** argv)try{
 require(argc==2,"Pass screenshot directory");juce::ScopedJuceInitialiser_GUI gui;juce::File dir(juce::String::fromUTF8(argv[1]));dir.createDirectory();compatibility();displayUnits();perspectiveCompatibility();
 auto p=std::unique_ptr<juce::AudioProcessor>(createPluginFilter());p->setNonRealtime(true);p->setRateAndBufferSizeDetails(48000,256);p->prepareToPlay(48000,256);
 require(std::abs(get(*p,46)-.75)<.001&&std::abs(get(*p,47)-.65)<.001,"New defaults");set(*p,46,.75);set(*p,47,.65);set(*p,7,1);
 juce::AudioBuffer<float> audio(2,256);juce::MidiBuffer midi;
 auto process=[&](){for(int i=0;i<256;++i)for(int c=0;c<2;++c)audio.setSample(c,i,float(.7*std::sin(i*.17)));midi.clear();p->processBlock(audio,midi);for(int c=0;c<2;++c)for(int i=0;i<256;++i)require(std::isfinite(audio.getSample(c,i)),"Nonfinite native output");};
 for(int i=0;i<30;++i)process();
 juce::MemoryBlock state;p->getStateInformation(state);set(*p,46,0);set(*p,47,0);p->setStateInformation(state.getData(),int(state.getSize()));
 require(std::abs(get(*p,46)-.75)<.001&&std::abs(get(*p,47)-.65)<.001,"State restore lost new parameters");
 auto editor=std::unique_ptr<juce::AudioProcessorEditor>(p->createEditor());require(editor!=nullptr,"Editor creation");editor->setTopLeftPosition(-10000,-10000);editor->addToDesktop(juce::ComponentPeer::windowIsTemporary);editor->setVisible(true);pump(500);
 auto* view=canvas(*editor);require(view,"Graphics canvas");const float w=float(view->getWidth()),h=float(view->getHeight());const float scale=juce::jlimit(.56f,1.65f,std::min(w/820,h/500));
 const float pad=std::max(10.f,16*scale),canvasY=std::max(36.f,48*scale)+pad;
 const juce::Point<float> units(w-pad-(148+8+38)*scale,24*scale);
 require(get(*p,48)==0,"Feet UI default");
 view->mouseMove(event(*view,{20,20},0));pump(200);
 const juce::Rectangle<int> unitsRect(int(units.x-30*scale),int(units.y-8*scale),int(60*scale),int(16*scale));
 auto header=[&](){return view->createComponentSnapshot(unitsRect);};
 auto differences=[](const juce::Image& a,const juce::Image& b){int n=0;for(int y=0;y<a.getHeight();++y)for(int x=0;x<a.getWidth();++x)n+=a.getPixelAt(x,y)!=b.getPixelAt(x,y);return n;};
 auto feetHeader=header();

 const float beforeDistance=get(*p,7),beforeLateral=get(*p,6),beforeDepth=get(*p,8),beforeRoomHeight=get(*p,44),beforeEarHeight=get(*p,45);
 click(*view,units);require(get(*p,48)==1,"Metric header click");view->mouseMove(event(*view,{20,20},0));for(int i=0;i<50;++i){process();pump(10);}require(get(*p,48)==1,"Metric selection reverted while rendering");require(differences(feetHeader,header())>5,"Metric text did not refresh in the rendered header");save(*editor,dir,"panner-metres.png");
 click(*view,units);require(get(*p,48)==0,"Feet header click");view->mouseMove(event(*view,{20,20},0));for(int i=0;i<50;++i){process();pump(10);}require(get(*p,48)==0,"Feet selection reverted while rendering");require(differences(feetHeader,header())==0,"Feet text did not return in the rendered header");save(*editor,dir,"panner-feet.png");
 click(*view,units);click(*view,units,juce::ModifierKeys::rightButtonModifier);require(get(*p,48)==0,"Units right-click feet reset");
 require(get(*p,7)==beforeDistance&&get(*p,6)==beforeLateral&&get(*p,8)==beforeDepth&&get(*p,44)==beforeRoomHeight&&get(*p,45)==beforeEarHeight,"Unit clicks changed positions or heights");

 click(*view,{w-19*scale,canvasY+240*scale});
 save(*editor,dir,"proximity-editor.png");
 const float drawerW=std::min(w*.55f,360*scale),drawerX=w-pad-drawerW,barX=drawerX+132*scale,barW=drawerW-211*scale;
 // Match the retained model drawer's row metrics at every supported size.
 const float fieldH=std::max(h-std::max(36.f,48*scale)-std::max(28.f,34*scale)-2*pad,230*scale);
 const float rowH=std::max(21*scale,(fieldH-108*scale)/10),startY=canvasY+63*scale;
 click(*view,{barX+barW*.80f,startY+rowH+10*scale});
 require(std::abs(get(*p,46)-.80)<.04,"Proximity canvas control");
 click(*view,{barX+barW*.30f,startY+2*rowH+10*scale});
 require(std::abs(get(*p,47)-.30)<.04,"Anti-Salience canvas control");
 juce::MouseWheelDetails wheel;wheel.deltaY=1;view->mouseWheelMove(event(*view,{barX+barW*.30f,startY+2*rowH+10*scale},0),wheel);pump(200);
 require(get(*p,47)>.31,"Anti-Salience wheel");
 click(*view,{barX+barW*.30f,startY+2*rowH+10*scale},juce::ModifierKeys::rightButtonModifier);
 require(std::abs(get(*p,47)-.65)<.001,"Anti-Salience reset");
 click(*view,{barX+barW*.80f,startY+rowH+10*scale},juce::ModifierKeys::rightButtonModifier);
 require(get(*p,46)==0,"Proximity legacy reset");set(*p,46,.75);for(int i=0;i<20;++i)process();view->mouseMove(event(*view,{20,20},0));pump(250);save(*editor,dir,"proximity-editor.png");
 // New Artistic rows follow the retained Proximity/Anti-Salience rows.
 auto rowPoint=[&](int row,float fraction){return juce::Point<float>(barX+barW*fraction,startY+row*rowH+10*scale);};
 click(*view,rowPoint(3,1));require(std::abs(get(*p,49)-180)<1,"Facing canvas control");
 click(*view,rowPoint(4,1));require(get(*p,50)>.99,"Directivity canvas control");
 view->mouseMove(event(*view,{20,20},0));for(int i=0;i<50;++i){process();pump(10);}save(*editor,dir,"panner-source-facing.png");
 click(*view,rowPoint(5,1));require(get(*p,51)>.99,"Internal POV canvas control");
 click(*view,rowPoint(6,.65));require(std::abs(get(*p,52)-.65)<.04,"Body Tone canvas control");
 view->mouseMove(event(*view,{20,20},0));for(int i=0;i<50;++i){process();pump(10);}save(*editor,dir,"panner-internal-pov.png");
 for(int row:{3,4,5,6})click(*view,rowPoint(row,.5),juce::ModifierKeys::rightButtonModifier);
 require(get(*p,49)==0&&get(*p,50)==0&&get(*p,51)==0&&get(*p,52)==.5,"Perspective right-click defaults");
 view->mouseWheelMove(event(*view,rowPoint(3,.5),0),wheel);pump(200);require(get(*p,49)>4,"Facing wheel control");
 std::cout<<"PERSPECTIVE UI PASS: facing/directivity/internal/body controls, wheel/reset, rendered screenshots\n";
 for(auto size:{juce::Point<int>(900,600),juce::Point<int>(1400,900)}){editor->setSize(size.x,size.y);pump(250);process();}
 p->editorBeingDeleted(editor.get());editor.reset();
 for(double rate:{44100.,96000.}){p->releaseResources();p->setRateAndBufferSizeDetails(rate,256);p->prepareToPlay(rate,256);process();}
 std::cout<<"NATIVE PASS: production rendering of feet/metres labels, defaults, stereo audio, state restore, editor render, Proximity/Anti-Salience clicks, wheel, reset, resize, reprepare\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
