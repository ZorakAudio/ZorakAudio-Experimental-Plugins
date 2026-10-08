#pragma once
#include "JsfxSliderDeclarations.h"
enum class JitControlKind { continuous, toggle, momentary };
inline JitControlKind jitControlKind(int slot,const juce::var& descriptor){
 auto controls=descriptor.getProperty("metadata",juce::var()).getProperty("editor_sliders",juce::var());
 if(auto* list=controls.getArray())for(auto& control:*list)if((int)control.getProperty("slot",-1)==slot){
  const auto type=control.getProperty("type","").toString();if(type=="button")return JitControlKind::momentary;
  if(type=="checkbox" || ((double)control.getProperty("minimum",0)==0 && (double)control.getProperty("maximum",1)==1 && (double)control.getProperty("step",0)>=1))return JitControlKind::toggle;
 }
 return JitControlKind::continuous;
}
inline std::vector<JsfxSliderDecl> jitDeclarations(const juce::String& source,const juce::var& descriptor){
 auto result=parseJsfxSliderDecls(descriptor.getProperty("expandedSource",source).toString().toRawUTF8());
 auto controls=descriptor.getProperty("metadata",juce::var()).getProperty("editor_sliders",juce::var());
 if(auto* list=controls.getArray())for(auto& v:*list){JsfxSliderDecl d;d.index0=(int)v.getProperty("slot",0);d.id=v.getProperty("id","").toString();d.name=v.getProperty("label","").toString();d.rangeStart=d.min=(float)v.getProperty("minimum",0);d.rangeEnd=d.max=(float)v.getProperty("maximum",1);d.def=(float)v.getProperty("default",0);d.step=(float)v.getProperty("step",0);result.push_back(d);}
 return result;
}
