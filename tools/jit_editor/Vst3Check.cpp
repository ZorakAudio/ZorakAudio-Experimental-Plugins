// Public VST3 ABI test: no JUCE linkage and no developer compiler dependency.
#define NOMINMAX
#include <windows.h>
#include "pluginterfaces/base/ipluginbase.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivsthostapplication.h"
#include "pluginterfaces/vst/ivstmessage.h"
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cmath>
using namespace Steinberg;using namespace Steinberg::Vst;
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
void pump(){MSG m;while(PeekMessageW(&m,nullptr,0,0,PM_REMOVE)){TranslateMessage(&m);DispatchMessageW(&m);}}
template<class T>bool same(const TUID id){return std::memcmp(id,T::iid.toTUID(),16)==0;}
struct Host:IHostApplication{
 tresult PLUGIN_API queryInterface(const TUID id,void** out)override{if(same<IHostApplication>(id)||same<FUnknown>(id)){*out=this;return kResultOk;}*out=nullptr;return kNoInterface;}
 uint32 PLUGIN_API addRef()override{return 1;}uint32 PLUGIN_API release()override{return 1;}
 tresult PLUGIN_API getName(String128 n)override{std::memset(n,0,sizeof(String128));n[0]='T';return kResultOk;}
 tresult PLUGIN_API createInstance(TUID,TUID,void** out)override{*out=nullptr;return kNoInterface;}
};
struct Handler:IComponentHandler{
 int32 flags=0;int restarts=0;
 tresult PLUGIN_API queryInterface(const TUID id,void** out)override{if(same<IComponentHandler>(id)||same<FUnknown>(id)){*out=this;return kResultOk;}*out=nullptr;return kNoInterface;}
 uint32 PLUGIN_API addRef()override{return 1;}uint32 PLUGIN_API release()override{return 1;}
 tresult PLUGIN_API beginEdit(ParamID)override{return kResultOk;}tresult PLUGIN_API performEdit(ParamID,ParamValue)override{return kResultOk;}tresult PLUGIN_API endEdit(ParamID)override{return kResultOk;}
 tresult PLUGIN_API restartComponent(int32 f)override{flags|=f;++restarts;return kResultOk;}
};
struct Stream:IBStream{
 std::string text;size_t pos=0;
 tresult PLUGIN_API queryInterface(const TUID id,void** out)override{if(same<IBStream>(id)||same<FUnknown>(id)){*out=this;return kResultOk;}*out=nullptr;return kNoInterface;}
 uint32 PLUGIN_API addRef()override{return 1;}uint32 PLUGIN_API release()override{return 1;}
 tresult PLUGIN_API read(void* b,int32 n,int32* got)override{auto count=std::min((size_t)n,text.size()-pos);std::memcpy(b,text.data()+pos,count);pos+=count;if(got)*got=(int32)count;return kResultOk;}
 tresult PLUGIN_API write(void* b,int32 n,int32* got)override{text.append((char*)b,n);if(got)*got=n;return kResultOk;}
 tresult PLUGIN_API seek(int64 n,int32 mode,int64* result)override{int64 target=(mode==kIBSeekSet?0:mode==kIBSeekCur?(int64)pos:(int64)text.size())+n;pos=(size_t)std::max<int64>(0,std::min<int64>(target,text.size()));if(result)*result=pos;return kResultOk;}
 tresult PLUGIN_API tell(int64* result)override{*result=pos;return kResultOk;}
};
std::string quote(const std::string& t){std::string r="\"";for(char c:t){if(c=='\n')r+="\\n";else{if(c=='\"'||c=='\\')r+='\\';r+=c;}}return r+'\"';}
int main(int argc,char** argv)try{
 require(argc==2,"Pass VST3 binary");auto module=LoadLibraryA(argv[1]);require(module,"Load VST3");
 auto init=(bool(*)())GetProcAddress(module,"InitDll");if(init)require(init(),"InitDll");
 auto getFactory=(IPluginFactory*(*)())GetProcAddress(module,"GetPluginFactory");require(getFactory,"Factory export");auto* factory=getFactory();
 PClassInfo ci{};require(factory->getClassInfo(0,&ci)==kResultOk,"Class info");IComponent* component=nullptr;
 require(factory->createInstance(ci.cid,IComponent::iid.toTUID(),(void**)&component)==kResultOk,"Component");
 Host host;Handler handler;require(component->initialize(&host)==kResultOk,"Initialize component");
 TUID cid{};component->getControllerClassId(cid);IEditController* controller=nullptr;require(factory->createInstance(cid,IEditController::iid.toTUID(),(void**)&controller)==kResultOk,"Controller");
 require(controller->initialize(&host)==kResultOk,"Initialize controller");controller->setComponentHandler(&handler);
 IConnectionPoint *cp=nullptr,*ep=nullptr;component->queryInterface(IConnectionPoint::iid.toTUID(),(void**)&cp);controller->queryInterface(IConnectionPoint::iid.toTUID(),(void**)&ep);require(cp&&ep,"Connection points");cp->connect(ep);ep->connect(cp);
 IAudioProcessor* audio=nullptr;component->queryInterface(IAudioProcessor::iid.toTUID(),(void**)&audio);require(audio,"Audio processor");
 ProcessSetup setup{kRealtime,kSample32,256,48000};require(audio->setupProcessing(setup)==kResultOk,"Processing setup");component->activateBus(kAudio,kInput,0,true);component->activateBus(kAudio,kOutput,0,true);SpeakerArrangement initial=SpeakerArr::kStereo;audio->setBusArrangements(&initial,1,&initial,1);component->setActive(true);audio->setProcessing(true);
 int inputChannels=2,outputChannels=2;std::vector<std::vector<float>> input(64,std::vector<float>(256)),output=input;float* in[64];float* out[64];for(int i=0;i<64;++i){in[i]=input[i].data();out[i]=output[i].data();}
 AudioBusBuffers ib{},ob{};ib.channelBuffers32=in;ob.channelBuffers32=out;ProcessData data{};data.processMode=kRealtime;data.symbolicSampleSize=kSample32;data.numSamples=256;data.numInputs=data.numOutputs=1;data.inputs=&ib;data.outputs=&ob;
 auto block=[&]{pump();if(handler.flags&kIoChanged){handler.flags=0;audio->setProcessing(false);component->setActive(false);pump();BusInfo bi{};component->getBusInfo(kAudio,kInput,0,bi);inputChannels=bi.channelCount;component->getBusInfo(kAudio,kOutput,0,bi);outputChannels=bi.channelCount;component->setActive(true);audio->setProcessing(true);}
  handler.flags=0;ib.numChannels=inputChannels;ob.numChannels=outputChannels;for(int i=0;i<256;++i){input[0][i]=.8f;input[1][i]=-.4f;}require(audio->process(data)==kResultOk,"Process");};
 block();require(std::abs(output[0][0]-.8f)<2e-6,"Factory passthrough");
 auto run=[&](const std::string& source,float gain,int controls,int channels){Stream state;state.text="{\"appliedFrontend\":\"cpp-frontend\",\"frontend\":\"cpp-frontend\",\"version\":2,\"appliedMode\":\"jsfx\",\"appliedSource\":"+quote(source)+"}";require(component->setState(&state)==kResultOk,"Load source");auto start=std::chrono::steady_clock::now();while(std::chrono::steady_clock::now()-start<std::chrono::seconds(45)){block();if(std::abs(output[0][0]-.8f*gain)<2e-6 && controller->getParameterCount()==controls+2 && outputChannels==channels)return;std::this_thread::sleep_for(std::chrono::milliseconds(5));}Stream diagnostic;component->getState(&diagnostic);std::cerr<<"params="<<controller->getParameterCount()<<" channels="<<outputChannels<<" restarts="<<handler.restarts<<" output="<<output[0][0]<<" state="<<diagnostic.text.substr(0,3000)<<"\n";throw std::runtime_error("VST3 compile/reconfiguration timed out");};
 run("slider1:gain=0.5<0,1,0.01>Gain\n@sample\nspl0*=gain;spl1*=gain;",.5f,1,2);
 ParameterInfo info{};controller->getParameterInfo(0,info);require(info.title[0]=='G',"Dynamic parameter name");
 run("@faust block\ng=hslider(\"Gain\",.5,0,1,.01); b=hslider(\"Binary\",0,0,1,1); n=nentry(\"Entry\",0,0,1,1); e=checkbox(\"Enable\"); t=button(\"Trigger\"); process=*(g+b+n+e+t),*(g+b+n+e+t);",.5f,5,2);
 for(int i=0;i<5;++i){ParameterInfo binary{};require(controller->getParameterInfo(i,binary)==kResultOk,"Faust parameter info");require((binary.stepCount==1)==(binary.title[0]!='G'),"Binary Faust parameters boolean; continuous Gain keeps its range");}
 run("@sample\nspl0*=0.25;spl1*=0.25;spl2*=0.25;spl3*=0.25;",.25f,0,4);
 run("@sample\nspl0*=0.5;",.5f,0,1);run("@sample\nspl0*=0.125;spl1*=0.125;",.125f,0,2);require(handler.restarts>=3,"Host restarts");
 audio->setProcessing(false);component->setActive(false);cp->disconnect(ep);ep->disconnect(cp);cp->release();ep->release();audio->release();controller->terminate();controller->release();component->terminate();component->release();factory->release();auto exit=(bool(*)())GetProcAddress(module,"ExitDll");if(exit)exit();FreeLibrary(module);
 std::cout<<"PACKAGED VST3 PASS: dynamic controls, aliases, binary/continuous Faust parameters, audio, inferred stereo/four-channel pins, host restart\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
