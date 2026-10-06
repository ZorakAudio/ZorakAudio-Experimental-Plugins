// Smoke-test packaged CLAP modules with generated input; opens no recordings.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <clap/clap.h>
#include <clap/ext/params.h>
#include <clap/ext/thread-check.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>
static bool audioThread=false;
static const clap_host_thread_check_t threads={[](const clap_host_t*){return !audioThread;},[](const clap_host_t*){return audioThread;}};
static const void* hostExtension(const clap_host_t*,const char* id){return std::strcmp(id,CLAP_EXT_THREAD_CHECK)==0?&threads:nullptr;}
static void request(const clap_host_t*){}
static uint32_t noEvents(const clap_input_events_t*){return 0;}
static const clap_event_header_t* noEvent(const clap_input_events_t*,uint32_t){return nullptr;}
static bool acceptEvent(const clap_output_events_t*,const clap_event_header_t*){return true;}
static const clap_event_param_value_t* currentEvent=nullptr;
static uint32_t oneEvent(const clap_input_events_t*){return 1;}
static const clap_event_header_t* getEvent(const clap_input_events_t*,uint32_t){return &currentEvent->header;}
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv)try{
 require(argc==2,"Pass the packaged CLAP module path");HMODULE module=LoadLibraryA(argv[1]);require(module!=nullptr,"Packaged module could not load");
 auto* entry=reinterpret_cast<const clap_plugin_entry_t*>(GetProcAddress(module,"clap_entry"));require(entry&&entry->init(argv[1]),"CLAP entry init");
 auto* factory=static_cast<const clap_plugin_factory_t*>(entry->get_factory(CLAP_PLUGIN_FACTORY_ID));require(factory&&factory->get_plugin_count(factory)==1,"CLAP factory");auto* descriptor=factory->get_plugin_descriptor(factory,0);
 clap_host_t host{CLAP_VERSION,nullptr,"Panner package smoke","ZorakAudio","","0",hostExtension,request,request,request};auto* plugin=factory->create_plugin(factory,&host,descriptor->id);require(plugin&&plugin->init(plugin),"CLAP plugin init");
 auto* params=static_cast<const clap_plugin_params_t*>(plugin->get_extension(plugin,CLAP_EXT_PARAMS));require(params,"CLAP params");clap_param_info_t model{};bool found=false;
 for(uint32_t i=0;i<params->count(plugin);++i){clap_param_info_t info{};require(params->get_info(plugin,i,&info),"Parameter info");if(std::strstr(info.name,"Manifold")){model=info;found=true;break;}}
 require(found,"Manifold parameter not exposed");clap_event_param_value_t event{};event.header={sizeof(event),0,CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_PARAM_VALUE,0};event.param_id=model.id;event.cookie=model.cookie;event.note_id=-1;event.port_index=-1;event.channel=-1;event.key=-1;event.value=model.min_value+.85*(model.max_value-model.min_value);
 currentEvent=&event;clap_input_events_t change{nullptr,oneEvent,getEvent};clap_output_events_t eventsOut{nullptr,acceptEvent};params->flush(plugin,&change,&eventsOut);
 double modelValue=0;require(params->get_value(plugin,model.id,&modelValue)&&std::abs(modelValue-event.value)<1e-4,"CMD Flow parameter did not update");
 require(plugin->activate(plugin,48000,1,256),"CLAP activation");audioThread=true;require(plugin->start_processing(plugin),"CLAP start processing");
 float left[256]{},right[256]{},outputL[256]{},outputR[256]{};float* inputChannels[]={left,right};float* outputChannels[]={outputL,outputR};clap_audio_buffer_t input{},output{};input.data32=inputChannels;input.channel_count=2;output.data32=outputChannels;output.channel_count=2;
 clap_input_events_t eventsIn{nullptr,noEvents,noEvent};clap_process_t process{};process.frames_count=256;process.audio_inputs=&input;process.audio_outputs=&output;process.audio_inputs_count=1;process.audio_outputs_count=1;process.in_events=&eventsIn;process.out_events=&eventsOut;
 double peak=0,energy=0;for(int block=0;block<188;++block){for(int k=0;k<256;++k)left[k]=right[k]=float(.05*std::sin(2*3.141592653589793*700*(block*256+k)/48000));process.steady_time=block*256;require(plugin->process(plugin,&process)!=CLAP_PROCESS_ERROR,"CLAP process error");for(int ch=0;ch<2;++ch)for(int k=0;k<256;++k){double x=outputChannels[ch][k];require(std::isfinite(x),"Nonfinite packaged audio");peak=std::max(peak,std::abs(x));energy+=x*x;}}
 require(peak>0&&peak<4&&energy>0,"Packaged audio magnitude");plugin->stop_processing(plugin);audioThread=false;plugin->deactivate(plugin);plugin->destroy(plugin);entry->deinit();FreeLibrary(module);
 std::cout<<"PACKAGED CLAP PASS: "<<argv[1]<<"; CMD Flow; finite nonzero audio; peak "<<peak<<"; energy "<<energy<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
