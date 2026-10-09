// Load the real packaged CLAP; synthetic input only, no external audio files.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "../../../../tools/jit_editor/TestModule.h"
#include <clap/clap.h>
#include <clap/ext/audio-ports.h>
#include <clap/ext/gui.h>
#include <clap/ext/params.h>
#include <clap/ext/thread-check.h>
#include <clap/ext/timer-support.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static bool audioThread=false;
static const clap_host_thread_check_t threads{
    [](const clap_host_t*) {return !audioThread;}, [](const clap_host_t*) {return audioThread;}};
static std::vector<clap_id> timerIds;
static const clap_host_timer_support_t timers{
    [](const clap_host_t*,uint32_t,clap_id* id){*id=static_cast<clap_id>(timerIds.size()+1);timerIds.push_back(*id);return true;},
    [](const clap_host_t*,clap_id id){timerIds.erase(std::remove(timerIds.begin(),timerIds.end(),id),timerIds.end());return true;}};
static const clap_host_gui_t hostGui{[](const clap_host_t*){},
    [](const clap_host_t*,uint32_t,uint32_t){return true;},
    [](const clap_host_t*){return false;},[](const clap_host_t*){return true;},[](const clap_host_t*,bool){}};
static const void* extension(const clap_host_t*,const char* name){
    if(std::strcmp(name,CLAP_EXT_THREAD_CHECK)==0)return &threads;
    if(std::strcmp(name,CLAP_EXT_TIMER_SUPPORT)==0)return &timers;
    if(std::strcmp(name,CLAP_EXT_GUI)==0)return &hostGui;
    return nullptr;
}
static void request(const clap_host_t*){}
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static clap_event_param_value_t change{};
static bool changed=false;
static uint32_t eventsSize(const clap_input_events_t*){return changed?1:0;}
static const clap_event_header_t* eventsGet(const clap_input_events_t*,uint32_t index){return changed&&index==0?&change.header:nullptr;}

int main(int argc,char** argv) try {
    require(argc==2,"Pass the packaged AntiSalienceMX.clap path");
    auto path=testArgument(1,argv);auto* module=loadTestModule(path.c_str());require(module,"Load CLAP");
    auto* entry=static_cast<const clap_plugin_entry_t*>(testSymbol(module,"clap_entry"));require(entry&&entry->init(path.c_str()),"Entry init");
    auto* factory=static_cast<const clap_plugin_factory_t*>(entry->get_factory(CLAP_PLUGIN_FACTORY_ID));require(factory&&factory->get_plugin_count(factory)==1,"Factory count");
    auto* descriptor=factory->get_plugin_descriptor(factory,0);
    require(std::string(descriptor->id)=="com.zorakaudio.experimental.antisaliencemx","Independent product ID");
    clap_host_t host{CLAP_VERSION,nullptr,"Anti-Salience release check","ZorakAudio","","1",extension,request,request,request};
    auto* plugin=factory->create_plugin(factory,&host,descriptor->id);require(plugin&&plugin->init(plugin),"Plugin init");
    auto* ports=static_cast<const clap_plugin_audio_ports_t*>(plugin->get_extension(plugin,CLAP_EXT_AUDIO_PORTS));
    require(ports&&ports->count(plugin,false)==1,"One output bus");
    const auto inputBusCount=ports->count(plugin,true);
    clap_audio_port_info_t port{};
    unsigned totalInputs=0;
    std::vector<unsigned> busChannels;
    for(uint32_t i=0;i<inputBusCount;++i){require(ports->get(plugin,i,true,&port),"Input port info");busChannels.push_back(port.channel_count);totalInputs+=port.channel_count;}
    require(totalInputs==4,"Source and reference: four inputs");
    require(ports->get(plugin,0,false,&port)&&port.channel_count==2,"Stereo outputs");
    auto* params=static_cast<const clap_plugin_params_t*>(plugin->get_extension(plugin,CLAP_EXT_PARAMS));require(params,"Parameters");
    auto parameter=[&](const char* name,double expected){
        for(uint32_t i=0;i<params->count(plugin);++i){clap_param_info_t info{};require(params->get_info(plugin,i,&info),"Parameter info");
            if(std::string(info.name)==name){double value=0;require(params->get_value(plugin,info.id,&value),"Parameter value");
                require(std::abs(value-info.default_value)<1e-7,"Host parameter starts at its declared default");
                // JUCE's base CLAP wrapper exposes normalized values; validate
                // source units through its public text conversion, including
                // the half-life slider's logarithmic/skewed mapping.
                char text[256]{};require(params->value_to_text(plugin,info.id,value,text,sizeof(text)),"Parameter text");
                if(std::string(name)!="Bypass")require(std::abs(std::stod(text)-expected)<1e-7,"Source parameter default");
                return info.id;}}
        throw std::runtime_error(std::string("Missing parameter: ")+name);
    };
    auto scatter=parameter("Scatter Mix (%)",0);auto bypass=parameter("Bypass",0);parameter("Erasure (%)",68);parameter("Lock Half-Life (ms)",520);
    require(plugin->activate(plugin,48000,1,256),"Activate");audioThread=true;require(plugin->start_processing(plugin),"Start processing");audioThread=false;
    std::array<std::array<float,256>,4> inputs{};std::array<std::array<float,256>,2> outputs{};
    float* inputPointers[]{inputs[0].data(),inputs[1].data(),inputs[2].data(),inputs[3].data()};float* outputPointers[]{outputs[0].data(),outputs[1].data()};
    std::vector<clap_audio_buffer_t> inputBuses(inputBusCount);unsigned offset=0;
    for(uint32_t i=0;i<inputBusCount;++i){inputBuses[i].data32=inputPointers+offset;inputBuses[i].channel_count=busChannels[i];offset+=busChannels[i];}
    clap_audio_buffer_t output{};output.data32=outputPointers;output.channel_count=2;
    clap_input_events_t inputEvents{nullptr,eventsSize,eventsGet};clap_output_events_t outputEvents{nullptr,[](const clap_output_events_t*,const clap_event_header_t*){return true;}};
    clap_process_t process{};process.frames_count=256;process.audio_inputs=inputBuses.data();process.audio_inputs_count=inputBusCount;process.audio_outputs=&output;process.audio_outputs_count=1;process.in_events=&inputEvents;process.out_events=&outputEvents;
    auto set=[&](clap_id id,double value){change={};change.header={sizeof(change),0,CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_PARAM_VALUE,0};change.param_id=id;change.value=value;change.note_id=-1;change.port_index=change.channel=change.key=-1;changed=true;};
    auto* pluginTimers=static_cast<const clap_plugin_timer_support_t*>(plugin->get_extension(plugin,CLAP_EXT_TIMER_SUPPORT));
    double peak=0;
    for(int phase=0;phase<4;++phase){
        if(phase==1)set(scatter,1); // 100% in the public normalized CLAP range.
        if(phase==2)set(scatter,0);
        if(phase==3)set(bypass,1);
        for(int block=0;block<240;++block){
            for(int f=0;f<256;++f){double t=(process.steady_time+f)/48000.;
                inputs[0][f]=phase==3?1.3f:float(.7*std::sin(t*2*3.141592653589793*997));
                inputs[1][f]=phase==3?-.8f:float(.5*std::sin(t*2*3.141592653589793*317));
                inputs[2][f]=float(.3*std::sin(t*2*3.141592653589793*521));inputs[3][f]=inputs[2][f];}
            audioThread=true;require(plugin->process(plugin,&process)!=CLAP_PROCESS_ERROR,"Process");audioThread=false;changed=false;
            for(auto& channel:outputs)for(float value:channel){require(std::isfinite(value),"Finite output");peak=std::max(peak,double(std::abs(value)));if(phase<3)require(std::abs(value)<=.98001f,"Processed peak containment");}
            if(phase==3&&block>200)require(std::abs(outputs[0][0]-1.3f)<1e-5f&&std::abs(outputs[1][0]+.8f)<1e-5f,"Settled bypass passes uncontained dry input");
            process.steady_time+=256;pumpTestMessages();if(pluginTimers)for(auto id:timerIds)pluginTimers->on_timer(plugin,id);
        }
    }
    // Exercise the actual native editor lifecycle without showing a window.
    auto* gui=static_cast<const clap_plugin_gui_t*>(plugin->get_extension(plugin,CLAP_EXT_GUI));require(gui,"Custom GUI");
#ifdef _WIN32
    require(gui->is_api_supported(plugin,CLAP_WINDOW_API_WIN32,false)&&gui->create(plugin,CLAP_WINDOW_API_WIN32,false),"Create native GUI");
    uint32_t w=0,h=0;require(gui->get_size(plugin,&w,&h)&&w>500&&h>400,"Full graphics size");
    w=1280;h=980;require(gui->adjust_size(plugin,&w,&h)&&gui->set_size(plugin,w,h),"Resize GUI");pumpTestMessages();gui->destroy(plugin);
#endif
    audioThread=true;plugin->stop_processing(plugin);audioThread=false;plugin->deactivate(plugin);plugin->destroy(plugin);entry->deinit();closeTestModule(module);
    std::cout<<"PASS: actual CLAP identity, 4-in/2-out, source defaults, core/scatter transitions, finite/contained audio, smoothed dry bypass, native editor creation/resize; peak="<<peak<<"\n";
    return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
