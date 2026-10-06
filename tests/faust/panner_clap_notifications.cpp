#include <juce_audio_utils/juce_audio_utils.h>
#include <clap/clap.h>
#include <clap/ext/params.h>
#include <clap/ext/thread-check.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
extern "C" const clap_plugin_entry_t clap_entry;
extern "C" juce::AudioProcessor* za_test_latest_processor();
extern "C" void za_test_gfx_button(juce::AudioProcessor*,int,double);
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
struct Event{uint16_t type;uint32_t flags;clap_id id;double value;};static std::vector<Event> events;
static uint32_t none(const clap_input_events_t*){return 0;}static const clap_event_header_t* noEvent(const clap_input_events_t*,uint32_t){return nullptr;}
static bool collect(const clap_output_events_t*,const clap_event_header_t*h){if(h->type==CLAP_EVENT_PARAM_VALUE){auto*p=reinterpret_cast<const clap_event_param_value_t*>(h);require(p->note_id==-1&&p->port_index==-1&&p->channel==-1&&p->key==-1,"UI value was not global");events.push_back({h->type,h->flags,p->param_id,p->value});}else if(h->type==CLAP_EVENT_PARAM_GESTURE_BEGIN||h->type==CLAP_EVENT_PARAM_GESTURE_END){auto*p=reinterpret_cast<const clap_event_param_gesture_t*>(h);events.push_back({h->type,h->flags,p->param_id,0});}return true;}
static const clap_host_thread_check_t threads={[](const clap_host_t*){return true;},[](const clap_host_t*){return false;}};
static const clap_host_params_t paramsHost={[](const clap_host_t*,clap_param_rescan_flags){},[](const clap_host_t*,clap_id,clap_param_clear_flags){},[](const clap_host_t*){}};
static const void* extension(const clap_host_t*,const char* id){if(!std::strcmp(id,CLAP_EXT_THREAD_CHECK))return &threads;if(!std::strcmp(id,CLAP_EXT_PARAMS))return &paramsHost;return nullptr;}static void request(const clap_host_t*){}
int main()try{juce::ScopedJuceInitialiser_GUI gui;require(clap_entry.init("notification-test"),"Init");auto*f=static_cast<const clap_plugin_factory_t*>(clap_entry.get_factory(CLAP_PLUGIN_FACTORY_ID));clap_host_t host{CLAP_VERSION,nullptr,"Notification test","ZorakAudio","","0",extension,request,request,request};auto*p=f->create_plugin(f,&host,f->get_plugin_descriptor(f,0)->id);require(p&&p->init(p),"Factory init");auto* params=static_cast<const clap_plugin_params_t*>(p->get_extension(p,CLAP_EXT_PARAMS));clap_id model=CLAP_INVALID_ID;for(uint32_t i=0;i<params->count(p);++i){clap_param_info_t info{};params->get_info(p,i,&info);if(std::strstr(info.name,"Render Model"))model=info.id;}require(model!=CLAP_INVALID_ID,"Missing model");clap_input_events_t in{nullptr,none,noEvent};clap_output_events_t out{nullptr,collect};auto*processor=za_test_latest_processor();require(processor,"Missing processor");
 za_test_gfx_button(processor,36,1);params->flush(p,&in,&out);require(events.size()==3,"Button touched unrelated parameters or lost gesture");require(events[0].type==CLAP_EVENT_PARAM_GESTURE_BEGIN&&events[1].type==CLAP_EVENT_PARAM_VALUE&&events[2].type==CLAP_EVENT_PARAM_GESTURE_END,"Wrong event order");for(auto&e:events)require(e.id==model&&(e.flags&CLAP_EVENT_IS_LIVE),"Host did not receive the live touched parameter");require(events[1].value==1,"Wrong model value");events.clear();za_test_gfx_button(processor,36,1);params->flush(p,&in,&out);require(events.empty(),"Unchanged all-slider notification polluted last touched");p->destroy(p);clap_entry.deinit();std::cout<<"REAL CLAP WRAPPER PASS: one live begin/value/end sequence, global value, correct parameter, no unchanged-slider gestures\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
