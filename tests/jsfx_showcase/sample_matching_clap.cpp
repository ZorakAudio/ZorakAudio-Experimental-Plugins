#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <clap/clap.h>
#include <clap/ext/state.h>
#include <clap/ext/params.h>
#include <clap/ext/thread-check.h>
#include <clap/ext/timer-support.h>
#include <clap/ext/render.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static bool audioThread=false, callbackPending=false, restartPending=false;
static const clap_plugin_t* current=nullptr;
static bool timerRegistered=false;
static clap_id timerId=1;
static const clap_host_thread_check_t threads={[](const clap_host_t*){return !audioThread;},[](const clap_host_t*){return audioThread;}};
static const clap_host_timer_support_t timers={[](const clap_host_t*,uint32_t,clap_id* id){*id=timerId;timerRegistered=true;return true;},[](const clap_host_t*,clap_id){timerRegistered=false;return true;}};
static const void* extension(const clap_host_t*,const char* id){if(!std::strcmp(id,CLAP_EXT_THREAD_CHECK))return &threads;if(!std::strcmp(id,CLAP_EXT_TIMER_SUPPORT))return &timers;return nullptr;}
static void request(const clap_host_t*){}
static void callback(const clap_host_t*){callbackPending=true;}
static void restart(const clap_host_t*){restartPending=true;}
static void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
struct Stream{std::string data;size_t pos=0;};
static int64_t read(const clap_istream_t* s,void* dst,uint64_t n){auto& v=*static_cast<Stream*>(s->ctx);auto c=std::min<size_t>(n,v.data.size()-v.pos);std::memcpy(dst,v.data.data()+v.pos,c);v.pos+=c;return c;}
static int64_t write(const clap_ostream_t* s,const void* src,uint64_t n){static_cast<Stream*>(s->ctx)->data.append(static_cast<const char*>(src),n);return n;}
static std::string file(const char* p){std::ifstream f(p,std::ios::binary);require(bool(f),std::string("Read ")+p);return {std::istreambuf_iterator<char>(f),{}};}
static void pump(){MSG msg;while(PeekMessage(&msg,nullptr,0,0,PM_REMOVE)){TranslateMessage(&msg);DispatchMessage(&msg);}if(callbackPending){callbackPending=false;current->on_main_thread(current);}if(timerRegistered){auto* t=static_cast<const clap_plugin_timer_support_t*>(current->get_extension(current,CLAP_EXT_TIMER_SUPPORT));if(t)t->on_timer(current,timerId);}}
static std::vector<clap_event_midi_t> midi;
static std::vector<clap_event_param_value_t> parameterEvents;
static uint32_t eventCount(const clap_input_events_t*){return uint32_t(parameterEvents.size()+midi.size());}
static const clap_event_header_t* eventGet(const clap_input_events_t*,uint32_t n){return n<parameterEvents.size()?&parameterEvents[n].header:&midi.at(n-parameterEvents.size()).header;}
static bool eventOut(const clap_output_events_t*,const clap_event_header_t*){return true;}
static void addMidi(int time,int a,int b,int c){clap_event_midi_t e{};e.header={sizeof(e),uint32_t(time),CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_MIDI,0};e.port_index=0;e.data[0]=a;e.data[1]=b;e.data[2]=c;midi.push_back(e);}
int main(int argc,char** argv)try{
 require(argc>=4,"inspect MODULE PREFIX | render MODULE STATE PREFIX RATE BLOCK DURATION PATTERN");
 const std::string cmd=argv[1],modulePath=argv[2];auto module=LoadLibraryA(modulePath.c_str());require(module,"Load module");
 auto* entry=reinterpret_cast<const clap_plugin_entry_t*>(GetProcAddress(module,"clap_entry"));require(entry&&entry->init(modulePath.c_str()),"Entry init");
 auto* factory=static_cast<const clap_plugin_factory_t*>(entry->get_factory(CLAP_PLUGIN_FACTORY_ID));require(factory,"Factory");auto* desc=factory->get_plugin_descriptor(factory,0);
 clap_host_t host{CLAP_VERSION,nullptr,"Sample Raw Tape null test","Local test","","1",extension,restart,request,callback};
 auto* p=factory->create_plugin(factory,&host,desc->id);require(p&&p->init(p),"Plugin init");current=p;
 auto* state=static_cast<const clap_plugin_state_t*>(p->get_extension(p,CLAP_EXT_STATE));auto* params=static_cast<const clap_plugin_params_t*>(p->get_extension(p,CLAP_EXT_PARAMS));require(state&&params,"State and params");
 auto save=[&](const std::string& path){Stream s;clap_ostream_t out{&s,write};require(state->save(p,&out),"Save state");std::ofstream f(path,std::ios::binary);f.write(s.data.data(),s.data.size());};
 if(cmd=="inspect"){
  save(std::string(argv[3])+".state");std::ofstream f(std::string(argv[3])+".params.tsv");f<<"index\tid\tname\tminimum\tmaximum\tdefault\tcurrent\n";
  for(uint32_t i=0;i<params->count(p);++i){clap_param_info_t info{};require(params->get_info(p,i,&info),"Param info");double v=0;params->get_value(p,info.id,&v);f<<i<<'\t'<<info.id<<'\t'<<info.name<<'\t'<<info.min_value<<'\t'<<info.max_value<<'\t'<<info.default_value<<'\t'<<v<<'\n';}
 }else{
  require(argc==10||argc==11,"render MODULE STATE PREFIX RATE BLOCK DURATION PATTERN VELOCITY [ADVANCE]");
  const int rate=std::stoi(argv[5]),block=std::stoi(argv[6]),velocity=std::stoi(argv[9]);const double duration=std::stod(argv[7]);const std::string pattern=argv[8],prefix=argv[4];
  Stream s{file(argv[3]),0};clap_istream_t in{&s,read};require(state->load(p,&in),"Load test state");
  if(const char* restore=std::getenv("SAMPLE_RESTORE_STATE")){Stream old{file(restore),0};clap_istream_t prior{&old,read};require(state->load(p,&prior),"Restore old state over current state");}
  auto* render=static_cast<const clap_plugin_render_t*>(p->get_extension(p,CLAP_EXT_RENDER));if(render)require(render->set(p,CLAP_RENDER_OFFLINE),"Offline render mode");
  require(p->activate(p,rate,1,block),"Activate");audioThread=true;require(p->start_processing(p),"Start");audioThread=false;
  std::vector<float> il(block),ir(block),ol(block),orr(block);float* ins[]={il.data(),ir.data()};float* outs[]={ol.data(),orr.data()};
  clap_audio_buffer_t input{},output{};input.data32=ins;input.channel_count=2;output.data32=outs;output.channel_count=2;
  clap_input_events_t inputEvents{nullptr,eventCount,eventGet};clap_output_events_t outputEvents{nullptr,eventOut};
  clap_event_transport_t transport{};transport.header={sizeof(transport),0,CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_TRANSPORT,0};transport.flags=CLAP_TRANSPORT_HAS_TEMPO|CLAP_TRANSPORT_HAS_SECONDS_TIMELINE|CLAP_TRANSPORT_IS_PLAYING;transport.tempo=120;
  clap_process_t process{};process.frames_count=block;process.audio_inputs=&input;process.audio_inputs_count=1;process.audio_outputs=&output;process.audio_outputs_count=1;process.in_events=&inputEvents;process.out_events=&outputEvents;process.transport=&transport;
  auto run=[&](){pump();require(!restartPending,"Unexpected plugin restart");std::fill(ol.begin(),ol.end(),0);std::fill(orr.begin(),orr.end(),0);transport.song_pos_seconds=clap_sectime((double(process.steady_time)/rate)*CLAP_SECTIME_FACTOR);audioThread=true;auto status=p->process(p,&process);audioThread=false;require(status!=CLAP_PROCESS_ERROR,"Process");process.steady_time+=block;};
  // Background file decoding + source analysis must settle before test notes.
  const auto begin=std::chrono::steady_clock::now();while(std::chrono::steady_clock::now()-begin<std::chrono::seconds(3)){midi.clear();run();std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  save(prefix+".loaded.state");
  // A fixed warm-up ends on a common transport position regardless of loader timing.
  process.steady_time=0;for(int n=0;n<(rate+block-1)/block;++n){midi.clear();run();}
  // Optional prior trigger changes random history without changing mode/settings.
  // Drain for two seconds before capture to remove the prior note and effect tails.
  if(argc==11&&std::stoi(argv[10])>0){for(int h=0;h<std::stoi(argv[10]);++h){for(int n=0;n<(rate*2+block-1)/block;++n){midi.clear();if(n==0)addMidi(0,0x90,60,velocity);if(n==int(rate*.5)/block)addMidi(0,0x80,60,0);run();}}}
  std::ofstream pcm(prefix+".f32",std::ios::binary);require(bool(pcm),"PCM output");
  const int total=int(std::ceil(duration*rate));double peak=0,energy=0;int nonzero=0;
  const int spacing=pattern=="dense"?int(.12*rate):int(.8*rate),hold=pattern=="dense"?int(.09*rate):int(.5*rate);
  for(int offset=0;offset<total;offset+=block){midi.clear();parameterEvents.clear();if(const char* change=std::getenv("SAMPLE_PARAM_CHANGE")){if(offset<=int(.5*rate)&&offset+block>int(.5*rate)){const std::string text=change;const auto comma=text.find(',');const int index=std::stoi(text.substr(0,comma))-1;clap_param_info_t info{};require(params->get_info(p,index,&info),"Change parameter info");clap_event_param_value_t e{};e.header={sizeof(e),0,CLAP_CORE_EVENT_SPACE_ID,CLAP_EVENT_PARAM_VALUE,0};e.param_id=info.id;e.cookie=info.cookie;e.note_id=-1;e.port_index=e.channel=e.key=-1;e.value=std::stod(text.substr(comma+1));parameterEvents.push_back(e);}}for(int k=0;k<block&&offset+k<total;++k){int t=offset+k-int(.25*rate);if(t>=0&&t%spacing==0)addMidi(k,0x90,60,velocity);if(t>=hold&&(t-hold)%spacing==0)addMidi(k,0x80,60,0);}run();int count=std::min(block,total-offset);for(int k=0;k<count;++k){float l=ol[k],r=orr[k];require(std::isfinite(l)&&std::isfinite(r),"Nonfinite audio");pcm.write(reinterpret_cast<const char*>(&l),4);pcm.write(reinterpret_cast<const char*>(&r),4);peak=std::max(peak,double(std::max(std::abs(l),std::abs(r))));energy+=double(l)*l+double(r)*r;nonzero+=std::abs(l)+std::abs(r)>1e-8f;}}
  require(peak>1e-5||std::getenv("SAMPLE_ALLOW_SILENCE"),"Bank did not produce audio");std::ofstream report(prefix+".render.json");report<<std::setprecision(17)<<"{\"sample_rate\":"<<rate<<",\"block_size\":"<<block<<",\"frames\":"<<total<<",\"peak\":"<<peak<<",\"rms\":"<<std::sqrt(energy/(total*2))<<",\"nonzero_frames\":"<<nonzero<<"}";save(prefix+".final.state");
  audioThread=true;p->stop_processing(p);audioThread=false;p->deactivate(p);std::cout<<prefix<<" peak="<<peak<<" rms="<<std::sqrt(energy/(total*2))<<std::endl;
 }
 p->destroy(p);current=nullptr;entry->deinit();FreeLibrary(module);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
