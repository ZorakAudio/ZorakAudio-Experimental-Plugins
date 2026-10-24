#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <juce_audio_utils/juce_audio_utils.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#include <mmsystem.h>
#endif
extern juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
extern "C" bool za_native_gfx_snapshot(juce::AudioProcessor*,const char*,double*,int*,size_t*);
extern "C" void za_sample_gfx_load_bank(juce::AudioProcessor*,const char* const*,int);
extern "C" void za_native_gfx_faults(juce::AudioProcessor*,uint32_t*,uint32_t*);
extern "C" int za_native_gfx_heap_copy(juce::AudioProcessor*,const char*,double*,int);
#include "corpus_diagnostics.h"
#include "corpus_pitch_check.h"
#include "corpus_duration_check.h"
#include "corpus_cloud_check.h"
#include "corpus_repeat_check.h"
#include "corpus_trajectory_check.h"
#include "corpus_follow_diagnostic.h"
#include "corpus_auto_diagnostic.h"
#include "corpus_detection_check.h"
#include "corpus_live_check.h"
#include "corpus_live_capture.h"
#include "corpus_cleanup_check.h"
#include "corpus_picker_check.h"
using Clock=std::chrono::steady_clock;
static double seconds(Clock::duration d){return std::chrono::duration<double>(d).count();}
static double cpu(){
#if defined(_WIN32)
 FILETIME c,e,k,u;GetProcessTimes(GetCurrentProcess(),&c,&e,&k,&u);
 ULARGE_INTEGER a,b;a.LowPart=k.dwLowDateTime;a.HighPart=k.dwHighDateTime;b.LowPart=u.dwLowDateTime;b.HighPart=u.dwHighDateTime;
 return (a.QuadPart+b.QuadPart)*1e-7;
#else
 return double(std::clock())/CLOCKS_PER_SEC;
#endif
}
struct Stats{double wall=0,processCpu=0,dsp=0,max=0;uint64_t blocks=0,overruns=0;std::vector<double> calls;};
int main(int argc,char** argv)try{
 if(argc<3)throw std::runtime_error("Pass the single input recording and output CSV path; optional --unpaced");
 juce::ScopedJuceInitialiser_GUI gui;
#if defined(_WIN32)
 struct Timer{Timer(){timeBeginPeriod(1);}~Timer(){timeEndPeriod(1);}} timer;
#endif
 constexpr double rate=48000;constexpr int blockSize=256;const double period=blockSize/rate;
 auto option=[&](const char* flag){for(int i=3;i<argc;++i) if(std::string(argv[i])==flag)return true;return false;};
 const bool paced=!option("--unpaced");bool reloaded=false;double reloadGeneration=0;
 juce::AudioFormatManager formats;formats.registerBasicFormats();
 std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(juce::File(juce::String::fromUTF8(argv[1]))));
 if(!reader)throw std::runtime_error("Unable to read the supplied recording");
 std::cout<<"input duration "<<reader->lengthInSamples/reader->sampleRate<<" s, "<<reader->sampleRate<<" Hz, "<<reader->numChannels<<" channels; paced="<<paced<<std::endl;
 reader.reset();
 std::unique_ptr<juce::AudioProcessor> p(createPluginFilter());
 p->setRateAndBufferSizeDetails(rate,blockSize);p->prepareToPlay(rate,blockSize);
 auto value=[&](const char* name){double v=0;int spans=0;size_t capacity=0;za_native_gfx_snapshot(p.get(),name,&v,&spans,&capacity);return v;};
 auto stage=[&](){
   for(const auto& item:std::vector<std::pair<const char*,const char*>>{{"build_phase","Index"},{"xc_phase","Features"},{"h_phase","Structure"},{"xc_post_phase","Grammar"},{"xc_map_phase","Map"},{"fc_phase","Focus"},{"pe_phase","PE"},{"pg_stage","Prune"}})
     if(value(item.first)>0)return std::string(item.second)+"/"+std::to_string(int(value(item.first)));
   return std::string(value("ready")>0?"Coordinator":"Decode/load");
 };
 for(int i=3;i+1<argc;++i){if(std::string(argv[i])=="--grain-budget"){int count=std::stoi(argv[i+1]);int index=int(std::round(std::log2(count/4096.)));if(index<0 || index>4 || count!=(4096<<index))throw std::runtime_error("Budget must be4096/8192/16384/32768/65536");corpusSet(p.get(),6,float(index));}if(std::string(argv[i])=="--micro-ms")corpusSet(p.get(),5,std::stof(argv[i+1]));}
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--detection-style")corpusSet(p.get(),242,std::stof(argv[i+1]));
 std::vector<std::string> bankNames;std::vector<const char*> bankPaths;
 if(option("--load-directory")){juce::Array<juce::File> files;juce::File(juce::String::fromUTF8(argv[1])).getParentDirectory().findChildFiles(files,juce::File::findFiles,false,"*.wav");std::sort(files.begin(),files.end());for(auto& file:files)bankNames.push_back(file.getFullPathName().toStdString());}else bankNames.push_back(argv[1]);for(auto& name:bankNames)bankPaths.push_back(name.c_str());
 auto start=Clock::now();double cpuStart=cpu();za_sample_gfx_load_bank(p.get(),bankPaths.data(),int(bankPaths.size()));
 juce::AudioBuffer<float> audio(std::max(2,p->getTotalNumOutputChannels()),blockSize);juce::MidiBuffer midi;
 std::map<std::string,Stats> stats;std::string last;uint64_t blocks=0;bool ready=false;
 auto next=start;auto lastReport=start;
 while(seconds(Clock::now()-start)<3600){
   auto label=stage();if(label!=last){std::cout<<seconds(Clock::now()-start)<<" s: "<<label<<std::endl;last=label;}
   if(!reloaded && (option("--reload-at-index") && value("cg_stage")==7 && value("build_phase")==3 || option("--reload-at-pe") && value("cg_stage")>=6 && value("pe_phase")==1 || option("--reload-at-copy") && value("cg_stage")==7 && value("pe_phase")==2)) {
     reloadGeneration=value("generation");za_sample_gfx_load_bank(p.get(),bankPaths.data(),int(bankPaths.size()));reloaded=true;std::cout<<"Reloading the same supplied recording during the graph"<<std::endl;
   }
   auto before=Clock::now();double c=cpu();audio.clear();midi.clear();auto callbackStart=Clock::now();p->processBlock(audio,midi);
   double callback=seconds(Clock::now()-callbackStart);auto& st=stats[label];st.blocks++;st.dsp+=callback;st.max=std::max(st.max,callback);st.calls.push_back(callback);st.overruns+=callback>period;
   ++blocks;
   ready=value("ready")>0 && value("xc_ready")>0 && value("h_ready")>0 && value("fc_ready")>0 && value("xc_grammar_ready")>0 && value("xc_map_ready")>0 && value("pe_ready")>0 && value("pg_ready")>0 && value("pg_initializing")==0 && value("pg_load_pending")==0;
   if(reloaded && value("generation")==reloadGeneration)ready=false;
   if(!ready){
     next+=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period));
     if(paced)std::this_thread::sleep_until(next);else if(blocks%64==0)std::this_thread::yield();
   }
   st.wall+=seconds(Clock::now()-before);st.processCpu+=cpu()-c;
   if(seconds(Clock::now()-lastReport)>=20){std::cout<<"progress "<<label<<" cursor "<<(label.rfind("Features",0)==0?value("xc_cursor"):label.rfind("PE",0)==0?value("pe_cursor"):value("h_cursor"))<<" / grains "<<value("grain_count")<<std::endl;lastReport=Clock::now();}
   if(ready)break;
 }
 uint32_t dspFault=0,gfxFault=0;za_native_gfx_faults(p.get(),&dspFault,&gfxFault);
 std::ofstream csv(argv[2]);csv<<"stage,wall_s,audio_clock_s,process_cpu_s,callback_time_s,callbacks,p95_ms,p99_ms,max_ms,deadline_overruns\n";
 double totalWall=seconds(Clock::now()-start),totalCpu=cpu()-cpuStart;
 for(auto& [name,s]:stats){std::sort(s.calls.begin(),s.calls.end());auto percentile=[&](double q){return s.calls[std::min(s.calls.size()-1,size_t(q*(s.calls.size()-1)))]*1000;};
   csv<<name<<','<<s.wall<<','<<s.blocks*period<<','<<s.processCpu<<','<<s.dsp<<','<<s.blocks<<','<<percentile(.95)<<','<<percentile(.99)<<','<<s.max*1000<<','<<s.overruns<<'\n';}
 csv<<"TOTAL,"<<totalWall<<','<<blocks*period<<','<<totalCpu<<",,,,,,\n";csv.close();
 std::cout<<"RESULT ready="<<ready<<" wall="<<totalWall<<" audio_clock="<<blocks*period<<" process_cpu="<<totalCpu<<" grains="<<value("grain_count")<<" regions="<<value("xc_coarse_n")<<" fft="<<value("xc_fft_count")<<" comparisons="<<value("xc_precompute_comparisons")<<" fallbacks="<<value("xc_task_fallbacks")<<" faults="<<dspFault<<','<<gfxFault<<std::endl;
 if(option("--dump-model") && ready) {
   std::ofstream model(std::string(argv[2])+".model.bin",std::ios::binary);
   for(const auto& [name,count]:std::vector<std::pair<const char*,int>>{
       {"source_base",int(value("source_count"))*8},{"grain_base",int(value("grain_count"))*64},
       {"hist",1024},{"norm_low",8},{"norm_span",8},{"counts",29},{"offsets",30},{"vertex_counts",29},{"role_stats",16},
       {"xc_features",int(value("grain_count"))*112},{"xc_coarse",int(value("xc_coarse_n"))*48},{"xc_matrix",512*512},
       {"hm_episodes",int(value("hm_n"))*96},{"hx_episodes",int(value("hx_n"))*128},
       {"hm_classes",int(value("hm_k"))*96},{"hx_classes",int(value("hx_k"))*128},
       {"hm_grainmap",int(value("grain_count"))},{"h_change",int(value("grain_count"))},
       {"focus_ep_vec",int(value("hm_n"))*64},{"focus_ep_conf",int(value("hm_n"))*64},
       {"xc_maps",int(value("hm_n"))*12},{"pe_vectors",int(value("grain_count"))*16},
       {"pe_stats",int(value("grain_count"))*16},{"pe_prefix",int(value("grain_count"))*4},
       {"pe_source_hash",int(value("source_count"))*4}}) {
     std::vector<double> data(size_t(count),0.);
     if(za_native_gfx_heap_copy(p.get(),name,data.data(),count)!=count)throw std::runtime_error("Model dump failed");
     model.write(name,std::streamsize(std::strlen(name)+1));model.write(reinterpret_cast<const char*>(&count),sizeof(count));
     model.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(double)));
   }
 }
 if(option("--check-playback") && ready) {
   bool heard=false;
   for(int b=0;b<600;++b){audio.clear();midi.clear();if(!b)midi.addEvent(juce::MidiMessage::noteOn(1,60,juce::uint8(100)),0);if(b==400)midi.addEvent(juce::MidiMessage::noteOff(1,60),0);p->processBlock(audio,midi);heard|=audio.getMagnitude(0,audio.getNumSamples())>1e-8f;}
   std::unique_ptr<juce::AudioProcessorEditor> editor(p->createEditor());if(!editor || !heard)throw std::runtime_error("Playback/editor check failed");
   std::cout<<"Playback and editor check passed"<<std::endl;
 }
 std::cout<<"GRAPH completed="<<value("cg_completed")<<" disabled="<<value("cg_disabled")<<" pe_read_errors="<<value("pe_read_errors")<<" spectral_read_errors="<<value("invalid_audio")<<std::endl;
 if(option("--compare-triggers") && ready)corpusCompare(p.get(),audio,midi,value,std::string(argv[2])+".triggers.csv");
 if(option("--check-mono") && ready)corpusMono(p.get(),audio,midi,value);
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--live-screenshots" && ready)corpusLiveCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--check-picker" && ready)corpusPickerCheck(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--check-repeat" && ready)corpusRepeatCheck(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if(option("--check-cleanup") && ready)corpusCleanupCheck(p.get(),audio,midi,value,std::string(argv[2])+".cleanup.csv");
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--cleanup-screenshots" && ready)corpusCleanupCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if(option("--check-live") && ready)corpusLiveCheck(p.get(),audio,midi,value,std::string(argv[2])+".live.csv");
 if(option("--check-detection") && ready)corpusDetectionCheck(p.get(),audio,midi,value,std::string(argv[2])+".detection.csv");
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--detection-screenshots" && ready)corpusDetectionCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if(option("--diagnose-auto") && ready)corpusAutoDiagnostic(p.get(),audio,midi,value,argv[1],std::string(argv[2])+".auto");
 if(option("--diagnose-follow") && ready)corpusFollowDiagnostic(p.get(),audio,midi,value,std::string(argv[2])+".follow");
 if(option("--check-trajectories") && ready)corpusTrajectoryCheck(p.get(),audio,midi,value,std::string(argv[2])+".trajectories.csv");
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--trajectory-screenshots" && ready)corpusTrajectoryCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if(option("--check-cloud") && ready)corpusCloudCheck(p.get(),audio,midi,value,std::string(argv[2])+".cloud.csv");
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--cloud-screenshots" && ready)corpusCloudCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if(option("--check-duration") && ready)corpusDurationCheck(p.get(),audio,midi,value,std::string(argv[2])+".duration.csv");
 if(option("--check-pitch-tame") && ready)corpusPitchCheck(p.get(),audio,midi,value,std::string(argv[2])+".pitch-tame.csv");
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--pitch-screenshots" && ready)corpusPitchCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 for(int i=3;i+1<argc;++i)if(std::string(argv[i])=="--screenshots" && ready)corpusCapture(p.get(),audio,midi,value,juce::File(juce::String::fromUTF8(argv[i+1])));
 if((option("--reload-at-index") || option("--reload-at-pe") || option("--reload-at-copy")) && !reloaded)throw std::runtime_error("Requested reload point was never reached");
 return ready && !dspFault && !gfxFault && value("pe_read_errors")==0 ?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
