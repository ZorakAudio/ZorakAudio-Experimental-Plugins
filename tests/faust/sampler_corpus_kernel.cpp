#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#if DSPJSFX_HAS_FAUST
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t size){assert(size<=s->memN);}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){assert(false);return 0;}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){assert(false);return 0;}
int main(int argc,char** argv){
 assert(argc==5);const int rate=std::stoi(argv[2]),block=std::stoi(argv[3]),scenario=std::stoi(argv[4]);
 const int frames=((int(rate*4)+block-1)/block)*block;
 std::vector<float> l(frames),r(frames),ol(frames),orr(frames);std::vector<double> times,states;
 for(int i=0;i<frames;++i){double t=double(i)/rate;double amplitude=(t<1.4||t>=2.6)?0.5:0;
 l[i]=float(amplitude*(0.5*std::sin(2*3.141592653589793*310*t)+0.15*std::sin(2*3.141592653589793*7000*t)));
 r[i]=float(amplitude*(0.35*std::sin(2*3.141592653589793*315*t)+0.1*std::sin(2*3.141592653589793*9000*t)));
 if(i%997==0&&amplitude)l[i]=0.7f;if(t>=2.3&&t<2.4){l[i]=1e-9f;r[i]=-1e-9f;}}
 uint64_t scalars=0,blocks=0;
 for(int trial=0;trial<4;++trial){DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.srate=rate;s.memN=DSPJSFX_MAX_MEM_CELLS;s.mem=static_cast<DSPJSFX_Cell*>(std::calloc(size_t(s.memN),sizeof(DSPJSFX_Cell)));
#if DSPJSFX_HAS_FAUST
 jsfx_faust::Engine engine;s.faustContext=&engine;engine.prepare(rate,block);
#endif
 jsfx_init(&s);
 auto settings=[&](int phase){
#if CORPUS_TEST
 s.sliders[44]=scenario==0?0:scenario==1?0.35:1;
 if(phase==1)s.sliders[44]=0;if(phase==2)s.sliders[44]=0.8;if(phase==3)s.sliders[44]=0.15;
#else
 s.sliders[0]=scenario==0?20:90;s.sliders[1]=scenario==0?20000:5500;s.sliders[2]=scenario<2?0:scenario==2?1:4;s.sliders[3]=scenario==0?0:9;s.sliders[4]=scenario==3?6:0;
 if(phase==1){s.sliders[0]=20;s.sliders[1]=20000;s.sliders[3]=0;s.sliders[4]=0;}
 if(phase==2){s.sliders[0]=200;s.sliders[1]=8000;s.sliders[2]=4;s.sliders[3]=-9;s.sliders[4]=4;}
#endif
 if(phase==3){
#if !CORPUS_TEST
 s.sliders[0]=350;s.sliders[1]=6500;s.sliders[3]=3;s.sliders[4]=2;
#endif
}
 jsfx_slider(&s);};settings(0);auto start=std::chrono::steady_clock::now();
 for(int i=0;i<frames;i+=block){if(i/block==frames/block/3)settings(1);if(i/block==frames/block*2/3)settings(2);if(i/block==frames/block*3/4)settings(3);
 const float* in[]={l.data()+i,r.data()+i};float* out[]={ol.data()+i,orr.data()+i};jsfx_process_block(&s,in,out,2,block);
 if(trial==3&&((i/block)%17==0||i+block==frames)){
#if CORPUS_TEST
  #if !PRIVATE_CORPUS
  for(int j=1;j<7100;++j)states.push_back(double(s.mem[j]));
#endif
  for(auto name:{"td_amount_rt","td_fast","td_slow","td_mix","td_dry_peak","td_block_trans","td_block_mix","td_block_guard","td_block_inpeak","td_block_outpeak"})for(const auto& v:DSPJSFX_VARS)if(!std::strcmp(v.name,name))states.push_back(double(s.vars[v.index]));
#else
  for(int j=1;j<118;++j)states.push_back(double(s.mem[j]));
  for(auto name:{"proc_hpf1_l","proc_hpf1_r","proc_lpf1_l","proc_lpf1_r","posteq_filter_silent"})for(const auto& v:DSPJSFX_VARS)if(!std::strcmp(v.name,name))states.push_back(double(s.vars[v.index]));
#endif
 }
 }if(trial<3)times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());assert(!s.memoryFault);
#if DSPJSFX_HAS_FAUST
 scalars=engine.scalarCalls;blocks=engine.blockCalls;
#if PRIVATE_CORPUS || PRIVATE_SAMPLE_BLOCK
 assert(scalars==0&&blocks==uint64_t(frames/block));
#else
 assert(scalars==uint64_t(frames)&&blocks==0);
#endif
#endif
 std::free(s.mem);}
 std::sort(times.begin(),times.end());std::ofstream bin(std::string(argv[1])+".bin",std::ios::binary);bin.write((char*)ol.data(),frames*sizeof(float));bin.write((char*)orr.data(),frames*sizeof(float));bin.close();
 std::ofstream st(std::string(argv[1])+".states.bin",std::ios::binary);st.write((char*)states.data(),states.size()*sizeof(double));
 std::ofstream report(argv[1]);report<<"{\"sample_rate\":"<<rate<<",\"block_size\":"<<block<<",\"scenario\":"<<scenario<<",\"frames\":"<<frames<<",\"median_seconds\":"<<times[1]<<",\"scalar_calls\":"<<scalars<<",\"block_calls\":"<<blocks<<",\"state_values\":"<<states.size()<<"}";
 std::cout<<argv[1]<<" median="<<times[1]<<" scalar="<<scalars<<std::endl;
}
