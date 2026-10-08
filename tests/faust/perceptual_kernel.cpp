#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#if DSPJSFX_HAS_FAUST
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
#endif
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
extern "C" void jsfx_ensure_mem(DSPJSFX_State*s,int64_t n){assert(n<=s->memN);}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){assert(false);return 0;}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){assert(false);return 0;}
int main(int argc,char**argv){assert(argc==5);int rate=atoi(argv[2]),block=atoi(argv[3]),scenario=atoi(argv[4]);int frames=((rate*8+block-1)/block)*block;std::vector<float>l(frames),r(frames),ol(frames),orr(frames);std::vector<double>times,states;
for(int i=0;i<frames;i++){double t=double(i)/rate,a=t<2?0.12:t<3?1.2:t<4.5?0:t<5?1e-9:t<6?0.45:2.4;l[i]=float(a*(sin(2*3.141592653589793*310*t)+.25*sin(2*3.141592653589793*7900*t)));r[i]=float(a*(.8*sin(2*3.141592653589793*347*t)+.2*sin(2*3.141592653589793*11000*t)));if(i%1709==0&&a>.1)l[i]=float(a*1.5);}
for(int trial=0;trial<4;trial++){DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.srate=rate;s.memN=DSPJSFX_MAX_MEM_CELLS;s.mem=(DSPJSFX_Cell*)calloc(s.memN,sizeof(DSPJSFX_Cell));
#if DSPJSFX_HAS_FAUST
jsfx_faust::Engine engine;s.faustContext=&engine;engine.prepare(rate,block);
#endif
jsfx_init(&s);auto settings=[&](int phase){s.sliders[0]=scenario==0||scenario==3?0:scenario==1?1:2;s.sliders[1]=scenario==0?0:scenario==4?24:12;s.sliders[2]=scenario>=3?.65:0;if(phase==1){s.sliders[1]=0;s.sliders[2]=0;}if(phase==2){s.sliders[1]=18;s.sliders[2]=scenario>=3?.35:0;}if(phase==3){s.sliders[0]=scenario==1?2:scenario==2?1:s.sliders[0];s.sliders[1]=6;s.sliders[2]=scenario>=3?.95:0;}jsfx_slider(&s);};settings(0);auto start=std::chrono::steady_clock::now();
for(int i=0;i<frames;i+=block){if(i/block==frames/block/3)settings(1);if(i/block==frames/block*2/3)settings(2);if(i/block==frames/block*3/4)settings(3);const float*in[]={l.data()+i,r.data()+i};float*out[]={ol.data()+i,orr.data()+i};jsfx_process_block(&s,in,out,2,block);if(trial==3&&i/block%31==0){for(int j=1;j<300;j++)states.push_back(s.mem[j]);for(auto name:{"posteq_char_wet","output_harm_runtime_active"})for(auto v:DSPJSFX_VARS)if(!strcmp(name,v.name))states.push_back(s.vars[v.index]);}}
if(trial<3)times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());assert(!s.memoryFault);
#if DSPJSFX_HAS_FAUST
assert(engine.scalarCalls==0 && engine.blockCalls==uint64_t(frames/block));
#endif
free(s.mem);}
std::sort(times.begin(),times.end());std::ofstream a(std::string(argv[1])+".bin",std::ios::binary);a.write((char*)ol.data(),frames*4);a.write((char*)orr.data(),frames*4);std::ofstream st(std::string(argv[1])+".states.bin",std::ios::binary);st.write((char*)states.data(),states.size()*8);std::ofstream report(argv[1]);report<<"{\"rate\":"<<rate<<",\"block\":"<<block<<",\"scenario\":"<<scenario<<",\"frames\":"<<frames<<",\"median_seconds\":"<<times[1]<<"}";std::cout<<argv[1]<<" "<<times[1]<<std::endl;}
