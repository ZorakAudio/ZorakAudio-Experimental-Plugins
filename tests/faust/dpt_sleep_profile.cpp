#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){if(n>s->memN)throw std::runtime_error("Unexpected allocation");}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){throw std::runtime_error("Unexpected GFX DSP call");}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){throw std::runtime_error("Unexpected string DSP call");}
int main(int argc,char** argv){
 if(argc<5)return 1;
 const int rate=std::stoi(argv[2]), block=std::stoi(argv[3]);const bool cooperative=std::stoi(argv[4]);
 const int frames=((rate*12+block-1)/block)*block;const int mode=argc>5?std::stoi(argv[5]):0;
 DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.srate=rate;s.memN=DSPJSFX_MAX_MEM_CELLS;s.mem=static_cast<DSPJSFX_Cell*>(std::calloc(size_t(s.memN),sizeof(DSPJSFX_Cell)));
 s.sliders[0]=0;s.sliders[1]=70;s.sliders[2]=mode;s.sliders[3]=0;jsfx_init(&s);jsfx_slider(&s);
 int ready=-1;for(auto v:DSPJSFX_VARS)if(std::strcmp(v.name,"za_sleep_ready")==0)ready=v.index;
 std::vector<float> l(block),r(block),ol(block),orr(block),dump;dump.reserve(size_t(frames)*2);
 bool sleeping=false;int skipped=0,grants=0;double elapsed=0;
 for(int pos=0;pos<frames;pos+=block){
  bool event=false;
  // The wake switches into Headphones, exercising history retained during Speakers sleep.
  if(pos/block==rate*10/block){s.sliders[0]=73;s.sliders[2]=1;jsfx_slider(&s);event=true;}
  bool input=false;
  for(int k=0;k<block;++k){const double t=double(pos+k)/rate;float x=0;
   if(t<2 || t>=11)x=float(0.1*std::sin(431*t)+0.03*std::sin(2703*t));
   else if(t>=10)x=1e-9f;
   l[k]=x;r[k]=x*0.7f;input|=x!=0;
  }
  if(event||input)sleeping=false;
  auto start=std::chrono::steady_clock::now();
  if(cooperative&&sleeping){std::fill(ol.begin(),ol.end(),0);std::fill(orr.begin(),orr.end(),0);++skipped;}
  else {
   if(ready>=0)s.vars[ready]=0;
   const float* in[]={l.data(),r.data()};float* out[]={ol.data(),orr.data()};jsfx_process_block(&s,in,out,2,block);
   bool zero=true;for(int k=0;k<block;++k)zero&=ol[k]==0&&orr[k]==0;
   if(ready>=0&&s.vars[ready]==1&&!input&&!event&&zero){++grants;if(cooperative)sleeping=true;}
  }
  elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if(s.memoryFault)throw std::runtime_error("Memory fault");
  for(int k=0;k<block;++k){if(!std::isfinite(ol[k])||!std::isfinite(orr[k]))throw std::runtime_error("Nonfinite output");dump.push_back(ol[k]);dump.push_back(orr[k]);}
 }
 std::ofstream bin(std::string(argv[1])+".bin",std::ios::binary);bin.write(reinterpret_cast<char*>(dump.data()),dump.size()*sizeof(float));
 std::ofstream report(argv[1]);report<<"{\"seconds\":"<<elapsed<<",\"skipped_blocks\":"<<skipped<<",\"grants\":"<<grants<<",\"mode\":"<<mode<<"}";
 std::free(s.mem);std::cout<<"DPT mode="<<mode<<" cooperative="<<cooperative<<" skipped="<<skipped<<" time="<<elapsed<<"\n";
}
