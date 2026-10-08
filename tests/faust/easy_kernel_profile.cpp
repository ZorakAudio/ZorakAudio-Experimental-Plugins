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
#include <string>
#include <vector>
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t size){assert(size<=s->memN);}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){assert(false);return 0;}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){assert(false);return 0;}
int main(int argc,char** argv){
 if(argc<2)return 1;const int block=argc>2?std::stoi(argv[2]):256;const int rate=argc>3?std::stoi(argv[3]):48000;
 const int frames=((rate*4+block-1)/block)*block;std::vector<float> left(frames),right(frames),outputL(frames),outputR(frames);
 for(int i=0;i<frames;++i){double t=double(i)/rate;double amplitude=(i/(rate/5))%3==0?0.0003:0.02;left[i]=float(amplitude*(std::sin(2*3.141592653589793*310*t)+0.3*std::sin(2*3.141592653589793*1950*t)));right[i]=float(amplitude*(0.7*std::sin(2*3.141592653589793*315*t)+0.1*std::sin(2*3.141592653589793*3000*t)));}
 std::vector<double> trials;
 for(int trial=0;trial<3;++trial){
  DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.srate=rate;s.memN=DSPJSFX_MAX_MEM_CELLS;s.mem=static_cast<DSPJSFX_Cell*>(std::calloc(size_t(s.memN),sizeof(DSPJSFX_Cell)));
  const double defaults[]={-40,24,50,0,20000};
  const char* aliases[]={"thresh_db","depth_db","contour","det_hpf_hz","det_lpf_hz"};
  auto sync=[&]{for(int c=0;c<5;++c)for(const auto& v:DSPJSFX_VARS)if(std::strcmp(v.name,aliases[c])==0)s.vars[v.index]=double(s.sliders[c]);};
  for(int c=0;c<5;++c)s.sliders[c]=defaults[c];sync();
#if DSPJSFX_HAS_FAUST
  jsfx_faust::Engine engine;s.faustContext=&engine;engine.prepare(rate,block);
#endif
  jsfx_init(&s);sync();jsfx_slider(&s);
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<frames;i+=block){
   if(i/block==frames/block/3){s.sliders[3]=200;s.sliders[4]=5000;sync();jsfx_slider(&s);}
   if(i/block==frames/block*2/3){s.sliders[0]=-35;s.sliders[2]=75;sync();jsfx_slider(&s);}
   const float* in[]={left.data()+i,right.data()+i};float* out[]={outputL.data()+i,outputR.data()+i};jsfx_process_block(&s,in,out,2,block);
  }
  trials.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());assert(!s.memoryFault);
#if DSPJSFX_HAS_FAUST
  assert(engine.scalarCalls==0 && engine.blockCalls==uint64_t(frames/block));
#endif
  std::free(s.mem);
 }
 std::sort(trials.begin(),trials.end());std::ofstream binary(std::string(argv[1])+".bin",std::ios::binary);binary.write(reinterpret_cast<const char*>(outputL.data()),frames*sizeof(float));binary.write(reinterpret_cast<const char*>(outputR.data()),frames*sizeof(float));
 std::ofstream report(argv[1]);report<<"{\"sample_rate\":"<<rate<<",\"block_size\":"<<block<<",\"frames\":"<<frames<<",\"median_seconds\":"<<trials[1]<<",\"ns_per_frame\":"<<trials[1]*1e9/frames<<"}";
 std::cout<<"kernel median="<<trials[1]<<" seconds / "<<frames<<" frames; "<<trials[1]*1e9/frames<<" ns/frame\n";
}
