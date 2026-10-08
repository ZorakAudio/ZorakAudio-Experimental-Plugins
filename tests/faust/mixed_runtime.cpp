#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
#include <cassert>
#include <iostream>
#include <thread>
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){assert(false);}
static int idx(const char* name) {for(const auto& v:DSPJSFX_VARS)if(std::strcmp(v.name,name)==0)return v.index;assert(false);return 0;}
int main(){
 DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.srate=48000;s.sliders[0]=0.5;
 jsfx_faust::Engine engine;s.faustContext=&engine;engine.prepare(48000,32);jsfx_init(&s);
 float a[33],b[33],c[33],d[33];for(int i=0;i<33;++i){a[i]=0.8f;b[i]=-0.4f;}
#if TEST_KIND==6 || TEST_KIND==7
 for(int i=0;i<32;++i)a[i]+=0.001f*i;
#endif
 const float* in[]={a,b};float* out[]={c,d};
 jsfx_process_block(&s,in,out,2,32);
 for(int i=0;i<32;++i){
#if TEST_KIND==14
  assert(c[i]==a[i] && d[i]==b[i]);
#elif TEST_KIND==10
  assert(std::isfinite(c[i]) && std::abs(c[i])<=1 && c[i]==d[i]);
#elif TEST_KIND==11
  assert(c[i]==48000.f && d[i]==b[i]);
#elif TEST_KIND==5
  assert(std::abs(c[i]-0.8*(i+1))<2e-6 && std::abs(d[i]+0.4f)<1e-6);
#elif TEST_KIND==6
  assert(c[i]==a[i]);
#elif TEST_KIND==7
  assert(std::abs(c[i]-a[i]*a[i])<1e-6 && std::abs(d[i]-b[i]*a[i])<1e-6);
#elif TEST_KIND==8
  assert(std::abs(c[i]-a[i]*(0.5+0.01*(i+1)))<1e-6);
#elif TEST_KIND==9
  assert(c[i]==a[i] && d[i]==b[i]);
#elif TEST_KIND==1
  assert(std::abs(c[i]-0.1f)<1e-6 && std::abs(d[i]+0.05f)<1e-6);
#elif TEST_KIND==2 || TEST_KIND==12 || TEST_KIND==13
  assert(std::abs(c[i]-a[i]*(0.5+0.01*(i+1)))<1e-6);
#elif TEST_KIND==3
  assert(std::abs(c[i]-0.4f)<1e-6 && std::abs(d[i]+0.2f)<1e-6);
#elif TEST_KIND==4
  assert(std::abs(c[i]-0.8f)<1e-6);
#else
  assert(std::abs(c[i]-0.4f)<1e-6 && std::abs(d[i]+0.2f)<1e-6);
#endif
 }
#if TEST_KIND==3
 assert(std::abs(double(s.vars[idx("meter")])-0.8)<1e-6);
#endif
#if TEST_KIND==14
 assert(engine.blockCalls==0 && engine.scalarCalls==0);
 s.vars[idx("gate")]=1;jsfx_process_block(&s,in,out,2,32);
 for(int i=0;i<32;++i)assert(std::abs(c[i]-0.8*(i+1))<2e-6);
 s.vars[idx("gate")]=0;jsfx_process_block(&s,in,out,2,32);
 for(int i=0;i<32;++i)assert(c[i]==a[i]);
 s.vars[idx("gate")]=1;jsfx_process_block(&s,in,out,2,32);
 for(int i=0;i<32;++i)assert(std::abs(c[i]-0.8*(i+33))<4e-6);
 assert(engine.blockCalls==2 && engine.scalarCalls==0);
#elif TEST_KIND==2 || TEST_KIND==6 || TEST_KIND==8
 assert(engine.scalarCalls==32 && engine.blockCalls==0);
#elif TEST_KIND==7
 assert(engine.scalarCalls==64 && engine.blockCalls==0);
#else
 assert(engine.blockCalls>0 && engine.scalarCalls==0);
#endif
 DSPJSFX_State other{};za::jsfx::StateVariables other_variables;other_variables.bind(other,DSPJSFX_VARS_COUNT);other.srate=96000;jsfx_faust::Engine second;other.faustContext=&second;second.prepare(96000,32);jsfx_init(&other);
 assert(other.faustContext!=s.faustContext);
#if TEST_KIND==13
 assert(double(s.vars[idx("blocks")])==1 && double(s.vars[idx("boundaries")])==4);
 assert(engine.blockCalls==4 && engine.scalarCalls==0);
 jsfx_process_block(&s,in,out,2,23);
 for(int i=0;i<23;++i)assert(std::abs(c[i]-a[i]*(0.5+0.01*(33+i)))<1e-6);
 assert(double(s.vars[idx("blocks")])==2 && double(s.vars[idx("boundaries")])==7);
 assert(engine.blockCalls==7 && engine.scalarCalls==0);
 jsfx_process_block(&s,in,out,2,33);
 for(int i=0;i<33;++i)assert(std::abs(c[i]-a[i]*(0.5+0.01*(56+i)))<1e-6);
 assert(double(s.vars[idx("blocks")])==3 && double(s.vars[idx("boundaries")])==12);
 assert(engine.blockCalls==12 && engine.scalarCalls==0);
#endif
#if TEST_KIND==5
 jsfx_process_block(&other,in,out,2,32);assert(std::abs(c[0]-0.8f)<1e-6);
 jsfx_process_block(&s,in,out,2,32);assert(std::abs(c[0]-0.8*33)<2e-6);
#endif
#if TEST_KIND==9
 assert(double(s.sliders[0])==0.25 && (s.pendingSliderChangeMask[0]&1));
#endif
#if TEST_KIND==11
 jsfx_process_block(&other,in,out,2,32);assert(c[0]==96000.f);
 jsfx_process_block(&s,in,out,2,32);assert(c[0]==48000.f);
 std::thread audioThread([&]{jsfx_process_block(&s,in,out,2,32);assert(c[0]==48000.f);});audioThread.join();
 s.srate=96000;engine.reset(96000);jsfx_process_block(&s,in,out,2,32);assert(c[0]==96000.f);s.srate=48000;
#endif
 engine.reset(48000);assert(engine.blockCalls==0 && engine.scalarCalls==0);
 jsfx_process_block(&s,in,out,2,0);assert(!s.memoryFault);
 std::cout<<"Mixed section ordering, bindings, calls and instance isolation passed\n";
}
