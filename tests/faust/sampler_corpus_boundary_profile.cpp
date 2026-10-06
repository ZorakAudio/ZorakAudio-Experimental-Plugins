#include "JSFXDSP.h"
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cassert>
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){assert(n<=s->memN);}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){return 0;}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){return 0;}
double rd(DSPJSFX_State&s,jsfx_faust::Binding b){switch(b.kind){case 0:return s.vars[b.index];case 1:return s.sliders[b.index];case 2:return s.spl[b.index];case 3:return s.srate;case 4:return s.samplesblock;default:return 0;}}
int main(){DSPJSFX_State s{};s.srate=48000;s.samplesblock=256;s.memN=DSPJSFX_MAX_MEM_CELLS;s.mem=(DSPJSFX_Cell*)calloc(s.memN,sizeof(DSPJSFX_Cell));jsfx_init(&s);
#if CORPUS_TEST
s.sliders[44]=.8;
#else
s.sliders[0]=90;s.sliders[1]=5500;s.sliders[2]=4;s.sliders[3]=9;s.sliders[4]=6;
#endif
jsfx_slider(&s);s.spl[0]=.2;s.spl[1]=.1;
for(auto&st:jsfx_faust::stages)if(st.kind==1) {st.eel(&s);break;}
for(auto&st:jsfx_faust::stages)if(st.kind==2){assert(!st.tableCount);std::vector<uint64_t>mem((st.size+7)/8);st.allocate(mem.data());st.classInit(48000);st.constants(mem.data(),48000);st.clear(mem.data());std::vector<double>values;for(int z=0;z<st.zoneCount;z++){double v=st.zones[z].source.kind>=0?rd(s,st.zones[z].source):st.zones[z].initial;values.push_back(v);memcpy((char*)mem.data()+st.zones[z].offset,&v,8);}std::vector<std::vector<double>>ib(st.inputs,std::vector<double>(1024,.2)),ob(st.outputs,std::vector<double>(1024));std::vector<double*>ip,op;for(auto&v:ib)ip.push_back(v.data());for(auto&v:ob)op.push_back(v.data());
std::cout<<"{\"zones\":"<<st.zoneCount<<",\"exports\":"<<st.exportCount<<",\"measurements\":[";bool first=true;
for(int mode=0;mode<4;mode++){int count=mode==3?256:1;std::vector<double>times;for(int trial=0;trial<5;trial++){const int frames=2097152;auto start=std::chrono::steady_clock::now();for(int f=0;f<frames;f+=count){if(mode==1||mode==2)for(int z=0;z<st.zoneCount;z++)if(st.zones[z].source.kind>=0)memcpy((char*)mem.data()+st.zones[z].offset,&values[z],8);st.compute(mem.data(),count,ip.data(),op.data());if(mode==2)for(int j=0;j<st.exportCount;j++){double v=op[st.audioOutputs+j][count-1];uint64_t bits;memcpy(&bits,&v,8);auto exponent=(bits>>52)&2047;if(DSPJSFX_EEL2_STORES&&(exponent==0||exponent==2047))v=0;auto b=st.exports[j];assert(b.kind==0);s.vars[b.index]=v;}}auto end=std::chrono::steady_clock::now();times.push_back(std::chrono::duration<double>(end-start).count()*1e9/frames);}std::sort(times.begin(),times.end());if(!first)std::cout<<",";first=false;std::cout<<"{\"mode\":"<<mode<<",\"nanoseconds_per_frame\":"<<times[2]<<"}";}std::cout<<"],\"checksum\":"<<op[0][0]<<"}\n";st.destroy(mem.data());}free(s.mem);}
