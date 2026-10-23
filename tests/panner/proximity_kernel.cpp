#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){require(n<=s->memN,"Memory extent");}
extern "C" double jsfx_native_gfx_dispatch(DSPJSFX_State*,int32_t,double**,int32_t){return 0;}
extern "C" double jsfx_native_string_dispatch(DSPJSFX_State*,int32_t,const double*,int32_t){return 0;}
extern "C" double jsfx_slider_show(DSPJSFX_State*,double,int32_t,double,int32_t){return 0;}
extern "C" double jsfx_native_gfx_getchar(DSPJSFX_State*,double){return 0;}
extern "C" double jsfx_native_slider_event(DSPJSFX_State*,double,int32_t,int32_t,double){return 0;}
static double observe(DSPJSFX_State& s,const char* name){for(auto v:DSPJSFX_VARS)if(!std::strcmp(v.name,name))return s.vars[v.index];return 0;}
static double texture(double t,int channel){
 double x=0;for(int i=0;i<11;++i)x+=std::sin(6.283185307179586*(83.0+channel*9.7)*(i+1.0)*(1+.013*i)*t+channel*.41*i)/std::sqrt(i+1.0);
 return std::tanh(2.8*x);
}
int main(int argc,char** argv)try{
 require(argc==18||argc==22,"prefix rate block fixture mode distance proximity anti level motion room late trim lateral model seconds");
 const char* prefix=argv[1];const int rate=std::atoi(argv[2]),block=std::atoi(argv[3]),fixture=std::atoi(argv[4]),mode=std::atoi(argv[5]);
 const double distance=std::atof(argv[6]),amount=std::atof(argv[7]),anti=std::atof(argv[8]),level=std::atof(argv[9]);
 const int motion=std::atoi(argv[10]);const double room=std::atof(argv[11]),late=std::atof(argv[12]),trim=std::atof(argv[13]),lateral=std::atof(argv[14]);
 const int model=std::atoi(argv[15]);const int frames=int(rate*std::atof(argv[16]));const int reset=std::atoi(argv[17]);
 require(rate>=8000&&rate<=192000&&block>0&&frames>0,"Fixture range");
 auto s=std::make_unique<DSPJSFX_State>();za::jsfx::StateVariables variables;variables.bind(*s,DSPJSFX_VARS_COUNT);
 s->srate=rate;s->memN=DSPJSFX_MAX_MEM_CELLS;s->mem=(DSPJSFX_Cell*)std::calloc(size_t(s->memN),sizeof(DSPJSFX_Cell));require(s->mem,"Memory allocation");
 jsfx_init(s.get());
 auto set=[&](int slider,double value){s->sliders[slider-1]=value;};
 const double defaults[]={0,0,1,0,1,0,.25,.82,.58,.10,.42,.30,0,.04,0,1,2,90,.55,1,0,0,.75,.45,10,0,0,.55,.72,0,0,0,0,0,1,0,1,1,0,1,0,0,0,4,1.6,0,.65};
 for(int i=0;i<47;++i)set(i+1,defaults[i]);
 set(6,lateral);set(7,distance);set(10,.65);set(12,room);set(14,0);set(15,trim);set(22,mode);set(23,1);set(27,late);set(36,model);set(46,amount);set(47,anti);
 set(48,0);set(49,argc==22?std::atof(argv[18]):0);set(50,argc==22?std::atof(argv[19]):0);set(51,argc==22?std::atof(argv[20]):0);set(52,argc==22?std::atof(argv[21]):.5);jsfx_slider(s.get());
 std::vector<float> l(frames),r(frames),ol(frames),orr(frames);double peak=0,energy=0,maxStep=0;float previous[2]{};
 for(int i=0;i<frames;++i){double t=double(i)/rate,a=texture(t,0),b=texture(t,1);
  if(fixture>=1&&fixture<=3){a=b=std::sin(6.283185307179586*(fixture==1?110:fixture==2?2500:9000)*t);}
  if(fixture==4){a=b=(i%(rate/5)==0?1:0);}
  if(fixture==5){double env=.5+.5*std::sin(6.283185307179586*3.1*t);a=b=.65*env*(std::sin(6.283185307179586*137*t)+.35*std::sin(6.283185307179586*1347*t));}
  if(fixture==6)a=b=0;
  if(fixture==7){a=.6*texture(t,0)+.4*texture(t-.071,1);b=.6*texture(t-.019,1)+.4*texture(t-.113,0);}
  if(fixture==8)b=-a;
  if(fixture==9)b=0;
  if(fixture==10)a=b=1;
  if(motion==2&&t>=1)a=b=0;
  l[i]=float(a*level);r[i]=float(b*level);
 }
 auto start=std::chrono::steady_clock::now();
 for(int at=0;at<frames;){
  int n=std::min(block,frames-at);
  if(motion==1){int phase=at/rate;n=std::min(n,rate-at%rate);set(7,phase%2?1:0);set(46,phase==2?0:amount);set(47,phase%2?1:anti);jsfx_slider(s.get());}
  if(motion==4){int phase=at/(rate/2);n=std::min(n,rate/2-at%(rate/2));set(49,phase%2?179:-179);set(51,phase%4==1?.5:phase%4==2?1:0);set(52,phase%2);jsfx_slider(s.get());}
  if(motion==3){n=std::min(n,rate-at%rate);set(46,at>=rate?0:amount);jsfx_slider(s.get());}
  if(reset&&at==rate*2){jsfx_init(s.get());set(7,distance);set(46,amount);jsfx_slider(s.get());}
  const float* in[]={l.data()+at,r.data()+at};float* out[]={ol.data()+at,orr.data()+at};jsfx_process_block(s.get(),in,out,2,n);
  require(!s->memoryFault,"DSP memory fault");at+=n;
 }
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 for(int ch=0;ch<2;++ch){const auto& samples=ch?orr:ol;for(float x:samples){require(std::isfinite(x),"Nonfinite output");peak=std::max(peak,std::abs(double(x)));energy+=x*x;maxStep=std::max(maxStep,std::abs(double(x)-previous[ch]));previous[ch]=x;}}
 std::ofstream audio(std::string(prefix)+".f32",std::ios::binary);audio.write((char*)ol.data(),frames*sizeof(float));audio.write((char*)orr.data(),frames*sizeof(float));
 std::ofstream report(std::string(prefix)+".json");report<<"{\"rate\":"<<rate<<",\"frames\":"<<frames<<",\"peak\":"<<peak<<",\"rms\":"<<std::sqrt(energy/(2*frames))<<",\"max_step\":"<<maxStep<<",\"process_seconds\":"<<seconds<<",\"restraint_db\":"<<observe(*s,"p72_last_reduction_db")<<",\"density\":"<<observe(*s,"p72_density")<<",\"guard\":"<<observe(*s,"p72_guard")<<",\"far\":"<<observe(*s,"p72_sm_far")<<",\"beam_direct_high\":"<<observe(*s,"p73_sm_direct_high")<<",\"beam_front_high\":"<<observe(*s,"p73_sm_er3_high")<<",\"internal\":"<<observe(*s,"p73_sm_internal")<<"}";
 std::free(s->mem);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
