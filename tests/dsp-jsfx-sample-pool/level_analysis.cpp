#include "SampleLevelAnalysis.h"
#include <iostream>
#include <limits>
#include <stdexcept>
using za::jsfx::analyzeSampleLevels;
static void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
static std::vector<float> tone(double rate,double seconds,double amplitude,double hz,int channels=1,bool opposite=false){
 std::vector<float> a(std::size_t(rate*seconds)*channels);
 for(std::size_t n=0;n<a.size()/channels;++n)for(int c=0;c<channels;++c)a[n*channels+c]=float(amplitude*std::sin(2*za::jsfx::level_detail::pi*hz*n/rate)*(opposite&&c==1?-1:1));return a;
}
int main()try{
 for(double rate:{44100.,48000.,96000.}){
  auto a=tone(rate,4,.1,997);auto x=analyzeSampleLevels(a.data(),a.size(),1,rate);
  require(std::abs(x[0]-.1/std::sqrt(2.))<.0001,"Whole-file RMS reference");
  require(std::abs(x[2]+20)<.08,"997 Hz dual-mono loudness calibration");require(x[6]==1,"Long signal integrated method");
  auto b=tone(rate,4,.05,997);auto y=analyzeSampleLevels(b.data(),b.size(),1,rate);
  require(std::abs((y[2]-x[2])+6.020599913)<.001,"6 dB loudness scale");
  auto s=tone(rate,.399,.1,997,2,true);auto z=analyzeSampleLevels(s.data(),s.size()/2,2,rate);
  require(z[0]>.06&&z[1]==0&&z[7]==0&&z[3]==-150,"Antiphase stereo is preserved, mono cancellation invalid");
 }
 std::vector<float> silent(24000);auto quiet=analyzeSampleLevels(silent.data(),silent.size(),1,48000);require(quiet[6]==0&&quiet[7]==0&&quiet[0]==0,"Silence invalid, no infinite boost");
 silent[20]=std::numeric_limits<float>::quiet_NaN();auto bad=analyzeSampleLevels(silent.data(),silent.size(),1,48000);require(bad[6]==0&&bad[0]==0,"Nonfinite input invalid");
 auto a=tone(48000,.399,.1,997);for(std::size_t n=2400;n<a.size();++n)a[n]=0;auto x=analyzeSampleLevels(a.data(),a.size(),1,48000);a.resize(std::size_t(48000*.401));auto y=analyzeSampleLevels(a.data(),a.size(),1,48000);
 require(x[6]==2&&y[6]==2&&std::abs(x[2]-y[2])<.001,"No 400 ms one-shot cliff");
 auto inter=tone(48000,.1,.5,12000);for(std::size_t n=0;n<inter.size();++n)inter[n]=float(.5*std::sin(za::jsfx::level_detail::pi*.5*n+za::jsfx::level_detail::pi*.25));
 auto p=analyzeSampleLevels(inter.data(),inter.size(),1,48000);require(p[4]>.49&&p[4]>.3536,"Intersample peak is detected");
 std::vector<float> tail(48000);tail.back()=.7f;auto t=analyzeSampleLevels(tail.data(),tail.size(),1,48000);require(t[4]>=.6999,"EOF peak included");
 std::vector<float> intro(48000);for(int n=24000;n<48000;++n)intro[n]=.2f;auto all=analyzeSampleLevels(intro.data(),intro.size(),1,48000);require(std::abs(all[0]-.2/std::sqrt(2.))<.00001,"Loud second half contributes to RMS");
 std::cout<<"PASS: loudness calibration at 44.1/48/96 kHz, gain scaling, stereo/mono cancellation, silence, NaN, short events, intersample and EOF peaks, full-file RMS\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
