#include "SampleLevelAnalysis.h"
#include <iostream>
#include <stdexcept>
static void require(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
int main()try{
 for(double rate:{44100.,48000.,96000.}){
  std::vector<float> samples(std::size_t(rate*.5)*2);
  for(std::size_t n=0;n<samples.size()/2;++n){samples[n*2]=float(.2*std::sin(2*za::jsfx::level_detail::pi*997*n/rate));samples[n*2+1]=-samples[n*2];}
  auto low=za::jsfx::analyzeSampleLevels(samples.data(),samples.size()/2,2,rate);
  require(std::abs(low[8]-997)<60,"Stereo spectral centroid calibration");require(low[9]<2,"Low tone is outside upper band");require(std::abs(low[10]-997)<120,"Low tone roll-off");
  auto again=za::jsfx::analyzeSampleLevels(samples.data(),samples.size()/2,2,rate);require(low[11]==again[11]&&low[11]>0,"Stable source identity");
  for(std::size_t n=0;n<samples.size()/2;++n){samples[n*2]=float(.2*std::sin(2*za::jsfx::level_detail::pi*12000*n/rate));samples[n*2+1]=-samples[n*2];}
  auto high=za::jsfx::analyzeSampleLevels(samples.data(),samples.size()/2,2,rate);
  require(std::abs(high[8]-12000)<100,"Upper tone centroid calibration");require(high[9]>98,"Upper tone energy");require(std::abs(high[10]-12000)<180,"Upper tone roll-off");require(high[11]!=low[11],"Changed source identity");
 }
 std::cout<<"PASS: centroid, high-frequency energy, roll-off, anti-phase stereo and source identity at 44.1/48/96 kHz\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
