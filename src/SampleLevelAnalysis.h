// SPDX-License-Identifier: Zlib
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <complex>
#include <cstring>

namespace za::jsfx {
// Fields 8..11 add centroid Hz, power above 6 kHz %, 95% power roll-off Hz,
// and an exact 52-bit PCM identity for saved exclusion state.
// Worker-only analysis of the channels Sample actually plays: L/R, or their
// average duplicated to L/R. Original PCM and existing peak/RMS are untouched.
// Fields: stereo RMS, mono RMS, stereo loudness, mono loudness, stereo true
// peak, mono true peak, stereo method, mono method. Method 0 invalid, 1 BS.1770
// integrated, 2 short-event estimate (including the transition to integrated).
using SampleLevels = std::array<double, 12>;
namespace level_detail {
constexpr double pi = 3.14159265358979323846;
struct Biquad {
    double b0=1,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
    double tick(double x) {
        const double y=b0*x+z1;z1=b1*x-a1*y+z2;z2=b2*x-a2*y;
        // Long quiet tails must not stall the loader on subnormal filter state.
        if(std::abs(z1)<1e-30)z1=0;if(std::abs(z2)<1e-30)z2=0;
        return y;
    }
};
inline std::array<Biquad,2> kFilters(double rate) {
    const double k=std::tan(pi*std::min(1681.974450955533,rate*.4)/rate);
    const double q=.7071752369554196,vh=std::pow(10.,3.999843853973347/20.),vb=std::pow(vh,.4996667741545416),a=1+k/q+k*k;
    Biquad shelf{(vh+vb*k/q+k*k)/a,2*(k*k-vh)/a,(vh-vb*k/q+k*k)/a,2*(k*k-1)/a,(1-k/q+k*k)/a};
    const double h=std::tan(pi*38.13547087602444/rate),hq=.5003270373238773,ha=1+h/hq+h*h;
    Biquad high{1,-2,1,2*(h*h-1)/ha,(1-h/hq+h*h)/ha};
    return {shelf,high};
}
inline double loudness(double energy) {return energy>0 ? -.691+10*std::log10(energy) : -150.;}
// Worker-only source descriptors. Channel spectra are combined after FFT, so
// anti-phase stereo is never mistaken for silence. No work runs in @gfx/audio.
inline void sourceDescriptors(const float* pcm,std::uint64_t frames,int channels,double rate,SampleLevels& result) {
    constexpr int n=2048,hop=1024;
    static const auto window=[] {std::array<double,n> w{};for(int i=0;i<n;++i)w[i]=.5-.5*std::cos(2*pi*i/(n-1));return w;}();
    std::array<std::complex<double>,n> data{};
    std::array<double,n/2+1> magnitude{},power{};
    for(std::int64_t center=0;center<static_cast<std::int64_t>(frames)+hop;center+=hop) {
        for(int channel=0;channel<std::min(channels,2);++channel) {
            for(int i=0;i<n;++i){const auto at=center+i-n/2;data[i]=at>=0&&at<static_cast<std::int64_t>(frames)?double(pcm[at*channels+channel])*window[i]:0.;}
            for(int i=1,j=0;i<n;++i){int bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(data[i],data[j]);}
            for(int len=2;len<=n;len<<=1){const auto step=std::polar(1.,-2*pi/len);for(int base=0;base<n;base+=len){std::complex<double> phase=1.;for(int j=0;j<len/2;++j){const auto a=data[base+j],b=data[base+j+len/2]*phase;data[base+j]=a+b;data[base+j+len/2]=a-b;phase*=step;}}}
            for(int k=0;k<=n/2;++k){const double p=std::norm(data[k]);power[k]+=p;magnitude[k]+=std::sqrt(p);}
        }
    }
    double mag=0,weighted=0,total=0,high=0;
    for(int k=0;k<=n/2;++k){const double f=k*rate/n;mag+=magnitude[k];weighted+=f*magnitude[k];total+=power[k];if(f>=6000)high+=power[k];}
    if(mag>0)result[8]=weighted/mag;
    if(total>0){result[9]=100*high/total;double running=0;for(int k=0;k<=n/2;++k){running+=power[k];if(running>=total*.95){result[10]=k*rate/n;break;}}}
}

struct Energy {double sum=0;std::uint64_t count=0;};
inline double gated(const std::vector<Energy>& blocks,double relativeDb) {
    const double absolute=std::pow(10.,(-70.+.691)/10.);double sum=0;std::uint64_t count=0;
    for(const auto& b:blocks) if(b.count&&b.sum/b.count>absolute){sum+=b.sum;count+=b.count;}
    if(!count)return 0.;const double gate=std::max(absolute,(sum/count)*std::pow(10.,relativeDb/10.));
    sum=0;count=0;for(const auto& b:blocks)if(b.count&&b.sum/b.count>gate){sum+=b.sum;count+=b.count;}
    return count?sum/count:0.;
}
// ITU-R BS.1770-5 Annex 2, 48-tap four-phase interpolation filter. Floating
// point needs no integer headroom attenuation. Include PCM peaks and flush
// the FIR tail; measurement introduces no playback latency.
constexpr double fir[4][12]={
 {.001708984375,.010986328125,-.0196533203125,.033203125,-.0594482421875,.1373291015625,.97216796875,-.102294921875,.047607421875,-.026611328125,.014892578125,-.00830078125},
 {-.0291748046875,.029296875,-.0517578125,.089111328125,-.16650390625,.465087890625,.77978515625,-.2003173828125,.1015625,-.0582275390625,.0330810546875,-.0189208984375},
 {-.0189208984375,.0330810546875,-.0582275390625,.1015625,-.2003173828125,.77978515625,.465087890625,-.16650390625,.089111328125,-.0517578125,.029296875,-.0291748046875},
 {-.00830078125,.014892578125,-.026611328125,.047607421875,-.102294921875,.97216796875,.1373291015625,-.0594482421875,.033203125,-.0196533203125,.010986328125,.001708984375}};
}

inline SampleLevels analyzeSampleLevels(const float* pcm,std::uint64_t frames,int channels,double rate) try {
    SampleLevels result{};result[2]=result[3]=-150.;
    if(!pcm||!frames||channels<1||!std::isfinite(rate)||rate<1000)return result;
    using namespace level_detail;
    auto left=kFilters(rate),right=kFilters(rate);
    const auto shortHop=std::max<std::uint64_t>(1,std::llround(rate*.02));
    const auto hop=std::max<std::uint64_t>(1,std::llround(rate*.1));
    std::array<std::vector<Energy>,2> shortBlocks,longBlocks;
    std::array<Energy,2> shortNow{},hopNow{};
    std::array<std::array<Energy,4>,2> recent{};
    std::array<std::array<double,12>,3> history{};
    std::array<double,3> peaks{};
    double rmsStereo=0,rmsMono=0;bool valid=true;std::uint64_t identity=1469598103934665603ull;
    std::uint64_t hopCount=0;int ring=0;
    for(std::uint64_t i=0;i<frames+11;++i){
        double l=0,r=0;
        if(i<frames){l=pcm[i*channels];r=channels>1?pcm[i*channels+1]:l;if(!std::isfinite(l)||!std::isfinite(r)){valid=false;l=r=0;}}
        const double m=.5*(l+r);const double xs[3]={l,r,m};
        for(int c=0;c<3;++c){
            history[c][ring]=xs[c];peaks[c]=std::max(peaks[c],std::abs(xs[c]));
            for(int p=0;p<4;++p){double y=0;for(int t=0;t<12;++t)y+=fir[p][t]*history[c][(ring-t+12)%12];peaks[c]=std::max(peaks[c],std::abs(y));}
        }
        ring=(ring+1)%12;
        if(i>=frames)continue;for(int c=0;c<std::min(channels,2);++c){std::uint32_t bits=0;std::memcpy(&bits,pcm+i*channels+c,sizeof(bits));identity=(identity^bits)*1099511628211ull;}
        rmsStereo+=.5*(l*l+r*r);rmsMono+=m*m;
        const double kl=left[1].tick(left[0].tick(l)),kr=right[1].tick(right[0].tick(r));
        const double e[2]={kl*kl+kr*kr,.5*(kl+kr)*(kl+kr)};
        for(int c=0;c<2;++c){shortNow[c].sum+=e[c];++shortNow[c].count;hopNow[c].sum+=e[c];++hopNow[c].count;}
        if((i+1)%shortHop==0||i+1==frames)for(int c=0;c<2;++c){shortBlocks[c].push_back(shortNow[c]);shortNow[c]={};}
        if((i+1)%hop==0){
            for(int c=0;c<2;++c){recent[c][hopCount%4]=hopNow[c];hopNow[c]={};if(hopCount>=3){Energy w{};for(const auto& part:recent[c]){w.sum+=part.sum;w.count+=part.count;}longBlocks[c].push_back(w);}}
            ++hopCount;
        }
    }
    if(!valid)return result;
    result[0]=std::sqrt(rmsStereo/frames);result[1]=std::sqrt(rmsMono/frames);
    result[4]=std::max(peaks[0],peaks[1]);result[5]=peaks[2];
    // A consistent 20 ms active-event estimate below 750 ms avoids a 400 ms
    // cliff for one-shots. Blend energy to standard integrated between .75 and
    // 1.25 s, so adding a few samples never abruptly changes the algorithm.
    const double blend=std::clamp((frames/rate-.75)/.5,0.,1.);
    for(int c=0;c<2;++c){const double event=gated(shortBlocks[c],-20.),integrated=gated(longBlocks[c],-10.);const double energy=integrated>0?event+(integrated-event)*blend:event;if(energy>0&&loudness(energy)>-70.){result[2+c]=loudness(energy);result[6+c]=(blend>=1&&integrated>0)?1:2;}}
    sourceDescriptors(pcm,frames,channels,rate,result);
    result[11]=double((identity^std::uint64_t(rate)^std::uint64_t(channels))&0x000fffffffffffffull);
    return result;
} catch (...) {
    // Optional descriptors must never make a load fail or terminate its worker
    // under memory pressure. Invalid metadata means unity compensation.
    return {0,0,-150,-150,0,0,0,0};
}
}
