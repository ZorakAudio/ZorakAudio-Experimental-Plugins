// SPDX-License-Identifier: Zlib
// Actual generated native DSP vs the repository's portable WDL/EEL2 engine.
// FFT/memcpy below are extracted verbatim from the production processor TU.
#include "JSFXDSP.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define EEL_TARGET_PORTABLE 1
#define EELSCRIPT_NO_LICE 1
#define EELSCRIPT_NO_NET 1
#define EELSCRIPT_NO_FILE 1
#define EELSCRIPT_NO_MDCT 1
#define EELSCRIPT_NO_PREPROC 1
#include "WDL/eel2/eelscript.h"
static std::mutex eelMutex;
extern "C" void NSEEL_HOSTSTUB_EnterMutex() { eelMutex.lock(); }
extern "C" void NSEEL_HOSTSTUB_LeaveMutex() { eelMutex.unlock(); }
#include "numeric_runtime.inc"
extern "C" void jsfx_ensure_mem(DSPJSFX_State* st,int64_t needed)
{
    // Tests preallocate the entire 8M-cell address space. No host/allocation
    // behaviour is being simulated or asserted by the numerical comparison.
    if(needed<=0 || !st->mem || needed>st->memN) {
        st->memoryFault=1;
        throw std::runtime_error("native test memory bound exceeded");
    }
    st->memUsed=std::max(st->memUsed,needed);
}
static int variable(const std::string& name)
{
    for(int i=0;i<DSPJSFX_VARS_COUNT;++i)if(name==DSPJSFX_VARS[i].name)return i;
    return -1;
}
static std::map<std::string,std::string> sections(const char* file)
{
    std::ifstream in(file);if(!in)throw std::runtime_error("Cannot read expanded source");
    std::map<std::string,std::string> r; std::string line,section;
    while(std::getline(in,line)) {
        auto pos=line.find_first_not_of(" \t\r");
        if(pos!=std::string::npos && line[pos]=='@') {
            section=line.substr(pos+1);section=section.substr(0,section.find_first_of(" \t\r"));
        } else if(!section.empty())r[section]+=line+'\n';
    }
    return r;
}
int main(int argc,char**argv)
try {
    if(argc<2)throw std::runtime_error("usage: dsp_reference expanded.jsfx [rate] [seconds] [mode] [profile] [sample|float]");
    const double sr=argc>2?std::stod(argv[2]):48000;
    const int count=int(sr*(argc>3?std::stod(argv[3]):3));
    const int mode=argc>4?std::stoi(argv[4]):0;
    const std::string profile=argc>5?argv[5]:"stress";
    const bool floatBlocks=argc>6 && std::string(argv[6])=="float";
    if(sr<8000||sr>96000||count<sr*2||count>sr*30||mode<0||mode>2)
        throw std::runtime_error("Test requires 8-96 kHz, 2-30 seconds, mode 0..2");
    if(profile!="stress"&&profile!="defaults"&&profile!="dry"&&profile!="automation")
        throw std::runtime_error("Unknown test profile");
    NSEEL_init();eelScriptInst::init();eelScriptInst ref;
    NSEEL_VM_setramsize(ref.m_vm,8*1024*1024);
    auto var=[&](const char*name){return NSEEL_VM_regvar(ref.m_vm,name);};
    std::vector<double> mem(8*1024*1024);
    DSPJSFX_State st{};st.mem=mem.data();st.memN=int64_t(mem.size());st.srate=sr;st.samplesblock=64;
    *var("srate")=sr;*var("samplesblock")=64;
    auto code=sections(argv[1]); std::map<std::string,NSEEL_CODEHANDLE> handles;
    for(auto name:{"init","slider","block","sample"}) {
        const char*error=nullptr;handles[name]=ref.compile_code(code[name].empty()?"0;":code[name].c_str(),&error);
        if(!handles[name])throw std::runtime_error(std::string("EEL @")+name+": "+(error?error:"compile failure"));
    }
    struct Slider{const char*alias;double value;};
    Slider defaults[]={{"diffusion",.6},{"current_verb_decay",.8},{"verb_mod_depth",.1},
        {"current_verb_mod_rate",.3},{"current_verb_lowpass",1},{"current_verb_highpass",0},
        {"shimmer",.2},{"drop_mode",double(mode)},{"drops",.3},{"nonlinearity",.15},{"wet",.65}};
    if(profile=="defaults"){defaults[8].value=0;defaults[9].value=0;defaults[10].value=.25;}
    if(profile=="dry")defaults[10].value=0;
    auto sliders=[&](){for(size_t i=0;i<std::size(defaults);++i){
        const auto&d=defaults[i];st.sliders[i]=d.value;*var(("slider"+std::to_string(i+1)).c_str())=d.value;
        int idx=variable(d.alias);if(idx>=0)st.vars[idx]=d.value;*var(d.alias)=d.value;
    }};
    sliders();jsfx_init(&st);NSEEL_code_execute(handles["init"]);sliders();
    jsfx_slider(&st);NSEEL_code_execute(handles["slider"]);
    // Compare selected namespaced initialisation values before any audio runs.
    int initBad=0;
    for(auto name:{"freemem","particles","shifter.buffer.scopebuffer","shifter_up.buffer.scopebuffer",
                   "shifter_shimmer.buffer.scopebuffer","verb.diffuser1.offset","verb.diffuser2.offset"}){
        int idx=variable(name);double native=idx>=0?st.vars[idx]:0,expected=*var(name);
        if(std::abs(native-expected)>1e-9){std::cerr<<"INIT "<<name<<" native="<<native<<" EEL="<<expected<<'\n';++initBad;}
    }
    auto*left=var("spl0");auto*right=var("spl1");
    std::vector<std::pair<int,EEL_F*>> sharedVars;
    for(int j=0;j<DSPJSFX_VARS_COUNT;++j)if(std::string(DSPJSFX_VARS[j].name).find("__fnlocal__")!=0)
        sharedVars.push_back({j,var(DSPJSFX_VARS[j].name)});
    bool printedState=false;
    double maxError=0,referenceEnergy=0,errorEnergy=0,tailEnergy=0,dryError=0,peak=0;
    int bad=0,first=-1,blockNumber=0,automationPhase=0;
    const int sizes[]={1,17,64,511,128,33};
    for(int begin=0;begin<count;){
        const int n=std::min(count-begin,floatBlocks?sizes[blockNumber%6]:64);
        st.samplesblock=n;*var("samplesblock")=n;
        if(profile=="automation"){
            const int phase=int(begin/(sr*.7));
            if(phase!=automationPhase){
                automationPhase=phase;
                defaults[0].value=phase%2?.75:.3;
                defaults[1].value=phase%2?.65:.85;
                defaults[3].value=phase%2?.7:.15;
                defaults[4].value=phase%2?.8:.95;
                defaults[5].value=phase%2?.04:.01;
                defaults[6].value=phase%2?.35:.05;
                defaults[7].value=phase%3;
                defaults[8].value=.45;
                defaults[9].value=phase%2?.3:0;
                defaults[10].value=phase%2?.9:.4;
                sliders();jsfx_slider(&st);NSEEL_code_execute(handles["slider"]);
            }
        }
        std::array<float,511> inputL{},inputR{},outputL{},outputR{};
        auto signal=[&](int i,int c){
            if(i>=int(sr))return 0.0;
            return c?(.17*std::sin(i*2*3.141592653589793*773/sr)+(i==197?.2:0)):
                     (.2*std::sin(i*2*3.141592653589793*317/sr)+(i==0?.3:0));
        };
        if(floatBlocks){
            for(int k=0;k<n;++k){inputL[k]=float(signal(begin+k,0));inputR[k]=float(signal(begin+k,1));}
            const float* inputs[]={inputL.data(),inputR.data()};
            float* outputs[]={outputL.data(),outputR.data()};
            jsfx_process_block(&st,inputs,outputs,2,n);
        }else jsfx_block(&st);
        NSEEL_code_execute(handles["block"]);
        for(int k=0;k<n;++k){
            const int i=begin+k;
            const double inL=floatBlocks?inputL[k]:signal(i,0),inR=floatBlocks?inputR[k]:signal(i,1);
            *left=inL;*right=inR;
            if(!floatBlocks){st.spl[0]=inL;st.spl[1]=inR;jsfx_sample(&st);}
            NSEEL_code_execute(handles["sample"]);
            if(!floatBlocks&&std::getenv("ZA_SHOWCASE_DIAG")&&!printedState){
                int differences=0;
                for(auto [j,p]:sharedVars)if(std::abs(st.vars[j]-*p)>1e-11){
                    if(differences++<12)std::cerr<<std::setprecision(17)<<"STATE at "<<i<<" "<<DSPJSFX_VARS[j].name<<" native="<<st.vars[j]<<" EEL="<<*p<<'\n';
                }
                if(differences)printedState=true;
            }
            for(int c=0;c<2;++c){
                const double a=floatBlocks?(c?outputR[k]:outputL[k]):st.spl[c];
                const double b=floatBlocks?double(float(c?*right:*left)):(c?*right:*left);
                const double d=std::abs(a-b);
                if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite audio at sample "+std::to_string(i));
                maxError=std::max(maxError,d);referenceEnergy+=b*b;errorEnergy+=d*d;peak=std::max(peak,std::abs(a));
                if(i>int(sr))tailEnergy+=a*a;
                if(profile=="dry")dryError=std::max(dryError,std::abs(a-(c?inR:inL)));
                if(d>2e-7){++bad;if(first<0){first=i;std::cerr<<std::setprecision(17)<<"FIRST "<<i<<" ch="<<c<<" native="<<a<<" EEL="<<b<<'\n';}}
            }
        }
        begin+=n;++blockNumber;
    }
    std::cout<<std::setprecision(12)<<"rate="<<sr<<" frames="<<count<<" mode="<<mode
      <<" profile="<<profile<<" path="<<(floatBlocks?"float_blocks":"double_samples")<<" init_mismatches="<<initBad<<" differing_samples="<<bad<<" max_error="<<maxError
      <<" relative_error="<<std::sqrt(errorEnergy/std::max(1e-30,referenceEnergy))<<" tail_energy="<<tailEnergy<<" dry_error="<<dryError<<" peak="<<peak<<" blocks="<<blockNumber<<'\n';
    return initBad||bad||(profile=="dry"?dryError>2e-7:tailEnergy<=1e-12) ? 1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
