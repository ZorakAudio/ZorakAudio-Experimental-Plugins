// SPDX-License-Identifier: Zlib
// Deterministic DSP-only comparison. Does not run or replace plugin @gfx.
#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include "JoepTestConfig.h"
#include "JsfxLegacyAtomics.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <chrono>
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
// Register the actual project WDL GFX function table so original @init
// definitions can contain uncalled drawing helpers. No @gfx is executed and
// this JUCE contract double is not used as a rendering/visual oracle.
#include "juce_contract_stub.h"
#include "YSFXGfxInterpreter.h"
#include "numeric_runtime.inc"
#include "WDL/eel2/ns-eel-int.h"
#ifdef EEL_TARGET_PORTABLE
#error Native benchmark must not use EEL_TARGET_PORTABLE
#endif
struct FpScope { int saved[2]; FpScope(){eel_enterfp(saved);} ~FpScope(){eel_leavefp(saved);} };
#ifdef JOEP_FULL_NATIVE
#include "NativeGfxPrototype.h"
#include "DspJsfxRuntime.h"

// Numerical runs use the identical native file callbacks on a non-realtime
// test thread. Production audio file loading is checked by the editor host.
static double fileCall(DSPJSFX_State* st,int opcode,std::initializer_list<double> values) {
  std::vector<double> args(values);std::vector<double*> refs;
  for(auto& value:args)refs.push_back(&value);
  return jsfx_native_gfx_dispatch(st,opcode,refs.data(),(int)refs.size());
}
#define ONE_FILE(name,OP) extern "C" double jsfx_file_##name(DSPJSFX_State* s,double a){return fileCall(s,DSPJSFX_FILE_##OP,{a});}
#define TWO_FILE(name,OP) extern "C" double jsfx_file_##name(DSPJSFX_State* s,double a,double b){return fileCall(s,DSPJSFX_FILE_##OP,{a,b});}
ONE_FILE(close,CLOSE) ONE_FILE(rewind,REWIND) ONE_FILE(avail,AVAIL) ONE_FILE(text,TEXT) ONE_FILE(multi_count,MULTI_COUNT)
TWO_FILE(open,OPEN) TWO_FILE(open_multi,OPEN_MULTI) TWO_FILE(seek,SEEK) TWO_FILE(multi_select,MULTI_SELECT)
extern "C" double jsfx_file_mem(DSPJSFX_State* s,double h,double d,double n){return fileCall(s,DSPJSFX_FILE_MEM,{h,d,n});}
extern "C" double jsfx_file_var(DSPJSFX_State* s,double h,double* out){double* a[]={&h,out};return jsfx_native_gfx_dispatch(s,DSPJSFX_FILE_VAR,a,2);}
extern "C" double jsfx_file_riff(DSPJSFX_State* s,double h,double* ch,double* sr){double* a[]={&h,ch,sr};return jsfx_native_gfx_dispatch(s,DSPJSFX_FILE_RIFF,a,3);}
extern "C" uint64_t jsfx_string_hash(DSPJSFX_State* s,double h){auto str=jsfx_native_gfx::stringContext(s)->read(h);uint64_t hash=14695981039346656037ull;for(unsigned char c:str){hash^=c;hash*=1099511628211ull;}return hash;}
extern "C" int jsfx_string_assign_utf8(DSPJSFX_State* s,double* out,const char* data,int n){if(!out)return 0;jsfxCellStore(out,jsfx_native_gfx::stringContext(s)->create(std::string(data,(size_t)n)));return n;}
#endif
// Short-message fixture adapter. Production editor tests exercise the JUCE
// MIDI host; here independent WDL/native guests consume identical event lists.
static std::vector<DSPJSFX_MidiEvent> oracleMidiInput, oracleMidiOutput;
static size_t oracleMidiRead;
static int fixtureBlockSize;
extern "C" int jsfx_midirecv(DSPJSFX_State* st,double* offset,double* a,double* b,double* c){
 if(st->midiInReadIndex>=st->midiInCount)return 0;
 auto e=st->midiIn[st->midiInReadIndex++];
 if(offset)jsfxCellStore(offset,e.sampleOffset);if(a)jsfxCellStore(a,e.msg1);
 if(b)jsfxCellStore(b,e.msg2);if(c)jsfxCellStore(c,e.msg3);return 1;
}
extern "C" int jsfx_midirecv_msg23(DSPJSFX_State* st,double* offset,double* a,double* packed){
 double b=0,c=0;if(!jsfx_midirecv(st,offset,a,&b,&c))return 0;
 if(packed)jsfxCellStore(packed,b+256*c);return 1;
}
static DSPJSFX_MidiEvent shortEvent(double offset,double a,double b,double c){
 auto byte=[](double v){return std::clamp(int(std::llround(v)),0,255);};
 return {std::clamp(int(std::llround(offset)),0,std::max(0,fixtureBlockSize-1)),byte(a),byte(b),byte(c)};
}
extern "C" int jsfx_midisend(DSPJSFX_State* st,double offset,double a,double b,double c){
 if(st->midiOutCount>=st->midiOutCapacity)return 0;
 auto e=shortEvent(offset,a,b,c);st->midiOut[st->midiOutCount++]=e;return e.msg1;
}
extern "C" int jsfx_midisend_msg23(DSPJSFX_State* st,double offset,double a,double packed){
 int x=int(std::llround(packed));return jsfx_midisend(st,offset,a,x&255,(x>>8)&255);
}
static EEL_F NSEEL_CGEN_CALL oracleRecv(void*,INT_PTR n,EEL_F** args){
 if(oracleMidiRead>=oracleMidiInput.size())return 0;
 auto e=oracleMidiInput[oracleMidiRead++];*args[0]=e.sampleOffset;*args[1]=e.msg1;
 *args[2]=n==3?e.msg2+256*e.msg3:e.msg2;if(n>=4)*args[3]=e.msg3;return 1;
}
static EEL_F NSEEL_CGEN_CALL oracleSend(void*,INT_PTR n,EEL_F** args){
 double b=*args[2],c=n>=4?*args[3]:((int(std::llround(b))>>8)&255);
 if(n==3)b=int(std::llround(b))&255;
 auto e=shortEvent(*args[0],*args[1],b,c);oracleMidiOutput.push_back(e);return e.msg1;
}
#include "JsfxMemoryBuiltins.h"
#include "host_slider_runtime.inc"
struct Oracle {
  std::array<double, DSPJSFX_MAX_SLIDERS> sliders{};
  std::map<std::string, int> aliases;
  static EEL_F *resolve(void *opaque, const char *name) {
    auto &o = *static_cast<Oracle *>(opaque);
    std::string n(name);
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    auto alias = o.aliases.find(n);
    if (alias != o.aliases.end()) return &o.sliders[alias->second];
    if (n.starts_with("slider") && n.size() > 6 &&
        n.find_first_not_of("0123456789", 6) == std::string::npos) {
      int i = std::stoi(n.substr(6)) - 1;
      if (i >= 0 && i < DSPJSFX_MAX_SLIDERS) return &o.sliders[i];
    }
    return nullptr;
  }
};
static Oracle *activeOracle;
static DSPJSFX_State oracleNotifications{};za::jsfx::StateVariables oracleNotifications_variables;static const bool oracleNotifications_bound=[] {oracleNotifications_variables.bind(oracleNotifications,DSPJSFX_VARS_COUNT);return true;}();
static EEL_F NSEEL_CGEN_CALL oracleSlider(void *, EEL_F *index) {
  int i = int(*index + 1.e-5) - 1;
  return i >= 0 && i < DSPJSFX_MAX_SLIDERS ? activeOracle->sliders[i] : 0.;
}
static EEL_F NSEEL_CGEN_CALL oracleNext(void *ctx, EEL_F *index, EEL_F *out) {
  *out = oracleSlider(ctx, index);
  return -1.;
}
static EEL_F NSEEL_CGEN_CALL oracleNotify(void *, INT_PTR count, EEL_F **args) {
  int direct = -1;
  for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
    if (args[0] == &activeOracle->sliders[i]) direct = i;
  return jsfx_slider_automate(&oracleNotifications, *args[0], direct,
                            count > 1 ? *args[1] : 0.);
}
static std::map<std::string, std::string> sections(const char *file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot read expanded JSFX source");
  std::map<std::string, std::string> result;
  std::string line, section;
  while (std::getline(in, line)) {
    auto pos = line.find_first_not_of(" \t\r");
    if (pos != std::string::npos && line[pos] == '@') {
      section = line.substr(pos + 1);
      section = section.substr(0, section.find_first_of(" \t\r"));
    } else if (!section.empty()) result[section] += line + '\n';
  }
  return result;
}
int main(int argc, char **argv) try {
  if (argc != 5) throw std::runtime_error("usage: joep_dsp_reference expanded.jsfx rate defaults|automation double|float");
  const double rate = std::stod(argv[2]);
  const bool automate = std::string(argv[3]) == "automation";
  const bool floatBlocks = std::string(argv[4]) == "float";
  const int frames = int(rate * (std::getenv("ZA_JOEP_SECONDS") ? std::stod(std::getenv("ZA_JOEP_SECONDS")) : 8.));
  const int blockSize=std::getenv("ZA_BENCH_BLOCK")?std::stoi(std::getenv("ZA_BENCH_BLOCK")):512;
  const int warmup=int(rate);
  const bool reverse=std::getenv("ZA_BENCH_REVERSE")!=nullptr;
  const bool fpPerCall=std::getenv("ZA_BENCH_FP_PER_CALL")!=nullptr;
  FpScope fpScope;
  if(!floatBlocks || blockSize<1 || blockSize>512)throw std::runtime_error("Benchmark requires float blocks of 1..512");
  Oracle oracle;
  for (const auto &s : joepSliders) {
    oracle.sliders[s.index] = s.value;
    if (s.alias[0]) oracle.aliases[s.alias] = s.index;
  }
  activeOracle = &oracle;
  jsfx_gfx::GfxVm ref;
  NSEEL_addfunc_retval("slider", 1, NSEEL_PProc_THIS, oracleSlider);
  NSEEL_addfunc_retval("slider_next_chg", 2, NSEEL_PProc_THIS, oracleNext);
  NSEEL_addfunc_varparm_ex("slider_automate", 1, 0, NSEEL_PProc_THIS,
                         oracleNotify, nullptr);
  NSEEL_addfunc_varparm_ex("midirecv",3,0,NSEEL_PProc_THIS,oracleRecv,nullptr);
  NSEEL_addfunc_varparm_ex("midisend",3,0,NSEEL_PProc_THIS,oracleSend,nullptr);
  NSEEL_VM_set_var_resolver(ref.m_vm, Oracle::resolve, &oracle);
  NSEEL_VM_setramsize(ref.m_vm, DSPJSFX_MAX_MEM_CELLS);
  auto variable = [&](const char *name) {
    auto *resolved = Oracle::resolve(&oracle, name);
    return resolved ? resolved : NSEEL_VM_regvar(ref.m_vm, name);
  };
  // Assert the reference really binds named aliases to the same slider cell.
  for (const auto &s : joepSliders) {
    if (s.alias[0]) {
      const int index = oracle.aliases[s.alias];
      const double saved = oracle.sliders[index];
      const std::string probe = std::string(s.alias) + "+=1;slider" +
                                std::to_string(index + 1) + "+=1;";
      const char *error = nullptr;
      auto handle = ref.compile_code(probe.c_str(), &error);
      if (!handle) throw std::runtime_error("WDL alias probe did not compile");
      NSEEL_code_execute(handle);
      if (std::abs(oracle.sliders[index] - (saved + 2.)) > 1.e-12)
        throw std::runtime_error("WDL compiled aliases did not bind canonical cells");
      oracle.sliders[index] = saved;
    }
  }
  *variable("srate") = rate;
  *variable("samplesblock") = 64;
  *variable("num_ch") = DSPJSFX_PROCESS_CHANNELS;
  *variable("tempo") = 120;
  *variable("play_state") = 1;
  const auto source = sections(argv[1]);
  std::map<std::string, NSEEL_CODEHANDLE> handles;
  for (auto section : {"init", "slider", "block", "sample"}) {
    const char *error = nullptr;
    const auto it = source.find(section);
    handles[section] = ref.compile_code(it == source.end() || it->second.empty()
                                          ? "0;" : it->second.c_str(), &error);
    if (!handles[section]) throw std::runtime_error(std::string("WDL @") + section + ": " + (error ? error : "compile failed"));
  }
  if(!fpPerCall)for(auto& [name,handle]:handles)static_cast<codeHandleType*>(handle)->compile_flags|=NSEEL_CODE_COMPILE_FLAG_NOFPSTATE;
  std::vector<DSPJSFX_Cell> memory(DSPJSFX_MAX_MEM_CELLS);
  std::mutex atomicMutex;
  DSPJSFX_State st{};za::jsfx::StateVariables st_variables;st_variables.bind(st,DSPJSFX_VARS_COUNT);
  st.mem = memory.data(); st.memN = memory.size(); st.srate = rate;
  st.samplesblock = 64; st.atomicContext = &atomicMutex;
  for (const auto &v : DSPJSFX_VARS) {
    if (std::string(v.name) == "num_ch") st.vars[v.index] = DSPJSFX_PROCESS_CHANNELS;
    if (std::string(v.name) == "tempo") st.vars[v.index] = 120;
    if (std::string(v.name) == "play_state") st.vars[v.index] = 1;
  }
  for (const auto &s : joepSliders) st.sliders[s.index] = s.value;
#ifdef JOEP_FULL_NATIVE
  za::jsfx::DspJsfxRuntime comm;comm.attachToState(&st);
  jsfx_native_gfx::Frame graphics;st.nativeStrings=&graphics.ownedStrings;graphics.bindLegacy(st,DSPJSFX_NATIVE_GFX_WIDTH,DSPJSFX_NATIVE_GFX_HEIGHT);graphics.frameRecording=false;
  for(auto name:{"gfx_r","gfx_g","gfx_b","gfx_a","gfx_a2"})graphics.set(name,1);
  graphics.set("gfx_texth",8);

  {std::ifstream input(argv[1]);std::stringstream text;text<<input.rdbuf();graphics.configureResources(text.str().c_str());}
#endif
  if(std::getenv("ZA_JOEP_DIAG"))std::cerr<<"STAGE native-init\n";
  jsfx_init(&st);
  if(std::getenv("ZA_JOEP_DIAG"))std::cerr<<"STAGE WDL-init\n";
  NSEEL_code_execute(handles["init"]);
  jsfx_slider(&st); NSEEL_code_execute(handles["slider"]);
  const bool drumsMultichannel=std::getenv("ZA_JOEP_DRUMS_MULTI") && std::string(joepPluginName)=="joep_saikedrums";
  if(drumsMultichannel)for(auto name:{"multi_out","uncoupled_hats"}) {
    bool found=false;
    for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name){
      const int alias=DSPJSFX_LEGACY_SLIDER_ALIASES[v.index];
      if(alias>=0)st.sliders[alias]=1;else st.vars[v.index]=1;
      found=true;
    }
    if(!found)throw std::runtime_error("Missing drums multichannel fixture option");
    *variable(name)=1;
  }
  bool activeArpPattern=false;
  // The default arpeggiator bank is empty. Program the same four steps in
  // each independent guest so the MIDI comparison exercises actual note
  // generation, without editing the plugin's source or compiled object.
  if (std::string(joepPluginName)=="joep_saike_midi_arp") {
    auto nativeVariable=[&](const char* name)->double {
      for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return double(st.vars[v.index]);
      throw std::runtime_error(std::string("Missing arp fixture variable: ")+name);
    };
    auto nativeBase=int64_t(nativeVariable("pattern_buffer"));
    auto oracleBase=int64_t(*variable("pattern_buffer"));
    for(int i=0;i<4;++i) {
      if(nativeBase+i<0 || nativeBase+i>=st.memN)throw std::runtime_error("Arp fixture memory bound");
      st.mem[nativeBase+i]=16; // Note on, probability nibble zero (always).
      auto* cell=NSEEL_VM_getramptr(ref.m_vm,int(oracleBase+i),nullptr);
      if(!cell)throw std::runtime_error("WDL arp fixture memory unavailable");
      *cell=16;
    }
    activeArpPattern=true;
  }
  auto diagnose = [&](const char *stage) {
    if (!std::getenv("ZA_JOEP_DIAG")) return;
    int differences = 0;
    for (const auto &v : DSPJSFX_VARS) {
      if (std::string(v.name).starts_with("__fnlocal__") ||
          DSPJSFX_LEGACY_SLIDER_ALIASES[v.index] >= 0) continue;
      auto *p = NSEEL_VM_getvar(ref.m_vm, v.name);
      if (p && std::abs(double(st.vars[v.index]) - *p) > 1.e-11 && differences++ < 20)
        std::cerr << std::setprecision(17) << stage << ' ' << v.name
                  << " native=" << double(st.vars[v.index]) << " WDL=" << *p << '\n';
    }
  };
  diagnose("INIT");
  std::array<EEL_F *, 64> spl{};
  for (int c = 0; c < DSPJSFX_PROCESS_CHANNELS; ++c)
    spl[c] = variable(("spl" + std::to_string(c)).c_str());
  std::array<DSPJSFX_MidiEvent,1024> nativeMidiOutput{};
  st.midiOut=nativeMidiOutput.data();st.midiOutCapacity=nativeMidiOutput.size();
  int midiInputEvents=0,midiOutputEvents=0,midiDifferences=0;
  std::vector<DSPJSFX_MidiEvent> scheduled;
  for(int event=0;event<int((frames+warmup)/rate*5);++event){
    int at=int(rate*.2*event);
    scheduled.push_back({at,event%2?0x80:0x90,60+(event/2)*2,event%2?0:96});
    if(event==2)scheduled.push_back({at+7,0xB0,1,80});
    if(event==4)scheduled.push_back({at+11,0xE0,0,70});
  }
  if(drumsMultichannel) {
    scheduled.clear();
    const int notes[]={60,62,64,65,66,67,68,69,70,71,72,73};
    for(int i=0;i<12;++i) {
      scheduled.push_back({int(rate*.12*i),0x90,notes[i],96});
      scheduled.push_back({int(rate*(.12*i+.08)),0x80,notes[i],0});
    }
  }
  double nativeSeconds=0,referenceSeconds=0;
  int phase=0,blocks=0,measuredBlocks=0,differing=0,exactDifferences=0,first=-1;
  double maxError=0,errorEnergy=0,energy=0,peak=0;
  std::array<std::array<float,512>,64> input{},output{},referenceOutput{};
  std::array<const float*,64> inputPtrs{};
  std::array<float*,64> outputPtrs{};
  for(int c=0;c<DSPJSFX_PROCESS_CHANNELS;++c){inputPtrs[c]=input[c].data();outputPtrs[c]=output[c].data();}
  auto signal=[&](int frame,int channel){
    if(drumsMultichannel&&channel>=DSPJSFX_INPUT_CHANNELS)return 0.;
    return .18*std::sin(frame*2*3.141592653589793*(317+channel*113)/rate)+
           .037*std::cos(frame*2*3.141592653589793*(2801+channel*151)/rate)+
           (frame%48000==0?.31:0.)+(frame%48000==197?-.27:0.);
  };
  for(int begin=0;begin<frames+warmup;){
    const int n=std::min(frames+warmup-begin,blockSize);
    st.samplesblock=n;*variable("samplesblock")=n;
    fixtureBlockSize=n;st.currentBlockSize=n;
    oracleMidiInput.clear();oracleMidiOutput.clear();oracleMidiRead=0;st.midiOutCount=0;
    for(auto event:scheduled)if(event.sampleOffset>=begin&&event.sampleOffset<begin+n){event.sampleOffset-=begin;oracleMidiInput.push_back(event);}
    midiInputEvents+=oracleMidiInput.size();
    st.midiIn=oracleMidiInput.data();st.midiInCount=(int)oracleMidiInput.size();st.midiInReadIndex=0;
    for(auto name:{"play_position","beat_position"}){
      double value=begin/rate*(std::string(name)=="beat_position"?2:1);
      *variable(name)=value;
      for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)st.vars[v.index]=value;
    }
    const int nextPhase=int(begin/(rate*.5));
    if(automate&&nextPhase!=phase){
      phase=nextPhase;
      for(const auto& s:joepChanges){const double value=phase%2?s.value:joepDefault(s.index);st.sliders[s.index]=value;oracle.sliders[s.index]=value;}
      jsfx_slider(&st);NSEEL_code_execute(handles["slider"]);
    }
    // Input generation, validation, transport, MIDI queues, and parameter
    // changes are outside both timed regions. WDL includes its required
    // per-sample spl marshalling; LLVM uses its production block entrypoint.
    for(int c=0;c<DSPJSFX_PROCESS_CHANNELS;++c)for(int k=0;k<n;++k)input[c][k]=float(signal(begin+k,c));
    auto runNative=[&]{
      const auto start=std::chrono::steady_clock::now();
      jsfx_process_block(&st,inputPtrs.data(),outputPtrs.data(),DSPJSFX_PROCESS_CHANNELS,n);
      const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(begin>=warmup)nativeSeconds+=elapsed;
    };
    auto runReference=[&]{
      const auto start=std::chrono::steady_clock::now();
      NSEEL_code_execute(handles["block"]);
      for(int k=0;k<n;++k){
        for(int c=0;c<DSPJSFX_PROCESS_CHANNELS;++c)*spl[c]=double(input[c][k]);
        NSEEL_code_execute(handles["sample"]);
        for(int c=0;c<DSPJSFX_OUTPUT_CHANNELS;++c)referenceOutput[c][k]=float(*spl[c]);
      }
      const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(begin>=warmup)referenceSeconds+=elapsed;
    };
    if(((blocks&1)!=0)^reverse){runReference();runNative();}else{runNative();runReference();}
    for(int c=0;c<DSPJSFX_OUTPUT_CHANNELS;++c)for(int k=0;k<n;++k){
      const double a=output[c][k],b=referenceOutput[c][k];
      if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite audio at frame "+std::to_string(begin+k));
      const double d=std::abs(a-b);maxError=std::max(maxError,d);errorEnergy+=d*d;energy+=b*b;peak=std::max(peak,std::abs(a));
      if(a!=b)++exactDifferences;
      if(d>2.e-7){++differing;if(first<0){first=begin+k;std::cerr<<std::setprecision(17)<<"FIRST frame="<<first<<" channel="<<c<<" native="<<a<<" WDL="<<b<<'\n';}}
    }
    midiOutputEvents+=st.midiOutCount;
    if((size_t)st.midiOutCount!=oracleMidiOutput.size())++midiDifferences;
    else for(int i=0;i<st.midiOutCount;++i){auto a=st.midiOut[i],b=oracleMidiOutput[i];if(a.sampleOffset!=b.sampleOffset||a.msg1!=b.msg1||a.msg2!=b.msg2||a.msg3!=b.msg3)++midiDifferences;}
    if(st.memoryFault)throw std::runtime_error("native memoryFault flag set");
    if(begin>=warmup)++measuredBlocks;
    begin+=n;++blocks;
  }
  const int measuredFrames=measuredBlocks*blockSize; // rounded below using total duration
  const int firstMeasured=((warmup+blockSize-1)/blockSize)*blockSize;
  const int actualMeasured=frames+warmup-firstMeasured;
  std::cout<<std::setprecision(12)<<"{\"rate\":"<<rate<<",\"frames\":"<<actualMeasured
           <<",\"warmup_frames\":"<<firstMeasured<<",\"block_size\":"<<blockSize<<",\"blocks\":"<<measuredBlocks
           <<",\"channels\":"<<DSPJSFX_OUTPUT_CHANNELS<<",\"backend\":\"WDL native x64 SSE JIT\",\"fp_per_call\":"<<(fpPerCall?"true":"false")
           <<",\"profile\":\""<<(automate?"automation":"defaults")<<"\",\"native_cpu_seconds\":"<<nativeSeconds<<",\"wdl_cpu_seconds\":"<<referenceSeconds
           <<",\"speedup\":"<<referenceSeconds/nativeSeconds<<",\"differing_samples\":"<<differing<<",\"exact_differences\":"<<exactDifferences
           <<",\"first_difference\":"<<first<<",\"max_error\":"<<maxError<<",\"relative_rms_error\":"<<std::sqrt(errorEnergy/std::max(energy,1.e-30))
           <<",\"midi_input_events\":"<<midiInputEvents<<",\"midi_output_events\":"<<midiOutputEvents<<",\"midi_differences\":"<<midiDifferences
           <<",\"peak\":"<<peak<<",\"reference_energy\":"<<energy<<"}\n";
  return midiDifferences||(activeArpPattern&&!midiOutputEvents)?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
