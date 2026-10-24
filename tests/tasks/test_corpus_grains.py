"""Native rendering regression checks using the actual Corpus JSFX functions.

Compiles a small fixture with the repository compiler in Native Legacy mode.
No recordings are loaded: the sample host supplies deterministic test signals.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def function(source, name):
    start = source.index('function ' + name + '(')
    end = source.index('\n);', start) + len('\n);')
    # Several foundational functions occupy one line.
    line_end = source.index('\n', start)
    if source[start:line_end].rstrip().endswith(');'):
        end = line_end
    return source[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path)
    ap.add_argument('--repo', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    root = args.repo or Path(__file__).resolve().parents[2]
    source_path = args.source or root / 'plugins/Spectral/Corpus/src/Corpus.jsfx'
    out = args.out or root / 'build/tasks/corpus-grains'
    out.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding='utf-8')
    sliders = '\n'.join(line for line in source.splitlines()
                        if any(line.startswith(f'slider{i}:') for i in list(range(65, 81))+[249]))
    helpers = '\n'.join(function(source, name) for name in (
        'cp_clamp', 'cp_01', 'cp_db', 'cp_amp', 'cp_fold', 'cp_g', 'cp_s', 'cp_e', 'cp_v',
        'gr_prepare', 'gr_window', 'gr_accumulate', 'gr_level',
        'pt_config', 'pt_center', 'pt_read', 'pt_walk_rate',
        'gt_gain', 'gt_guard',
        'cc_begin','cc_grain_begin','cc_feed','cc_grain_end','cc_prepare','cc_voice_begin','cc_apply',
        'cp_voice', 'cp_record_voice', 'cp_replay_step',
        'ui_notify', 'ui_end_gesture'))
    mixer = source.split('@sample\n', 1)[1].split(
        '// Transient Diffusion is deliberately post-selection.', 1)[0]
    fixture = sliders + '\noptions:maxmem=100000\n@init\n' + '''
function cp_env(e t) (e[8];);
function cp_valid(id) (id>=0 && id<grain_count;);
function pg_limit_span(id start frames) (pg_span_start=start;pg_span_frames=frames;);
function h_m(i) (hm_episodes+i*96;);
function cp_trace(a b c d e f g h i j k l m n o) (1;);
function xc_ok(id) (test_tonal>0 && id==3;);
function xc_g(id) (6200+id*112;);
function pe_at(src frame) (test_cleanup_map ? 3 : (frame<24000 ? 3 : 4););
''' + helpers + '''
voice_base=0;voice_list=2048;event_base=2176;grain_base=4096;source_base=6000;
walker_per_event=13;walker_total=104;record_base=10000;h_record_tags=40000;
h_record_fades=47000;gr_record=48000;gr_norm=57000;pt_periods=gr_norm+walker_total*4;
gt_amounts=pt_periods+64;gt_record=gt_amounts+64;cc_meta=60000;cc_envelope=62500;cc_voices=65000;cc_record=67000;cc_srate=-1;
grain_count=8;voice_list_n=0;clock=0;srate=48000;
@block
// Retain the entry points for native checks, including exact slider selectors.
test_op==1 ? gr_prepare(0,19200,slider65,slider66,slider67);
test_op==2 ? test_result=gr_window(9600,19200,9600,9600,1);
test_op==3 ? (gr_accumulate(0,1);test_result=gr_level(0,slider68););
test_op==4 ? test_result=cp_voice(0,1,0,1,0,0,0,0,19200,0,1,0);
test_op==5 ? cp_record_voice(0,-1,0,0,0,0,0);
test_op==6 ? cp_replay_step(0,100000);
test_op==7 ? (ui_notify(test_id,0.37,1);ui_end_gesture(test_id););
test_op==8 ? test_result=cp_fold(1,-1);
test_op==9 ? test_result=pt_walk_rate(30000,123);
test_op==10 ? test_result=gt_gain(3,test_start,test_frames,test_gain,test_amount);
test_op==11 ? test_result=gt_guard(test_peak,test_amount);
test_op==12 ? cc_grain_begin(test_id);
test_op==13 ? cc_feed(test_power);
test_op==14 ? cc_grain_end(test_id);
test_op==15 ? cc_voice_begin(0,0,test_amount);
test_op==16 ? test_result=cc_apply(0,0,test_age);
test_op==17 ? cc_begin();
@sample
''' + mixer + '\nspl0=out_l;spl1=out_r;clock+=1;\n'
    (out / 'fixture.jsfx').write_text(fixture, encoding='utf-8')
    obj = out / ('fixture.obj' if os.name == 'nt' else 'fixture.o')
    exe = out / ('native.exe' if os.name == 'nt' else 'native')
    command = [sys.executable, str(root / 'dsp_jsfx_aot.py'), str(out / 'fixture.jsfx'),
               '--native-gfx-legacy', '--out-ll', str(out / 'fixture.ll'), '--out-h', str(out / 'JSFXDSP.h'),
               '--out-obj', str(obj), '--meta', str(out / 'meta.json'), '--opt', '2']
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        command += ['--target', 'x86_64-pc-windows-msvc']
    subprocess.run(command, check=True, cwd=root)
    (out / 'native.cpp').write_text(NATIVE, encoding='utf-8')
    command = ['clang++', '-std=c++20', '-O1', '-UNDEBUG', '-I'+str(out),
               '-I'+str(root / 'src'), '-I'+str(root / 'libs/signalsmith-stretch/include'),
               '-I'+str(root / 'libs/signalsmith-linear/include'), str(out / 'native.cpp'),
               str(obj), '-o', str(exe)]
    subprocess.run(command, check=True, cwd=root)
    subprocess.run([str(exe)], check=True, timeout=45, cwd=root)
    print('PASS Corpus native grain rendering, replay and parameter checks', flush=True)


NATIVE = r'''#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include "DspJsfxSpectralPitch.h"
#include <memory>
#include <atomic>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
// Avoid Windows assertion dialogs in automated checks.
#undef assert
#define assert(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
static std::atomic<long> spectralAllocations{0};
void* operator new(std::size_t n){++spectralAllocations;if(void* p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
static int lastChanged=-1,lastAutomated=-1,gestureEnds=0;
static double minRead=1e30,maxRead=-1e30;
#define jsfx_fn_gr_prepare jsfx_fn___fn__block__gr_prepare
#define jsfx_fn_gr_window jsfx_fn___fn__block__gr_window
#define jsfx_fn_gr_accumulate jsfx_fn___fn__block__gr_accumulate
#define jsfx_fn_gr_level jsfx_fn___fn__block__gr_level
#define jsfx_fn_cp_voice jsfx_fn___fn__block__cp_voice
#define jsfx_fn_cp_record_voice jsfx_fn___fn__block__cp_record_voice
#define jsfx_fn_cp_replay_step jsfx_fn___fn__block__cp_replay_step
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){assert(n<=s->memN);}
extern "C" int jsfx_sliderchange(DSPJSFX_State*,double,int index){lastChanged=index;return 1;}
extern "C" int jsfx_slider_automate(DSPJSFX_State*,double,int index,double end){lastAutomated=index;gestureEnds+=end>0;return 1;}
extern "C" double jsfx_fn_gr_window(DSPJSFX_State*,double,double,double,double,double);
extern "C" double jsfx_fn_gr_level(DSPJSFX_State*,double,double);
extern "C" double jsfx_fn_gr_accumulate(DSPJSFX_State*,double,double);
extern "C" double jsfx_fn_gr_prepare(DSPJSFX_State*,double,double,double,double,double);
extern "C" double jsfx_fn_cp_voice(DSPJSFX_State*,double,double,double,double,double,double,double,double,double,double,double,double);
extern "C" double jsfx_fn_cp_record_voice(DSPJSFX_State*,double,double,double,double,double,double,double);
extern "C" double jsfx_fn_cp_replay_step(DSPJSFX_State*,double,double);
// This host uses constant, phase-varying and decorrelated test signals. Signature
// matches the production builtin; it writes exactly one stereo pair per read.
extern "C" double jsfx_sample_read2_interp(DSPJSFX_State*,double,double id,double frame,double* l,double* r){
  minRead=std::min(minRead,frame);maxRead=std::max(maxRead,frame);
  if(id==3){*l=std::sin(frame*2*3.14159265358979323846*440/48000);*r=*l*.7;return 1;}
  *l=id==0 ? 1 : id==1 ? std::sin(frame*.071) : std::sin(frame*.113+1.2);
  *r=id==0 ? .7 : id==1 ? std::cos(frame*.053) : std::cos(frame*.137+.4);return 1;
}
static std::unique_ptr<za::jsfx::SpectralSamplePitch> spectral;
extern "C" double jsfx_sample_pitch_read2(DSPJSFX_State* s,double pool,double id,double start,double frames,double pitch,double age,double voice,double character,double* l,double* r){
  auto read=[&](double at,double& a,double& b){jsfx_sample_read2_interp(s,pool,id,at,&a,&b);};
  double a=0,b=0;bool ok=spectral->read(int(voice),1,id,start,frames,1,s->srate,pitch,character,int(age),read,a,b);
  *l=a;*r=b;return ok?1:0;
}
static int idx(const char* name){for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return v.index;assert(false);return -1;}
static void set(DSPJSFX_State& s,const char* name,double x){s.vars[idx(name)]=x;}
static double get(DSPJSFX_State& s,const char* name){return double(s.vars[idx(name)]);}
static bool near(double a,double b,double tol=1e-10){return std::abs(a-b)<=tol;}
int main(){
  spectral=std::make_unique<za::jsfx::SpectralSamplePitch>();spectral->prepare();
  DSPJSFX_State s{};za::jsfx::StateVariables vars;vars.bind(s,DSPJSFX_VARS_COUNT);
  std::vector<DSPJSFX_Cell> memory(100000);s.mem=memory.data();s.memN=memory.size();s.srate=48000;
  jsfx_init(&s);
  auto reset=[&](){for(auto& x:memory)x=0;jsfx_init(&s);set(s,"output_faults",0);};
  auto prepare=[&](int shape,double softness,double balance){memory[9]=30;memory[10]=45;jsfx_fn_gr_prepare(&s,0,400,shape,softness,balance);};
  auto window=[&](double t){return jsfx_fn_gr_window(&s,t,400,double(memory[9]),double(memory[10]),double(memory[21]));};
  // Original is exactly the former smoothstep trapezoid, even with nonzero
  // stored softness/balance. Softness zero also preserves that reference.
  prepare(0,1,1);assert(double(memory[9])==30 && double(memory[10])==45);
  assert(window(0)==0 && window(400)==0 && window(200)==1);
  assert(near(window(7.5),.15625) && near(window(15),.5) && near(window(22.5),.84375));
  prepare(1,0,.5);assert(near(window(7.5),.15625));
  prepare(1,1,.5);assert(near(window(100),.5));assert(near(window(200),1));
  prepare(2,1,.5);assert(double(memory[9])>double(memory[10]));assert(window(100)<.5);
  for(int shape=0;shape<3;++shape)for(double soft:{0.,.2,.65,1.})for(double bal:{0.,.5,1.}){
    prepare(shape,soft,bal);assert(double(memory[9])+double(memory[10])<=400.000001);
    for(int t=-1;t<=401;++t){double x=window(t);assert(std::isfinite(x) && x>=0 && x<=1.000000001);}
  }
  // Single-grain edges stay intact, full overlaps are reduced, and unrelated
  // walker groups remain independent. Compensation never exceeds +3.01 dB.
  reset();jsfx_fn_gr_accumulate(&s,0,.2);
  assert(near(jsfx_fn_gr_level(&s,0,1),1));
  set(s,"clock",1);jsfx_fn_gr_accumulate(&s,0,1);jsfx_fn_gr_accumulate(&s,0,1);
  assert(near(jsfx_fn_gr_level(&s,0,1),1/std::sqrt(2.)));
  assert(near(jsfx_fn_gr_level(&s,0,2),.5));assert(jsfx_fn_gr_level(&s,0,0)==1);
  jsfx_fn_gr_accumulate(&s,13,.3);assert(near(jsfx_fn_gr_level(&s,13,1),1));
  double previous=1;
  for(int i=0;i<=1000;++i){set(s,"clock",i+2);double w=i/1000.;jsfx_fn_gr_accumulate(&s,0,.5);jsfx_fn_gr_accumulate(&s,0,w);
    double x=jsfx_fn_gr_level(&s,0,1);assert(x>0 && x<=std::sqrt(2.)+1e-12);assert(std::abs(x-previous)<.01);previous=x;}
  auto voice=[&](int slot,int group,double gain,double eventGain,int mode,double curve,double a,double b){
    int v=slot*32;memory[v]=1;memory[v+1]=0;memory[v+2]=0;memory[v+3]=400;
    memory[v+4]=0;memory[v+5]=0;memory[v+6]=400;memory[v+7]=1;memory[v+8]=gain;
    memory[v+9]=a;memory[v+10]=b;memory[v+14]=400;memory[v+17]=7;
    memory[v+21]=curve;memory[v+22]=mode;memory[v+24]=1;memory[v+25]=1;
    memory[v+26]=group/13;memory[v+31]=group;memory[2048+slot]=slot;
    memory[2176+(group/13)*64]=1;memory[2176+(group/13)*64+1]=7;memory[2176+(group/13)*64+8]=eventGain;
  };
  // Exercise the actual two-pass Corpus mixer: amplitude and energy laws,
  // separate notes, and event release remaining multiplicative.
  reset();voice(0,0,1,.25,2,0,30,45);voice(1,0,1,.25,2,0,30,45);
  set(s,"voice_list_n",2);set(s,"clock",200);jsfx_sample(&s);assert(near(double(s.spl[0]),.25));
  reset();voice(0,0,1,.25,1,0,30,45);voice(1,0,1,.25,1,0,30,45);
  set(s,"voice_list_n",2);set(s,"clock",200);jsfx_sample(&s);assert(near(double(s.spl[0]),std::sqrt(2.)*.25));
  reset();voice(0,0,1,.25,2,0,30,45);voice(1,13,.5,.5,2,0,30,45);
  set(s,"voice_list_n",2);set(s,"clock",200);jsfx_sample(&s);assert(near(double(s.spl[0]),.5));
  reset();voice(0,0,1,.25,2,0,30,45);voice(1,0,1,.25,2,0,30,45);
  memory[29]=190;memory[30]=210;memory[61]=190;memory[62]=210;
  set(s,"voice_list_n",2);set(s,"clock",200);jsfx_sample(&s);assert(near(double(s.spl[0]),.125));
  // Original + Off: reference samples, stereo ratio, and unchanged event gain.
  for(int t=0;t<400;++t){reset();voice(0,0,.8,.6,0,0,30,45);set(s,"voice_list_n",1);set(s,"clock",t);jsfx_sample(&s);
    double u=std::clamp(std::min({1.,t/30.,(400.-t)/45.}),0.,1.);double expected=u*u*(3-2*u)*.8*.6;
    assert(near(double(s.spl[0]),expected));assert(near(double(s.spl[1]),expected*.7));}
  // Real cp_voice/record/replay functions: replace live controls after recording
  // and verify REPEAT restores windows, overlap mode and relative child group.
  reset();memory[4096]=0;memory[4097]=0;memory[4100]=48000;memory[6002]=48000;memory[6003]=48000;
  memory[2176]=1;memory[2177]=42;memory[2178]=60;memory[2179]=0;memory[2184]=1;
  memory[2186]=100000;memory[2190]=100000;memory[2191]=0;
  s.sliders[64]=2;s.sliders[65]=.8;s.sliders[66]=.7;s.sliders[67]=2;
  int v=int(jsfx_fn_cp_voice(&s,0,1,0,1,0,0,0,0,19200,0,42,3));assert(v>=0);
  double a=memory[v*32+9],b=memory[v*32+10],curve=memory[v*32+21];
  set(s,"record_event",42);set(s,"record_start",0);set(s,"record_n",0);set(s,"record_done",0);set(s,"replaying",0);
  jsfx_fn_cp_record_voice(&s,v,-1,0,0,0,0,0);assert(get(s,"record_n")==1);
  s.sliders[64]=0;s.sliders[65]=0;s.sliders[66]=0;s.sliders[67]=0;
  memory[v*32]=0;set(s,"voice_list_n",0);set(s,"replay_cursor",0);set(s,"steps_this_block",0);set(s,"replaying",1);
  jsfx_fn_cp_replay_step(&s,0,1);assert(get(s,"voice_list_n")==1);
  int rv=int(memory[2048])*32;assert(near(double(memory[rv+9]),a) && near(double(memory[rv+10]),b));
  assert(near(double(memory[rv+21]),curve) && double(memory[rv+22])==2 && double(memory[rv+31])==3);
  s.sliders[64]=2;s.sliders[65]=1;s.sliders[66]=1;s.sliders[67]=2;
  int preview=int(jsfx_fn_cp_voice(&s,0,4,0,1,0,-1,0,0,19200,0,-1,-1))*32;
  assert(preview>=0 && double(memory[preview+21])==0 && double(memory[preview+22])==0 && double(memory[preview+31])==-1);
  assert(double(memory[preview+9])<=96 && double(memory[preview+10])<=192);
  // Mono folds source stereo before panning, preserves correlated levels,
  // cancels antiphase input and glides monotonically when switched live.
  reset();s.sliders[68]=1;set(s,"source_mix",1);voice(0,0,1,1,0,0,0,0);
  set(s,"voice_list_n",1);set(s,"clock",100);jsfx_sample(&s);
  assert(near(double(s.spl[0]),.85) && near(double(s.spl[1]),.85));
  set(s,"test_op",8);jsfx_block(&s);assert(get(s,"test_result")==0);
  reset();set(s,"source_mix",0);set(s,"source_mix_a",.2);voice(0,0,1,1,0,0,0,0);
  set(s,"voice_list_n",1);set(s,"clock",100);double previousDifference=.3;
  for(int i=0;i<150;++i){jsfx_sample(&s);double l=s.spl[0],r=s.spl[1];
    assert(std::isfinite(l) && std::isfinite(r));assert(l>=.85-1e-12 && l<=1 && r>=.7 && r<=.85+1e-12);
    assert(l-r<=previousDifference+1e-12);previousDifference=l-r;}
  assert(get(s,"source_mix")==1 && near(double(s.spl[0]),double(s.spl[1])));
  for(int id=65;id<=80;++id){set(s,"test_id",id);set(s,"test_op",7);int ends=gestureEnds;jsfx_block(&s);
    assert(lastChanged==id-1 && lastAutomated==id-1 && gestureEnds==ends+1);assert(near(double(s.sliders[id-1]),.37));}
  std::cout<<"PASS windows; overlap bounds/continuity; stereo rendering; note isolation; event release; Original reference; replay; mono fold/glide; high slider automation\n";
  // Actual per-voice pitch engine and mixer, with phase-aligned tonal evidence.
  // All PCM reads must remain inside the selected source span at every shift.
  auto pitchVoice=[&](int mode,int pitch,int frames=48000,int sourceRate=48000){
    reset();set(s,"replaying",0);s.sliders[68]=0;s.sliders[73]=0;set(s,"source_mix",0);set(s,"source_mix_a",0);s.sliders[72]=mode;
    s.sliders[64]=0;s.sliders[65]=0;s.sliders[66]=.5;s.sliders[67]=0;
    set(s,"test_tonal",1);memory[6200+3*112+28]=1;memory[6200+3*112+29]=440;memory[6200+3*112+30]=1;
    int g=4096+3*64;memory[g]=3;memory[g+1]=0;memory[g+4]=sourceRate;memory[g+5]=2;
    memory[6002]=200000;memory[6003]=sourceRate;memory[2176]=1;memory[2177]=42;memory[2184]=1;memory[2186]=1e9;memory[2190]=1e9;
    return int(jsfx_fn_cp_voice(&s,3,1,0,1,pitch,0,0,12000,frames,0,42,0));
  };
  std::vector<double> zeroTape;
  for(int mode=0;mode<=1;++mode)for(int pitch:{-24,0,24}){
    int slot=pitchVoice(mode,pitch);assert(slot>=0);int v=slot*32;
    int expected=mode==1 ? 48000 : int(std::ceil(48000/std::pow(2.,pitch/12.)));
    assert(double(memory[v+3])==expected);assert((double(memory[v+23])<0)==(mode==1 && pitch!=0));
    std::vector<double> rendered;minRead=1e30;maxRead=-1e30;double previous=0,maxDelta=0;int crossing=0;
    for(int t=0;t<expected;++t){jsfx_sample(&s);double l=s.spl[0],r=s.spl[1];assert(std::isfinite(l) && std::isfinite(r));assert(near(r,l*.7,.003));
      if(t>=9600 && t<38400 && expected>=48000){crossing+=previous<=0 && l>0;maxDelta=std::max(maxDelta,std::abs(l-previous));}
      previous=l;rendered.push_back(l);}
    assert(minRead>=12000 && maxRead<=59999);
    if(mode==1){double frequency=crossing/.6;double wanted=440*std::pow(2.,pitch/12.);std::cout<<"pitch="<<pitch<<" frequency="<<frequency<<" expected="<<wanted<<" half="<<double(memory[v+23])<<" period="<<double(memory[57416+slot])<<" maxDelta="<<maxDelta<<std::endl;assert(std::abs(frequency-wanted)<4);assert(maxDelta<.28);}
    if(pitch==0){if(mode==0)zeroTape=rendered;else assert(rendered==zeroTape);}
  }
  for(int sourceRate:{22050,44100,48000,96000}){int v=pitchVoice(1,-24,sourceRate,sourceRate)*32;assert(double(memory[v+3])==48000);}
  int tiny=pitchVoice(1,24,2)*32;assert(double(memory[tiny+23])==0 && double(memory[tiny+3])==1);
  pitchVoice(1,0);memory[0]=0;set(s,"voice_list_n",0);int nominalZero=int(jsfx_fn_cp_voice(&s,3,1,0,1,.000001,0,0,12000,48000,0,42,0))*32;assert(double(memory[nominalZero+23])==0);
  // Coherent REPEAT retains its duration, head window and tonal alignment even
  // after switching back to Tape. Recorded natural duration is independent of
  // the event deadline and the existing hierarchy stop metadata.
  int pv=pitchVoice(1,24)*32;double duration=memory[pv+3],half=memory[pv+23],period=memory[57416+pv/32];
  set(s,"record_event",42);set(s,"record_start",0);set(s,"record_n",0);set(s,"record_done",0);set(s,"replaying",0);
  jsfx_fn_cp_record_voice(&s,pv/32,-1,0,0,0,0,0);assert(get(s,"record_n")==1);
  memory[pv]=0;set(s,"voice_list_n",0);set(s,"replay_cursor",0);set(s,"steps_this_block",0);set(s,"replaying",1);s.sliders[72]=0;
  jsfx_fn_cp_replay_step(&s,0,1);int replay=int(memory[2048])*32;
  assert(double(memory[replay+3])==duration && double(memory[replay+14])==duration && double(memory[replay+23])==half && double(memory[57416+replay/32])==period);
  set(s,"test_op",9);memory[30047]=.25;jsfx_block(&s);assert(get(s,"test_result")==.25);memory[30047]=0;jsfx_block(&s);assert(get(s,"test_result")==123);
  std::cout<<"PASS Tape/Coherent -24/0/+24: duration, actual transposition, spectral tone continuity, exact zero bypass, stereo linking, strict source bounds, sample rates, tiny spans, replay mode isolation and source-motion tracking\n";
  // A 10 kHz input would fold to 8 kHz under an unfiltered 4x read.
  // Texture mapping should move it near13.3kHz without that aliased tone.
  double a8=0,b8=0,a13=0,b13=0,totalStereoError=0,carrierEnergy=0,minAt=1e30,maxAt=-1e30;
  const long allocationStart=spectralAllocations.load();
  auto highRead=[&](double at,double& l,double& r){minAt=std::min(minAt,at);maxAt=std::max(maxAt,at);l=.15*std::sin(at*2*3.141592653589793*10000/48000);r=l*.7;};
  for(int i=0;i<48000;++i){double l=0,r=0;assert(spectral->read(1,1,10,0,48000,1,48000,24,0,i,highRead,l,r));
    if(i>=9600 && i<38400){double f=2*3.141592653589793*i/48000;
      carrierEnergy+=l*l;a8+=l*std::cos(8000*f);b8+=l*std::sin(8000*f);a13+=l*std::cos(13300*f);b13+=l*std::sin(13300*f);}
    totalStereoError=std::max(totalStereoError,std::abs(r-.7*l));}
  assert(spectralAllocations.load()==allocationStart);
  double alias=(2*std::hypot(a8,b8)/28800)/std::max(1e-12,std::sqrt(2*carrierEnergy/28800));
  std::cout<<"spectral alias ratio="<<alias<<" stereo error="<<totalStereoError<<std::endl;
  assert(alias<.01 && carrierEnergy>100 && totalStereoError<.003 && minAt>=0 && maxAt<=47999);
  double badL=1,badR=1;assert(!spectral->read(64,1,10,0,48000,1,48000,24,0,0,highRead,badL,badR) && badL==0 && badR==0);
  assert(!spectral->read(1,1,10,0,48000,1,48000,1e30,0,0,highRead,badL,badR));
  std::cout<<"PASS texture mapping, high-frequency alias rejection, strict lookahead bounds and zero playback allocations\n";
  // Taming uses only cached metadata at voice creation. No quiet-grain makeup;
  // zero is exact bypass, including the linked soft peak guard.
  auto tame=[&](double rms,double peak,double amount,double baseGain=1){
    memory[4096+3*64+6]=rms;memory[4096+3*64+7]=peak;set(s,"pe_ready",0);
    set(s,"test_start",12000);set(s,"test_frames",48000);set(s,"test_gain",baseGain);set(s,"test_amount",amount);set(s,"test_op",10);jsfx_block(&s);return get(s,"test_result");};
  auto guard=[&](double peak,double amount){set(s,"test_peak",peak);set(s,"test_amount",amount);set(s,"test_op",11);jsfx_block(&s);return get(s,"test_result");};
  assert(tame(.01,.1,1)==1 && tame(10,100,0)==1 && tame(0,0,1)==1);
  double full=tame(.8,8,1),partial=tame(.8,8,.5);assert(full<.12 && full>0 && partial>full && partial<1);
  double last=0;for(int i=0;i<=20000;++i){double peak=i/1000.,output=peak*guard(peak,1);assert(output>=last-1e-12 && output<=1);assert(guard(peak,0)==1);last=output;}
  assert(guard(.7,1)==1 && std::abs(.70000001*guard(.70000001,1)-.70000001)<1e-12);
  // An expanded excerpt includes the louder neighbour's cached measurements.
  tame(.1,.2,1);int g3=4096+3*64,g4=4096+4*64;memory[g3+2]=12000;memory[g3+3]=12000;
  memory[g4+1]=0;memory[g4+2]=24000;memory[g4+3]=36000;memory[g4+6]=.8;memory[g4+7]=8;
  set(s,"pe_ready",1);set(s,"source_count",1);set(s,"test_op",10);jsfx_block(&s);assert(get(s,"test_result")<.12);
  set(s,"test_frames",8000);jsfx_block(&s);assert(get(s,"test_result")==1);
  // Render a loud and a quiet grain together: the loud voice is tamed before
  // mixing, while the unrelated quiet voice contributes its full unchanged gain.
  reset();s.sliders[68]=0;set(s,"source_mix",0);set(s,"source_mix_a",0);voice(0,0,10,1,0,0,0,0);
  memory[57480]=1;set(s,"voice_list_n",1);set(s,"clock",200);jsfx_sample(&s);double loud=s.spl[0];assert(loud<1 && loud>.7);
  voice(1,13,.1,1,0,0,0,0);set(s,"voice_list_n",2);set(s,"clock",200);jsfx_sample(&s);assert(near(double(s.spl[0]),loud+.1) && near(double(s.spl[1]),(loud+.1)*.7));
  // Replay retains the fixed metadata gain and guard amount, with no second
  // gain ride even when the live amount is changed.
  int gv=pitchVoice(0,0)*32;memory[gv]=0;set(s,"voice_list_n",0);s.sliders[73]=1;memory[g3+6]=.8;memory[g3+7]=8;
  gv=int(jsfx_fn_cp_voice(&s,3,1,0,1,0,0,0,12000,48000,0,42,0))*32;double fixedGain=memory[gv+8];assert(fixedGain<.12 && memory[57480+gv/32]==1);
  set(s,"record_event",42);set(s,"record_start",0);set(s,"record_n",0);set(s,"record_done",0);set(s,"replaying",0);
  jsfx_fn_cp_record_voice(&s,gv/32,-1,0,0,0,0,0);memory[gv]=0;set(s,"voice_list_n",0);set(s,"replay_cursor",0);set(s,"steps_this_block",0);set(s,"replaying",1);s.sliders[73]=0;
  jsfx_fn_cp_replay_step(&s,0,1);int gr=int(memory[2048])*32;assert(near(double(memory[gr+8]),fixedGain) && memory[57480+gr/32]==1);
  std::cout<<"PASS per-grain taming: cached RMS/peak/expanded span, attenuation only, exact bypass, soft-knee continuity, bounded stereo-linked peak guard, independent quiet voices and replay without double gain reduction\n";
  // Cache derives from the actual complete PCM energy stream, retaining tail
  // resolution even when a Coherent passage is long. It never reads PCM here.
  reset();set(s,"grain_count",4);set(s,"total_seconds",1);set(s,"test_op",17);jsfx_block(&s);
  int cg=4096+3*64;memory[cg+2]=0;memory[cg+3]=48000;memory[cg+4]=48000;
  set(s,"test_id",3);set(s,"test_op",12);jsfx_block(&s);
  for(int i=0;i<48000;++i){double level=i<4800 || i>=38400 ? .2 : .001;set(s,"test_power",level*level);set(s,"test_op",13);jsfx_block(&s);}
  set(s,"test_op",14);jsfx_block(&s);assert(memory[60012+1]==100 && memory[60012+3]==480 && near(double(memory[60012+2]),.2));assert(get(s,"cc_samples")==48000 && get(s,"cc_used")==100);
  set(s,"pe_ready",1);set(s,"test_cleanup_map",1);memory[1]=3;memory[5]=0;memory[6]=48000;memory[26]=0;
  auto cleanup=[&](double amount,int kind,int duration){memory[3]=duration;memory[27]=kind;set(s,"test_amount",amount);set(s,"test_op",15);jsfx_block(&s);};
  auto gain=[&](int age){set(s,"test_age",age);set(s,"test_op",16);jsfx_block(&s);return get(s,"test_result");};
  for(double rate:{32000.,44100.,48000.,96000.,192000.})for(int pitch:{-24,0,24}){
   s.srate=rate;int duration=int(rate/std::pow(2.,pitch/12.));cleanup(1,3,duration);double low=1,delta=0,last=1;
   for(int t=0;t<duration;++t){double q=gain(t);assert(std::isfinite(q)&&q>=std::pow(10.,-24./20.)-1e-12&&q<=1);delta=std::max(delta,std::abs(q-last));last=q;if(t>duration*.4&&t<duration*.75)low=std::min(low,q);if(t<duration*.05)assert(q==1);}
   assert(low<.3 && last>.98 && delta<.025);
  }
  s.srate=48000;cleanup(0,3,48000);for(int t=0;t<48000;t+=31)assert(gain(t)==1);cleanup(1,0,48000);assert(memory[65000]==0);for(int t=0;t<48000;t+=31)assert(gain(t)==1);
  memory[26]=-1;cleanup(1,3,48000);assert(memory[65000]==0);memory[26]=0;set(s,"pe_ready",0);cleanup(1,3,48000);assert(memory[65000]==0);set(s,"pe_ready",1);
  // Quiet recordings retain the exact relative behavior; absolute recording
  // level is not the threshold. Constant low-level content stays fully open.
  cleanup(.7,3,48000);std::vector<double> relative;for(int t=0;t<48000;++t)relative.push_back(gain(t));
  memory[60014]*=.0001;for(int i=0;i<100;++i)memory[62500+i]*=.0001;
  cleanup(.7,3,48000);for(int t=0;t<48000;++t)assert(near(gain(t),relative[t],1e-10));
  memory[60014]=.000001;for(int i=0;i<100;++i)memory[62500+i]=.000001;cleanup(1,3,48000);for(int t=0;t<48000;++t)assert(gain(t)==1);
  // Missing, single-bin or silent evidence is safe; broad cells back off.
  memory[60013]=1;cleanup(1,3,48000);for(int t=0;t<1000;++t)assert(gain(t)==1);memory[60013]=100;memory[60014]=1e-12;cleanup(1,3,48000);for(int t=0;t<1000;++t)assert(gain(t)==1);
  set(s,"test_id",249);set(s,"test_op",7);int ends=gestureEnds;jsfx_block(&s);assert(lastChanged==248&&lastAutomated==248&&gestureEnds==ends+1);
  // Large-bank allocation reserves at least one cell for every later grain.
  set(s,"grain_count",512);set(s,"total_seconds",51200000);set(s,"test_op",17);jsfx_block(&s);
  for(int id=0;id<512;++id){int q=4096+id*64;memory[q+3]=48000.*100000;memory[q+4]=48000;set(s,"test_id",id);set(s,"test_op",12);jsfx_block(&s);int m=60000+id*4;assert(memory[m]>=0&&memory[m+1]>=1&&memory[m]+memory[m+1]<=524288);}
  assert(get(s,"cc_used")<=524288&&get(s,"cc_scale")>1);
  std::cout<<"PASS child cleanup: full RMS bins, automatic relative reference, five rates/three source-motion speeds, bounded smooth attenuation, opening protection, exact Off/parent/audition bypass, quiet-source invariance, safe fallback, automation and fixed cache capacity\n";

}
'''

if __name__ == '__main__':
    main()
