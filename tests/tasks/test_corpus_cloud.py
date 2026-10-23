"""Native checks of Corpus cloud parameters and the actual linked smear DSP."""
import argparse,os,subprocess,sys
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--source',type=Path);args=ap.parse_args()
    root=args.repo or Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/tasks'))
    from test_corpus_preparation import function
    source=(args.source or root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text()
    out=args.out or root/'build/tasks/corpus-cloud';out.mkdir(parents=True,exist_ok=True)
    helpers=['cp_clamp','cp_01','cp_bend','cp_env_curves','cp_env_pre','cp_rng','cp_e','cp_w','cp_v','cp_child_legacy','cp_child_count','cp_child_mode','cp_child_kind','cp_hold_count','cp_start','cp_child','td_stage','td_prepare','td_clear','td_reset_grains','td_random','td_grain','td_process']
    fixture='\n'.join(l for l in source.splitlines() if l.startswith('slider'))+'''\noptions:maxmem=131072
@init
function cp_event_find(note chan ext) (-1;);
function cp_stop_event(ei when fade) (1;);
function cp_valid(id) (0;);
function cp_feat(id d) (0.5;);
function amp_evt(ei) (80000+ei*40;);
'''+ '\n'.join(function(source,n) for n in helpers)+'''
td_buf_l=0;td_buf_r=32768;td_pos_base=65536;td_heads=td_pos_base+8;td_window=td_pos_base+64;
event_base=70000;walker_base=71000;walker_per_event=13;walker_total=104;amp_curve_base=81000;xc_params=82000;
ready=1;pe_ready=1;pg_ready=1;playable_grains=8;rng_state=12345;
@block
test_op==0 ? (td_prepare();td_clear(););
test_op==1 ? (memset(event_base,0,512);test_ei=cp_start(1,60,0,0,0,0,127,127,0););
test_op==2 ? cp_child(1,test_kind,0,200,1,test_count);
test_op==3 ? test_result=cp_child_kind(test_child,test_mode);
@sample
slider45>0 ? (td_process(spl0,spl1,slider45,slider80,test_gate);spl0=td_out_l;spl1=td_out_r;);
'''
    (out/'fixture.jsfx').write_text(fixture);(out/'native.cpp').write_text(NATIVE)
    obj=out/('fixture.obj' if os.name=='nt' else 'fixture.o');exe=out/('native.exe' if os.name=='nt' else 'native')
    cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(out/'fixture.jsfx'),'--native-gfx-legacy','--out-ll',str(out/'fixture.ll'),'--out-h',str(out/'JSFXDSP.h'),'--out-obj',str(obj),'--opt','2']
    if os.name=='nt':cmd+=['--target','x86_64-pc-windows-msvc']
    subprocess.run(cmd,check=True,cwd=root)
    subprocess.run(['clang++','-std=c++20','-O1','-UNDEBUG','-I'+str(out),'-I'+str(root/'src'),str(out/'native.cpp'),str(obj),'-o',str(exe)],check=True,cwd=root)
    subprocess.run([str(exe)],check=True,timeout=60,cwd=root)
NATIVE=r'''#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){CHECK(false);}
extern "C" double jsfx_memcpy(DSPJSFX_State* s,double d,double a,double n){std::vector<double> v;for(int i=0;i<int(n);++i)v.push_back(double(s->mem[int(a)+i]));for(int i=0;i<int(n);++i)s->mem[int(d)+i]=v[i];return d;}
static int idx(const char* n){for(auto& v:DSPJSFX_VARS)if(std::string(v.name)==n)return v.index;CHECK(false);return -1;}
struct Fixture{
 DSPJSFX_State s{};za::jsfx::StateVariables vars;std::vector<DSPJSFX_Cell> heap;
 Fixture(double rate=48000):heap(131072){vars.bind(s,DSPJSFX_VARS_COUNT);s.mem=heap.data();s.memN=heap.size();s.srate=rate;s.sliders[24]=800;s.sliders[25]=5;s.sliders[26]=100;s.sliders[27]=.75;s.sliders[28]=180;s.sliders[36]=.25;s.sliders[37]=.85;s.sliders[52]=.3;s.sliders[77]=.6;s.sliders[78]=.2;s.sliders[79]=1;jsfx_init(&s);set("test_gate",1);}
 void set(const char* n,double x){s.vars[idx(n)]=x;}double get(const char* n){return double(s.vars[idx(n)]);}
 void reset(double amount,int mode=1){s.sliders[44]=amount;s.sliders[79]=mode;set("test_op",0);jsfx_block(&s);}
 std::array<double,2> sample(double l,double r){s.spl[0]=l;s.spl[1]=r;jsfx_sample(&s);return {double(s.spl[0]),double(s.spl[1])};}
};
int main(){
 for(double rate:{32000.,44100.,48000.,96000.,192000.})for(int mode:{0,1})for(double amount:{0.,.15,.5,1.}){
  Fixture f(rate);f.reset(amount,mode);double peak=0,error=0;
  for(int i=0;i<int(rate*.25);++i){double input=.23*std::sin(i*.071)+.12*std::sin(i*.317);auto y=f.sample(input,input);CHECK(std::isfinite(y[0])&&std::isfinite(y[1]));CHECK(y[0]==y[1]);if(amount==0)CHECK(y[0]==input);peak=std::max(peak,std::abs(y[0]));}
  CHECK(peak<1.1);f.reset(amount,mode);
  for(int i=0;i<10000;++i){double x=.35*std::sin(i*.071);auto y=f.sample(x,x*.4);error=std::max(error,std::abs(y[1]-y[0]*.4));}CHECK(error<1e-12);
 }
 std::cout<<"PASS actual smear at five rates / four amounts: true mono exact, proportional stereo linked, finite bounds and exact zero bypass\n";
 std::array<double,4> difference{};int n=0;
 for(double amount:{.15,.5,.75,1.}){Fixture f;f.reset(amount);double error=0,power=0,inputPower=0;uint32_t random=12345;
  for(int i=0;i<48000;++i){random=random*1664525u+1013904223u;double x=(double(random)/4294967296.-.5)*.6;auto y=f.sample(x,x);if(i>8000){error+=(y[0]-x)*(y[0]-x);power+=y[0]*y[0];inputPower+=x*x;}}
  difference[n++]=std::sqrt(error/inputPower);double ratio=std::sqrt(power/inputPower);CHECK(ratio>.45 && ratio<1.3);std::cout<<"GRAIN amount="<<amount<<" difference="<<difference[n-1]<<" rms_ratio="<<ratio<<'\n';}
 CHECK(difference[0]<.02 && difference[1]>difference[0]*8 && difference[3]>.8);
 Fixture impulse;impulse.reset(1);double energy=0,weighted=0,late=0;int nonzero=0;for(int i=0;i<12000;++i){auto y=impulse.sample(i==0?1:0,i==0?1:0);double p=y[0]*y[0];energy+=p;weighted+=i*p;if(i>1440)late+=p;if(std::abs(y[0])>1e-8)++nonzero;}
 std::cout<<"GRAIN impulse centroid_ms="<<weighted/energy/48<<" late_energy="<<late<<" nonzero="<<nonzero<<'\n';CHECK(late>1e-4 && nonzero>=6);
 Fixture stop;stop.reset(1);for(int i=0;i<4800;++i)stop.sample(.2,.2);stop.set("test_gate",0);for(int i=0;i<5000;++i){auto y=stop.sample(0,0);CHECK(y[0]==0 && y[1]==0);}
 Fixture original,replay;original.reset(.8);replay.reset(.8);original.set("test_gate",0);replay.set("test_gate",0);for(int i=0;i<17;++i)original.sample(0,0);for(int i=0;i<144;++i)replay.sample(0,0);original.set("test_gate",1);replay.set("test_gate",1);
 for(int i=0;i<10000;++i){double x=.2*std::sin(i*.19);CHECK(original.sample(x,x)==replay.sample(x,x));}
 std::cout<<"PASS gentle low range, strong grain cloud, diffuse impulse energy, end gating and deterministic restart after different silent gaps\n";
 // Setting a new plan must not silently rewrite the already-triggered event.
 Fixture cloud;cloud.s.sliders[75]=7;cloud.s.sliders[76]=4;cloud.s.sliders[77]=.72;cloud.s.sliders[78]=.67;cloud.s.sliders[37]=.9;cloud.s.sliders[52]=.8;
 cloud.set("test_op",1);jsfx_block(&cloud.s);int e=70000;CHECK(cloud.heap[e+45]==7 && cloud.heap[e+46]==4);CHECK(std::abs(double(cloud.heap[e+47])-.72)<1e-12);CHECK(cloud.heap[e+60]==.9 && cloud.heap[e+61]==.8);
 cloud.s.sliders[75]=0;cloud.s.sliders[76]=0;cloud.s.sliders[77]=0;cloud.s.sliders[78]=0;cloud.s.sliders[37]=0;cloud.s.sliders[52]=0;
 CHECK(cloud.heap[e+45]==7 && cloud.heap[e+46]==4);cloud.set("test_op",2);cloud.set("test_kind",5);cloud.set("test_count",7);jsfx_block(&cloud.s);
 CHECK(cloud.heap[71000+64+43]==.72 && cloud.heap[71000+64+44]==.67);CHECK(std::abs(double(cloud.heap[71000+64+9]))<=.8);
 const int roster[]{3,1,6,2,5,4};for(int i=1;i<=12;++i){cloud.set("test_op",3);cloud.set("test_child",i);cloud.set("test_mode",6);jsfx_block(&cloud.s);CHECK(cloud.get("test_result")==roster[(i-1)%6]);}
 for(int old:{0,1,2,3}){cloud.s.sliders[35]=old;cloud.s.sliders[75]=-1;cloud.set("test_op",1);jsfx_block(&cloud.s);CHECK(cloud.heap[e+45]==(old==0?0:old==3?2:1));cloud.s.sliders[75]=0;cloud.set("test_op",1);jsfx_block(&cloud.s);CHECK(cloud.heap[e+45]==0);}
 std::cout<<"PASS saved-slider cloud plan, event snapshots (count/mode/density/scatter/inheritance/width), Mixed roster and explicit Off over legacy morphology\n";
}
'''
if __name__=='__main__':main()
