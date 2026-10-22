"""Compile actual Corpus packet/envelope functions in the native Legacy runtime."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path);ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path)
    args=ap.parse_args();root=args.repo or Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/tasks'))
    from test_corpus_preparation import function
    source=(args.source or root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text(encoding='utf-8')
    out=args.out or root/'build/tasks/corpus-duration';out.mkdir(parents=True,exist_ok=True)
    names=['cp_clamp','cp_01','cp_bend','cp_curve','cp_env_curves','cp_env_pre','cp_env','cp_env_save','cp_release','cp_rng','cp_g','cp_e','cp_w','cp_v','cp_event_find','cp_hold_count','cp_phase','cp_start','cp_stop_event','cp_stop_all','cp_packet_execute','cp_packet_cc','cp_midi_cc','cp_midi_noteoff']
    for name in ('cp_packet_envelope','cp_packet_release'):
        if 'function '+name+'(' in source:names.append(name)
    headers='\n'.join(line for line in source.splitlines() if line.startswith('slider'))
    fixture=headers+'''\noptions:maxmem=32768
@init
function cp_child_count() (0;);
function cp_child_mode() (0;);
function amp_evt(ei) (amp_event_base+ei*40;);
function amp_custom_eval(e t ei) (0;);
function cp_valid(id) (0;);
function cp_feat(id d) (0.5;);
'''+ '\n'.join(function(source,n) for n in names)+'''
xc_params=256;event_base=2176;walker_base=3000;walker_per_event=13;walker_total=104;
grain_base=10000;voice_base=11000;voice_list=13100;amp_curve_base=14000;amp_event_base=14100;env_old_base=15000;packet_base=20000;pedal_base=21000;
ready=1;pe_ready=1;pg_ready=1;playable_grains=8;rng_state=12345;
@block
test_op==1 ? cp_midi_cc(test_channel,test_cc,test_value,test_when);
test_op==2 ? test_result=cp_env(cp_e(test_ei),test_time);
test_op==3 ? test_ei=cp_start(1,60,0,test_when,0,0,127,127,0);
test_op==4 ? cp_midi_noteoff(60,test_channel,test_when);
'''
    (out/'fixture.jsfx').write_text(fixture,encoding='utf-8')
    obj=out/('fixture.obj' if os.name=='nt' else 'fixture.o');exe=out/('native.exe' if os.name=='nt' else 'native')
    cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(out/'fixture.jsfx'),'--native-gfx-legacy','--out-ll',str(out/'fixture.ll'),'--out-h',str(out/'JSFXDSP.h'),'--out-obj',str(obj),'--opt','2']
    if os.name=='nt':cmd+=['--target','x86_64-pc-windows-msvc']
    subprocess.run(cmd,check=True,cwd=root)
    (out/'native.cpp').write_text(NATIVE,encoding='utf-8')
    subprocess.run(['clang++','-std=c++20','-O1','-UNDEBUG','-I'+str(out),'-I'+str(root/'src'),str(out/'native.cpp'),str(obj),'-o',str(exe)],check=True,cwd=root)
    subprocess.run([str(exe)],check=True,timeout=30,cwd=root)

NATIVE=r'''#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){CHECK(false);}
extern "C" double jsfx_memcpy(DSPJSFX_State* s,double d,double a,double n){std::vector<double> copy;for(int i=0;i<int(n);++i)copy.push_back(double(s->mem[int(a)+i]));for(int i=0;i<int(n);++i)s->mem[int(d)+i]=copy[i];return d;}
static int index(const char* name){for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return v.index;CHECK(false);return -1;}
struct Fixture {
 DSPJSFX_State s{};za::jsfx::StateVariables vars;std::vector<DSPJSFX_Cell> heap;
 Fixture(double rate=48000):heap(32768){vars.bind(s,DSPJSFX_VARS_COUNT);s.mem=heap.data();s.memN=heap.size();s.srate=rate;
  s.sliders[9]=0;s.sliders[24]=800;s.sliders[25]=5;s.sliders[26]=100;s.sliders[27]=.75;s.sliders[28]=180;s.sliders[46]=2;s.sliders[48]=1;
  s.sliders[74]=1;jsfx_init(&s);}
 void set(const char* n,double x){s.vars[index(n)]=x;} double get(const char* n){return double(s.vars[index(n)]);}
 double e(int n,int ei=0){return double(heap[2176+ei*64+n]);}
 void cc(int c,int v,double when,int ch=0){set("test_op",1);set("test_channel",ch);set("test_cc",c);set("test_value",v);set("test_when",when);jsfx_block(&s);}
 double env(double t,int ei=0){set("test_op",2);set("test_ei",ei);set("test_time",t);jsfx_block(&s);return get("test_result");}
 void packet(int id,double duration,int boundary=1,int a=127,int r=127,int action=0,int target=0,double when=0,int ch=0,bool fine=true,bool badChecksum=false){
  // Sampler MIDI Hub v1.0.1 H.data / H.packet_events wire format.
  int ticks=int(std::floor(duration*(fine?64000:1000)+.5));
  std::array<int,20> p{1,action,id/128,id%128,target/128,target%128,60,action==7?0:100,boundary,1,0,64,64,fine?1:0,ticks/16384,(ticks/128)%128,ticks%128,a,r,0};
  int sum=0;cc(118,83,when,ch);for(int i=0;i<20;++i){sum=(sum+(i+1)*p[i])%128;cc(119,p[i],when,ch);}cc(119,(sum+int(badChecksum))%128,when,ch);cc(118,84,when,ch);
 }
};
static void near(double a,double b,double tolerance=1.01){if(std::abs(a-b)>tolerance){std::cerr<<"expected "<<b<<" got "<<a<<std::endl;CHECK(false);}}
int main(){
 // The default Hub 'Instrument' envelope should keep its nominal proportions
 // when the incoming window is five times as long, including a 900ms release.
 Fixture longNote;longNote.packet(1,4);near(longNote.e(6),1200);near(longNote.e(7),25200);near(longNote.e(9),148800);near(longNote.e(10),192000);near(longNote.e(19),43200);
 CHECK(longNote.env(3.55*48000)>.3 && longNote.env(3.55*48000)<.5);CHECK(longNote.env(4*48000)==0);
 std::cout<<"PASS default Hub Instrument: 4s duration scales 5/100/180ms ADSR to 25/500/900ms\n";
 for(double rate:{32000.,44100.,48000.,96000.})for(double length:{.02,.1,.8,4.,20.})for(int mode:{0,1,2}){
  Fixture f(rate);f.s.sliders[69]=.45;f.s.sliders[70]=-.3;f.s.sliders[71]=-.6;f.packet(1,length,mode);
  double total=std::floor(length*rate+.5);near(f.e(6),std::floor(total*5/800));near(f.e(7),std::floor(total*5/800)+std::floor(total*100/800));near(f.e(19),std::max(1.,total*180/800));
  near(f.e(9),mode==1?total-f.e(19):total);near(f.e(10),mode==1?total:total+f.e(19));
  double previous=f.env(f.e(9));CHECK(previous>.74 && previous<=.75);for(int i=1;i<=100;++i){double v=f.env(f.e(9)+(f.e(10)-f.e(9))*i/100);CHECK(std::isfinite(v)&&v<=previous+1e-12&&v>=0);previous=v;}CHECK(previous==0);
  double tail=f.e(10);f.packet(1,0,mode,127,127,7,0,total);near(f.e(10),tail);CHECK(f.e(14)>tail);
 }
 std::cout<<"PASS relative timing at 4 rates, 5 durations, Fit/Free/Tail; curved releases and on-time End\n";
 for(int relative:{0,1})for(int mode:{0,1,2}){
  Fixture f;f.s.sliders[74]=relative;f.s.sliders[26]=5000;f.packet(1,.1,mode,10,80);
  near(f.e(6),480);near(f.e(19),3840);near(f.e(9),mode==1?960:4800);near(f.e(10),mode==1?4800:8640);
  CHECK(f.e(7)<=f.e(9));CHECK(f.env(f.e(9)+1920)>.3);
 }
 std::cout<<"PASS explicit A/R percentages override either timing mode; long Decay cannot steal Release\n";
 for(int mode:{0,1,2}){
  Fixture f;f.s.sliders[74]=0;f.packet(1,.1,mode,100,100);near(f.e(6),mode==1?2400:4800);near(f.e(19),mode==1?2400:4800);CHECK(f.e(7)<=f.e(9));
  Fixture zero;zero.packet(1,1./64000,mode,0,0);CHECK(zero.e(10)>=1 && zero.e(19)>=1);for(int t=0;t<5;++t)CHECK(std::isfinite(zero.env(t)));
 }
 std::cout<<"PASS overcommitted percentages fit together; zero stages and sub-sample durations stay finite\n";
 Fixture absolute;absolute.s.sliders[74]=0;absolute.packet(1,4);near(absolute.e(6),240);near(absolute.e(7),5040);near(absolute.e(19),8640);
 Fixture normal;normal.set("test_op",3);jsfx_block(&normal.s);near(normal.e(6),240);near(normal.e(7),5040);near(normal.e(19),8640);near(normal.e(10),38400);
 Fixture ms;ms.packet(1,4,1,127,127,0,0,37,0,false);near(ms.e(10),192037);near(ms.e(19),43200);
 std::cout<<"PASS Absolute retains milliseconds, ordinary notes unchanged, millisecond and fine duration units\n";
 for(int relative:{0,1})for(int mode:{0,1,2}){
  Fixture f;f.s.sliders[74]=relative;f.s.sliders[71]=.5;f.packet(1,.8,mode);
  const double t=mode==1?30000:38400;double before=f.env(t-100),at=f.env(t);
  f.s.sliders[74]=1-relative;f.s.sliders[28]=5000; // Continue keeps birth-time timing references.
  f.packet(2,4,mode,127,127,6,1,t);near(f.e(19),relative?43200:8640);CHECK(f.e(4)==2);CHECK(f.env(t-100)==before);near(f.env(t),at,1e-12);
  CHECK(f.get("packet_accepted")==2 && f.get("packet_rejected")==0);
 }
 for(int mode:{0,1,2}){
  Fixture f;f.packet(1,.8,mode);f.packet(2,.1,mode,127,100,6,1,38400);near(f.e(19),4800);
  Fixture inherited;inherited.s.sliders[74]=0;inherited.s.sliders[28]=5000;inherited.packet(1,6,mode);inherited.packet(2,.1,mode,127,127,6,1,288000);near(inherited.e(19),mode==1?4800:240000);
 }
 std::cout<<"PASS Continue scales inherited relative Release, keeps captured mode, preserves past samples, and allows full Free/Tail release\n";
 Fixture invalid;invalid.packet(1,1,1,127,127,0,0,0,0,true,true);CHECK(invalid.get("packet_rejected")==1 && invalid.e(0)==0);
 Fixture ch;ch.packet(1,1,1,127,127,0,0,0,3);CHECK(ch.e(3)==3);ch.set("test_op",4);ch.set("test_channel",3);ch.set("test_when",100);jsfx_block(&ch.s);CHECK(ch.e(9)>100);
 std::cout<<"PASS actual CC118/119 checksum decoder, channel routing and duration-owned Note Off\n";
}
'''
if __name__=='__main__':main()
