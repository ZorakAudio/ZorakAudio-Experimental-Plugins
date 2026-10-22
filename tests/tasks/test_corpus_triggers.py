"""Native checks of Corpus event seeding and MIDI/Generate velocity semantics."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from test_corpus_preparation import function

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path);ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path)
    a=ap.parse_args();root=a.repo or Path(__file__).resolve().parents[2]
    source=(a.source or root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text(encoding='utf-8')
    out=a.out or root/'build/tasks/corpus-triggers';out.mkdir(parents=True,exist_ok=True)
    headers='\n'.join(line for line in source.splitlines() if line.startswith('slider'))
    helpers='\n'.join(function(source,n) for n in ('cp_clamp','cp_01','cp_bend','cp_curve','cp_env_curves','cp_env_pre','cp_env','cp_env_save','cp_release','cp_rng','cp_g','cp_e','cp_w','cp_feat','cp_event_find','cp_hold_count','cp_phase','cp_start','cp_target'))
    fixture=headers+'''\noptions:maxmem=32768
@init
function cp_stop_event(ei when fade) (cp_e(ei)[0]=0;);
function cp_child_count() (0;);
function cp_child_mode() (0;);
function amp_evt(ei) (amp_event_base+ei*40;);
function amp_custom_eval(e t ei) (0;);
function cp_valid(id) (0;);
function tr_progress(e t) (0;);
function cp_path_eval(u) (path_x=0.5;path_y=0.5;);
function cp_desc(id d) (0.5;);
function cp_desc_conf(id d) (1;);
'''+helpers+'''
event_base=2176;walker_base=3000;walker_per_event=13;walker_total=104;
grain_base=10000;target_base=12000;target_role=12010;amp_curve_base=13000;amp_event_base=13100;env_old_base=14000;
ready=1;pe_ready=1;pg_ready=1;playable_grains=8;rng_state=max(1,floor(slider11));
@block
test_op==0 ? (
memset(event_base,0,512);
test_ei=cp_start(test_velocity,test_note,0,test_when,0,0,127,127,0);
test_e=cp_e(test_ei);test_seed=test_e[17];
cp_target(cp_w(test_ei*walker_per_event),test_e,test_when);test_force=target_base[0];
);
test_op==1 ? test_result=cp_env(event_base,test_time);
test_op==2 ? cp_release(0,test_time);
test_op==3 ? cp_env_save(event_base,test_time);
test_op==4 ? test_result=cp_curve(test_u,test_curve);
'''
    (out/'fixture.jsfx').write_text(fixture,encoding='utf-8')
    obj=out/('fixture.obj' if os.name=='nt' else 'fixture.o');exe=out/('native.exe' if os.name=='nt' else 'native')
    cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(out/'fixture.jsfx'),'--native-gfx-legacy','--out-ll',str(out/'fixture.ll'),'--out-h',str(out/'JSFXDSP.h'),'--out-obj',str(obj),'--opt','2']
    if os.name=='nt':cmd+=['--target','x86_64-pc-windows-msvc']
    subprocess.run(cmd,check=True,cwd=root)
    (out/'native.cpp').write_text(NATIVE,encoding='utf-8')
    subprocess.run(['clang++','-std=c++20','-O1','-UNDEBUG','-I'+str(out),'-I'+str(root/'src'),str(out/'native.cpp'),str(obj),'-o',str(exe)],check=True,cwd=root)
    subprocess.run([str(exe)],check=True,timeout=20,cwd=root)

NATIVE=r'''#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#undef assert
#define assert(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){assert(false);}
extern "C" double jsfx_memcpy(DSPJSFX_State* s,double dest,double source,double count){
  std::vector<double> copy;for(int i=0;i<int(count);++i)copy.push_back(double(s->mem[int(source)+i]));
  for(int i=0;i<int(count);++i)s->mem[int(dest)+i]=copy[i];return dest;
}
static int index(const char* name){for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return v.index;assert(false);return -1;}
static double get(DSPJSFX_State& s,const char* name){return double(s.vars[index(name)]);}
static void set(DSPJSFX_State& s,const char* name,double x){s.vars[index(name)]=x;}
int main(){
  DSPJSFX_State s{};za::jsfx::StateVariables vars;vars.bind(s,DSPJSFX_VARS_COUNT);std::vector<DSPJSFX_Cell> heap(32768);s.mem=heap.data();s.memN=heap.size();s.srate=48000;
  s.sliders[0]=.5;s.sliders[8]=1;s.sliders[9]=0;s.sliders[10]=12345;s.sliders[11]=0;
  s.sliders[24]=800;s.sliders[25]=5;s.sliders[26]=100;s.sliders[27]=.75;s.sliders[28]=180;
  s.sliders[30]=.5;s.sliders[31]=.5;s.sliders[32]=.5;s.sliders[46]=2;s.sliders[39]=1;s.sliders[40]=2;
  jsfx_init(&s);set(s,"test_note",60);set(s,"test_velocity",1);set(s,"test_when",0);
  std::set<double> evolving;
  for(int i=0;i<12;++i){set(s,"test_when",i*48000);jsfx_block(&s);assert(get(s,"test_ei")>=0);evolving.insert(get(s,"test_seed"));}
  assert(evolving.size()==12);std::cout<<"PASS Evolve: 12 identical MIDI notes receive 12 different event seeds\n";
  s.sliders[11]=1;std::set<double> fixed;
  for(int i=0;i<12;++i){jsfx_block(&s);fixed.insert(get(s,"test_seed"));}
  assert(fixed.size()==1);std::cout<<"PASS Fixed: identical notes deliberately reuse their event seed\n";
  set(s,"test_note",60);set(s,"test_velocity",.5);jsfx_block(&s);const auto midiSeed=get(s,"test_seed");const auto midiForce=get(s,"test_force");
  set(s,"test_note",-1);jsfx_block(&s);assert(get(s,"test_seed")==midiSeed && get(s,"test_force")==midiForce);
  assert(std::abs(midiForce-.5)<1e-12);set(s,"test_note",60);set(s,"test_velocity",1);jsfx_block(&s);assert(std::abs(get(s,"test_force")-1)<1e-12);
  s.sliders[8]=0;jsfx_block(&s);assert(std::abs(get(s,"test_force")-.5)<1e-12);
  std::cout<<"PASS matched velocity: MIDI/Generate share seed and target; full MIDI velocity changes Force from 0.5 to 1; Velocity Link 0 removes that difference\n";
  auto evaluate=[&](double time){set(s,"test_time",time);set(s,"test_op",1);jsfx_block(&s);return get(s,"test_result");};
  auto curve=[&](double u,double c){set(s,"test_u",u);set(s,"test_curve",c);set(s,"test_op",4);jsfx_block(&s);return get(s,"test_result");};
  auto e=2176;
  auto event=[&](double c){s.sliders[69]=c;s.sliders[70]=c;s.sliders[71]=c;set(s,"test_op",0);jsfx_block(&s);
    heap[e+5]=100;heap[e+6]=200;heap[e+7]=300;heap[e+8]=.25;heap[e+9]=500;heap[e+10]=600;heap[e+11]=.25;heap[e+19]=100;heap[e+33]=100;};
  event(0);
  for(int t=0;t<=700;++t){double expected=t<100 ? 0 : t<200 ? (t-100)/100. : t<300 ? 1-.75*(t-200)/100. : t<500 ? .25 : .25*std::max(0.,(600-t)/100.);
    assert(std::abs(evaluate(t)-expected)<1e-12);}
  for(double c:{-1.,-.5,0.,.5,1.}){double previous=0;assert(curve(0,c)==0 && curve(1,c)==1);
    for(int i=0;i<=1000;++i){double val=curve(i/1000.,c);assert(std::isfinite(val) && val>=previous && val>=0 && val<=1);previous=val;}}
  assert(curve(.5,-.7)>.5 && curve(.5,.7)<.5);
  event(.7);double captured=evaluate(150);s.sliders[69]=-.7;s.sliders[70]=-.7;s.sliders[71]=-.7;assert(evaluate(150)==captured);
  const auto before=evaluate(125);set(s,"test_op",2);set(s,"test_time",150);jsfx_block(&s);
  assert(evaluate(125)==before && std::abs(evaluate(150)-captured)<1e-12 && evaluate(250)==0);
  double previous=captured;for(int t=150;t<=250;++t){double val=evaluate(t);assert(val<=previous+1e-12 && val>=0);previous=val;}
  event(.6);const auto old=evaluate(150);set(s,"test_op",3);set(s,"test_time",400);jsfx_block(&s);
  heap[e+51]=-6;heap[e+54]=1/(std::exp(-6)-1);heap[e+5]=400;heap[e+6]=500;heap[e+7]=600;heap[e+9]=800;heap[e+10]=900;
  assert(evaluate(150)==old);std::cout<<"PASS curved ADSR: Linear reference, bounded monotone curves, per-note snapshot, causal early release, Continue history\n";
}
'''
if __name__=='__main__':main()
