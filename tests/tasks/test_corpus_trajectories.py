"""Compile Corpus's actual trajectory functions with deterministic cached evidence."""
import argparse,os,subprocess,sys
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path);ap.add_argument('--out',type=Path);args=ap.parse_args()
    root=args.repo or Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/tasks'))
    def function(source,name):
        a=source.index('function '+name+'(');e=source.index('\n',a);line=source[a:e]
        return line if line.count('(')==line.count(')') else source[a:source.index('\n);',a)+3]
    source=(root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text(encoding='utf-8')
    out=args.out or root/'build/tasks/corpus-trajectories';out.mkdir(parents=True,exist_ok=True)
    helpers=['cp_clamp','cp_01','tr_curve_eval_samples','tr_curve_eval','tr_fill_linear_samples','tr_param_notify','tr_param_id','tr_param_set','tr_store_lane','tr_sync_params','tr_enabled','tr_time','tr_fold_mode','tr_fold','tr_lane','tr_progress','tr_lane_value','tr_prepare','tr_penalty_raw','tr_penalty','tr_future_penalty','tr_reset_lane','tr_set_linear','tr_history_clear','tr_clear_all','tr_brush_begin','tr_brush_data','tr_brush_commit','tr_preset','tr_meta','tr_candidates','ui_notify','ui_end_gesture']
    fixture='\n'.join(l for l in source.splitlines() if l.startswith('slider'))+'''\noptions:maxmem=65536
@init
function cp_desc(id d) (features[id*53+d];);
function cp_desc_conf(id d) (confidence;);
function xc_ok(id) (id>=0 && id<64;);
function xc_n(id) (neighbors+id*16;);
function xc_next(id) (min(63,id+1););
function cp_valid(id) (id>=0 && id<64;);
function xc_allowed(id src transfer guide) (cp_valid(id););
function pe_active_request() (0;);
function h_add_candidate(id route src) (candidate_n<64 && cp_valid(id) ? (candidates[candidate_n]=id;candidate_n+=1;););
'''+ '\n'.join(function(source,n) for n in helpers)+'''
traj_base=0;traj_runtime_base=500;traj_history_base=600;traj_brush_stage=1500;tr_param_cache=1600;
features=2000;neighbors=5500;candidates=7000;desc_res_counts=8000;desc_reservoir=9000;path_res_counts=40000;path_reservoir=41000;
confidence=1;xc_ready=1;memset(tr_param_cache,-1,156);
tr_eval_base=54000;tr_eval_tags=tr_eval_base+64;tr_eval_values=tr_eval_tags+768;
@block
test_op==0 ? tr_sync_params();
test_op==1 ? test_result=tr_penalty(test_id,test_u);
test_op==2 ? test_result=tr_progress(7500,test_t);
test_op==3 ? (tr_reset_lane(0,1,1);tr_set_linear(0,1,0,0,1,test_strength);tr_store_lane(0););
test_op==4 ? tr_meta(test_arg);
test_op==5 ? (candidate_n=0;traj_u=test_u;tr_candidates(0););
test_op==6 ? test_result=tr_future_penalty(test_id,test_u,test_du);
test_op==7 ? test_result=tr_curve_eval_samples(100,32,2,test_u);
test_op==8 ? (ui_notify(test_id,test_u,1);ui_end_gesture(test_id););
test_op==9 ? tr_brush_commit();
test_op==10 ? tr_prepare(test_u,test_du);
test_op==11 ? test_result=tr_penalty_raw(test_id,test_u,-1);
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
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
static int changed=-1,automated=-1,ended=-1;
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){CHECK(false);}
extern "C" double jsfx_memcpy(DSPJSFX_State* s,double d,double a,double n){std::vector<double> v;for(int i=0;i<int(n);++i)v.push_back(double(s->mem[int(a)+i]));for(int i=0;i<int(n);++i)s->mem[int(d)+i]=v[i];return d;}
extern "C" int jsfx_sliderchange(DSPJSFX_State*,double,int i){changed=i;return 0;}
extern "C" int jsfx_slider_automate(DSPJSFX_State*,double,int i,double end){automated=i;if(end>0)ended=i;return 0;}
static int idx(const char* n){for(auto& v:DSPJSFX_VARS)if(std::string(v.name)==n)return v.index;CHECK(false);return -1;}
struct Fixture{
 DSPJSFX_State s{};za::jsfx::StateVariables vars;std::vector<DSPJSFX_Cell> heap;
 Fixture():heap(65536){vars.bind(s,DSPJSFX_VARS_COUNT);s.mem=heap.data();s.memN=heap.size();s.srate=48000;s.sliders[80]=1;s.sliders[82]=2;s.sliders[84]=-1;for(int l=0;l<12;++l){s.sliders[85+l*13]=0;s.sliders[86+l*13]=.85;for(int j=0;j<11;++j)s.sliders[87+l*13+j]=j==10?8127:1040319;}jsfx_init(&s);op(0);}
 void set(const char* n,double x){s.vars[idx(n)]=x;}double get(const char* n){return double(s.vars[idx(n)]);}
 double op(int x){set("test_op",x);jsfx_block(&s);return get("test_result");}
 double penalty(double strength){s.sliders[86]=strength;op(0);set("test_id",0);set("test_u",0);return op(1);}
};
int main(){
 Fixture f;f.set("test_strength",1);f.op(3);f.heap[2001]=1;
 CHECK(std::abs(f.penalty(1)-1)<1e-12);CHECK(std::abs(f.penalty(.2)-.2)<1e-12);CHECK(f.penalty(0)==0 && f.get("tr_active_eval")==0);
 f.s.sliders[80]=0;CHECK(f.penalty(1)==0);f.op(5);CHECK(f.get("candidate_n")==0);f.s.sliders[80]=2;CHECK(f.penalty(1)==2);f.s.sliders[80]=1;
 f.set("test_arg",4);f.op(4);CHECK(f.penalty(1)==2);CHECK(std::abs(f.penalty(.2)-1.2)<1e-12);CHECK(f.penalty(0)==1);f.set("test_arg",512);f.op(4);
 f.s.sliders[84]=1;CHECK(f.penalty(1)==0);f.s.sliders[84]=-1;
 f.s.sliders[85]=3072+1+64;f.op(0);f.heap[2001]=.2;f.set("test_u",.5);CHECK(f.op(1)==0);f.heap[2001]=.8;CHECK(f.op(1)>0);
 f.s.sliders[85]=3072+1+128;f.op(0);f.heap[2001]=.8;CHECK(f.op(1)==0);f.heap[2001]=.2;CHECK(f.op(1)>0);
 std::cout<<"PASS actual strength 0/20/100%, master exact bypass/boost, Solo and one-sided constraints\n";
 f.set("test_arg",3);f.op(4);CHECK(f.heap[0]==0 && f.heap[4]==1);f.op(4);CHECK(f.heap[0]==1);f.set("test_arg",4);f.op(4);for(int j=0;j<40;++j)CHECK(f.heap[j]==f.heap[40+j]);
 auto state=f.s.sliders;std::vector<double> before(480);for(int j=0;j<480;++j)before[j]=f.heap[j];for(int j=0;j<480;++j)f.heap[j]=0;for(int j=0;j<156;++j)f.heap[1600+j]=-1;f.op(0);
 for(int l=0;l<12;++l){for(int j:{0,1,2,3,4,5,6})CHECK(f.heap[l*40+j]==before[l*40+j]);for(int j=8;j<40;++j)CHECK(std::abs(double(f.heap[l*40+j])-before[l*40+j])<=1./252+1e-12);}
 for(int j=85;j<241;++j)if((j-85)%13!=1)CHECK(double(float(f.s.sliders[j]))==f.s.sliders[j]);
 // Host-applied curve storage changes update all 32 samples without reanalysis.
 f.s.sliders[87]=0;f.op(0);CHECK(f.heap[8]==0 && f.heap[9]==0 && f.heap[10]==0);
 std::cout<<"PASS mute/duplicate and float-exact persistent curve/meta/strength decoding\n";
 f.heap[7505]=1000;f.heap[7512]=1;f.heap[7510]=49000;f.set("test_t",25000);f.s.sliders[81]=0;CHECK(f.op(2)==.5);
 f.heap[7512]=0;CHECK(f.op(2)==.25);f.set("test_t",121000);CHECK(f.op(2)==.25);f.s.sliders[81]=1;f.set("test_t",121000);CHECK(f.op(2)==1);
 f.s.sliders[81]=2;CHECK(f.op(2)==.25);f.s.sliders[81]=3;CHECK(f.op(2)==.75);f.s.sliders[83]=.5;CHECK(f.op(2)==.25);
 f.set("test_t",1000);CHECK(f.op(2)==.5);f.s.sliders[83]=0;
 f.s.sliders[81]=0;f.heap[100]=.1;f.heap[131]=.9;f.set("test_u",1);CHECK(f.op(7)==.9);
 std::cout<<"PASS Event/held/Once/Repeat/Bounce clocks, phase, and Step endpoint\n";
 // All cached neighbors have brightness .1: Repeat's future must wrap to .1.
 f.set("test_arg",512);f.op(4);f.set("test_strength",1);f.op(3);f.s.sliders[81]=2;
 for(int id=0;id<64;++id){f.heap[2000+id*53+1]=.1;for(int j=0;j<4;++j)f.heap[5500+id*16+j]=id;}
 f.set("test_id",0);f.set("test_u",.9);f.set("test_du",.2);CHECK(f.op(6)<.061);
 for(int id=0;id<64;++id){f.set("test_id",id);f.set("test_u",.9);f.op(10);for(double u:{.9,1.1,1.3}){f.set("test_u",u);double cached=f.op(1);CHECK(cached==f.op(11));CHECK(f.op(1)==cached);}}
 f.set("test_u",.9);f.op(10);f.heap[2000+256*53+1]=.93;for(int id:{0,256,0}){f.set("test_id",id);CHECK(f.op(1)==f.op(11));}
 f.set("tr_cache_prepared",0);std::cout<<"PASS bounded score cache is exactly equal to uncached scoring at all three instants\n";
 // Cached target-route candidates exist before the general graph can fill.
 for(int b=0;b<16;++b){f.heap[8000+16+b]=8;for(int j=0;j<8;++j)f.heap[9000+(16+b)*32+j]=b;}
 f.set("test_u",.5);f.op(5);CHECK(f.get("candidate_n")>0 && f.get("candidate_n")<=24);CHECK(f.heap[7000]>=7 && f.heap[7000]<=9);
 for(int id=81;id<=241;++id){changed=automated=ended=-1;f.set("test_id",id);f.set("test_u",.37);f.op(8);CHECK(changed==id-1 && automated==id-1 && ended==id-1);}
 std::cout<<"PASS loop-aware lookahead, bounded candidate route, and every exact host notification 81..241\n";
}
'''
if __name__=='__main__':main()
