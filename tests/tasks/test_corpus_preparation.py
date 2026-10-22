"""Exercise Corpus's actual coordinator with real failed/cancelled arena tasks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def function(source,name):
    start=source.index('function '+name+'(')
    end=source.index('\n);',start)+4
    line=source.index('\n',start)
    return source[start:line if source[start:line].rstrip().endswith(');') else end]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path);ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path)
    a=ap.parse_args();root=a.repo or Path(__file__).resolve().parents[2]
    source=(a.source or root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text(encoding='utf-8')
    out=a.out or root/'build/tasks/corpus-preparation';out.mkdir(parents=True,exist_ok=True)
    helpers='\n'.join(function(source,n) for n in ('cg_key','cg_retire','cg_cancel','cg_error','cg_fail','cg_step'))
    fallback=next(line for line in source.splitlines() if '? cp_build(min(4096' in line)
    fixture='''options:maxmem=65536
@init
function cp_hold_count() (0;);
function cp_busy() (0;);
function cp_build(work) (test_fallback_calls+=1;build_phase=0;ready=loaded_count>0;);
function cg_index() (task_report(5,1,0,234);1;);
function cg_features() (
  task_report(1,1,0,234);
  test_mode==1 ? task_result(-999999);
  test_mode==2 ? loop(4096,while(1)(task_report(1,1,0,234);););
  1;
);
function cg_structure() (1;);
function cg_context() (1;);
function cg_pe() (1;);
function cg_publish() (ready=1;build_phase=0;test_published+=1;1;);
'''+helpers+'''
hist=64;candidates=128;source_base=256;traj_base=512;traj_memory_end=1024;
td_buf_l=2048;td_memory_end=2304;xc_params=64;pe_event_locks=96;
gr_record=3000;gr_memory_end=4000;grain_count=234;
build_phase=1;loaded_count=1;bank_pool=0;generation=0;auto_config_key=123;
cg_retry_key=123;
@block
cg_step();
test_invalid=cg_error(999999);
'''+fallback+'\n'
    (out/'fixture.jsfx').write_text(fixture,encoding='utf-8')
    obj=out/('fixture.obj' if os.name=='nt' else 'fixture.o')
    cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(out/'fixture.jsfx'),'--native-gfx-legacy','--out-ll',str(out/'fixture.ll'),'--out-h',str(out/'JSFXDSP.h'),'--out-obj',str(obj),'--opt','2']
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
        cmd+=['--target','x86_64-pc-windows-msvc']
    subprocess.run(cmd,check=True,cwd=root)
    (out/'native.cpp').write_text(NATIVE,encoding='utf-8')
    exe=out/('native.exe' if os.name=='nt' else 'native')
    subprocess.run(['clang++','-std=c++20','-O1','-UNDEBUG','-I'+str(out),'-I'+str(root/'src'),str(out/'native.cpp'),str(obj),'-o',str(exe)],check=True,cwd=root)
    subprocess.run([str(exe)],check=True,timeout=30,cwd=root)

NATIVE=r'''#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include "JsfxTasks.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#undef assert
#define assert(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::_Exit(1);}}while(0)
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){assert(false);}
static int index(const char* name){for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return v.index;assert(false);return -1;}
static double get(DSPJSFX_State& s,const char* name){return double(s.vars[index(name)]);}
static void set(DSPJSFX_State& s,const char* name,double x){s.vars[index(name)]=x;}
static int denied=0;
static bool boundary(DSPJSFX_State*){if(denied>0){--denied;return false;}return true;}
int main(){
  for(int mode=0;mode<7;++mode){
    auto runtime=std::make_unique<jsfx_tasks::Runtime>();DSPJSFX_State s{};
    za::jsfx::StateVariables variables;variables.bind(s,DSPJSFX_VARS_COUNT);
    s.memN=65536;s.mem=static_cast<DSPJSFX_Cell*>(std::calloc(s.memN,sizeof(DSPJSFX_Cell)));s.taskContext=runtime.get();s.srate=48000;
    runtime->setHeapSwapHooks([](DSPJSFX_State* x){return x->mem && x->memN==65536;},[](DSPJSFX_State*){},boundary,[](DSPJSFX_State*){});
    if(mode==5)runtime->setHeapSwapHooks([](DSPJSFX_State*){return false;},[](DSPJSFX_State*){});
    jsfx_init(&s);set(s,"test_mode",mode);bool cancelled=false,finished=false;int status=0;
    if(mode==6)set(s,"loaded_count",0);
    denied=mode==3 ? 40 : 0;double heldArena=0;
    if(mode==4){double args[]={65536,0,0};heldArena=jsfx_task_api(&s,30,args,3);assert(heldArena>0);}
    for(int tick=0;tick<5000;++tick){
      if(mode==4 && tick>=40 && heldArena>0){if(jsfx_task_api(&s,26,&heldArena,1)>0)heldArena=0;}
      jsfx_block(&s);
      if(mode==3 || mode==4)assert(get(s,"test_fallback_calls")==0);
      if(mode==2 && !cancelled && get(s,"cg_t1")>0){
        double task=get(s,"cg_t1");status=int(jsfx_task_api(&s,0,&task,1));
        if(status==1 || status==2){double result=jsfx_task_api(&s,3,&task,1);cancelled=result>0;}
      }
      if(mode==0 || mode==3 || mode==4 ? get(s,"cg_completed")==1 : mode==6 ? get(s,"build_phase")==0 : get(s,"cg_disabled")>0){finished=true;break;}
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(finished && get(s,"test_invalid")==-1);
    if(mode==0 || mode==3 || mode==4){assert(get(s,"ready")==1 && get(s,"test_published")==1 && get(s,"cg_failures")==0);assert(mode==0 || get(s,"cg_clone_retries")>=40);}
    else if(mode==1 || mode==2){
      assert(get(s,"test_published")==0 && get(s,"ready")==0);
      assert(get(s,"rebuild_request")==1 && get(s,"cg_failures")<=1);
      // A fast failure may reject the next submission before status polling;
      // that existing recovery path does not increment the polling counter.
      if(get(s,"cg_failures")>0){
        assert(get(s,"cg_last_failure")==4 || get(s,"cg_last_failure")==5);
        // Pending cancellation can precede the feature writer's first report.
        assert(get(s,"cg_last_fail_stage")==1 || (mode==2 && get(s,"cg_last_fail_stage")==5));
      }
      assert(mode!=2 || cancelled);
    }else assert(get(s,"test_fallback_calls")==1 && get(s,"ready")==(mode==5 ? 1 : 0) && get(s,"test_published")==0);
    for(int tick=0;tick<2000 && (get(s,"cg_retiring")>0 || runtime->hasOutstandingWorkForIdle());++tick){
      jsfx_block(&s);std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(get(s,"cg_retiring")==0 && !runtime->hasOutstandingWorkForIdle());
    runtime.reset();std::free(s.mem);
    const char* labels[]={"success","failed feature writer recovery","cancelled feature writer recovery","busy heap boundary retry","occupied arena retry","unsupported arena incremental fallback","empty bank completes without a graph"};
    std::cout<<"PASS coordinator "<<labels[mode]<<"\n";
  }
}
'''

if __name__=='__main__':main()
