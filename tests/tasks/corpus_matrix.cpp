#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include "JsfxTasks.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){assert(false);}
static double& var(DSPJSFX_State& s,const char* name){
    for(const auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return s.vars[v.index];
    assert(false);return s.vars[0];
}
int main(){
    auto runtime=std::make_unique<jsfx_tasks::Runtime>();
    std::vector<double> heap(300000);
    DSPJSFX_State s{};za::jsfx::StateVariables s_variables;s_variables.bind(s,DSPJSFX_VARS_COUNT);s.mem=heap.data();s.memN=heap.size();s.taskContext=runtime.get();s.srate=48000;s.samplesblock=256;
    jsfx_init(&s);
    for(int run=0;run<12;++run){
        std::cout<<"matrix run "<<run<<std::endl;int n=run==0?512:80;var(s,"xc_coarse_n")=n;var(s,"test_restart")=1;
        for(int a=0;a<n;++a)for(int i=0;i<30;++i)heap[a*48+i]=((a*7+i*13+run)%101)/101.;
        std::vector<double> held;
        if(run==11)for(int i=0;i<8;++i){double size=1,h=-2;while(h==-2)h=jsfx_task_api(&s,10,&size,1);assert(h>0);held.push_back(h);}
        bool restarted=false,submitted=false,finished=false;
        for(int block=0;block<50000;++block){
            jsfx_block(&s);if(block%5000==0)std::cout<<"phase "<<var(s,"xc_phase")<<" stage "<<var(s,"xc_task_stage")<<" cursor "<<var(s,"xc_cursor")<<" retiring "<<var(s,"xc_task_retiring")<<std::endl;submitted|=var(s,"xc_task")>0;
            if(run==1 && !restarted && var(s,"xc_task")>0){
                var(s,"test_restart")=1;heap[0]=.987;restarted=true;
            }
            if(var(s,"xc_phase")==5 && !var(s,"xc_task_retiring")){finished=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(finished);assert(var(s,"xc_task_fallbacks")==0);assert(run==11?!submitted:submitted);assert(run!=1 || restarted);
        double error=0;
        for(int a=0;a<n;++a)for(int b=0;b<n;++b){double v=0,h=0;
            for(int i=0;i<8;++i)v+=std::abs(std::sqrt(heap[a*48+i])-std::sqrt(heap[b*48+i]))*.065+std::abs(heap[a*48+20+i]-heap[b*48+20+i])*.035;
            for(int i=0;i<12;++i)h+=std::abs(heap[a*48+8+i]-heap[b*48+8+i]);
            double expected=std::clamp(1-v-.40*std::min(heap[a*48+28],heap[b*48+28])*h*.5,0.,1.);
            error=std::max(error,std::abs(expected-heap[24576+a*512+b]));
        }
        assert(error<1e-12);
        for(double h:held){double rc=-2;while(rc==-2)rc=jsfx_task_api(&s,14,&h,1);assert(rc>0);}
    }
    std::cout<<"Corpus matrix: 512 regions, multiple batches, restart, 12 rebuilds and exhausted-buffer fallback passed\n";
}
