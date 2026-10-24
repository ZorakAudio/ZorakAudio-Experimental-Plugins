from pathlib import Path
import importlib.util,os,subprocess,sys
root=Path(__file__).resolve().parents[2];out=root/'build/tasks/corpus-detection';out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('grains',root/'tests/tasks/test_corpus_grains.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=(root/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text(encoding='utf-8')
names=['cp_clamp','cp_01','cp_db','cp_amp','cp_g','cp_s','cp_auto','cp_quiet_floor','cp_quiet','cp_gap_cache','cp_gap_cost','cp_gap_request','cp_gap_filter','cp_length_value','cp_desc_extent','cp_length_index','cp_axis','cp_filter_code','ui_tr_scale','ui_tr_to_view','ui_tr_from_view','cp_play_seconds','cp_bin_active','cp_play_region','cp_sensor_change','cp_micro_target','cp_merge_group','cp_passage_finish','cp_passage_step','cp_merge_begin','cp_merge_step','cp_target']
fixture='options:maxmem=100000\n'+ '\n'.join(x for x in source.splitlines() if x.startswith('slider') and ':' in x) + '''
@init
function cp_feat(id d) (cp_g(id)[16+d];);
function cp_fingerprint_add(v) (test_hash+=v;);
function cp_phase(e t) (phase_stage=2;phase_release=0;);
function tr_time(e t) (0;);
function tr_prepare(u du) (0;);
function tr_enabled(i) (test_lane>=0 && i==test_lane;);
function tr_lane(i) (75000+i*40;);
function tr_lane_value(i u) (test_lane_target;);
function cp_path_eval(u) (0;);
function cp_valid(id) (id>=0 && id<grain_count && cp_g(id)[9]<0.5;);
function cp_range(vertex src) (range_lo=0;grain_count;);
function cp_rng() (0.25;);
function xc_allowed(id src transfer guide) (cp_valid(id) && (!transfer || cp_g(id)[1]!=guide););
function cp_desc(id d) (cp_feat(id,d););
'''+'\n'.join(m.function(source,n) for n in names)+'''
test_lane=-1;desc_value_range=71000;ui_tr_range=71000;desc_res_counts=70000;desc_reservoir=72000;candidates=3500;postings=3600;grain_base=4096;source_base=0;auto_sources=64;passage_work=2000;hist=1024;scout_hist=256;
@block
test_op==1 ? cp_merge_begin();
test_op==2 ? cp_merge_step(test_work);
test_op==3 ? cp_play_region(test_id,1);
test_op==4 ? cp_target(3000,3200,5000);
test_op==5 ? cp_gap_cache(test_id);
test_op==6 ? test_result=cp_gap_cost(test_id);
test_op==7 ? test_result=cp_length_value(test_id);
test_op==8 ? cp_length_index(test_id);
test_op==9 ? cp_gap_filter(0);
test_op==10 ? cp_gap_request();
test_op==11 ? test_result=cp_axis(test_id);
test_op==12 ? test_result=cp_filter_code();
test_op==13 ? (ui_tr_scale(test_id);test_result=ui_tr_from_view(test_u);test_roundtrip=ui_tr_to_view(test_result););
test_op==14 ? cp_desc_extent(test_id,test_u);
@sample
spl0=0;spl1=0;
'''
(out/'fixture.jsfx').write_text(fixture)
cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(out/'fixture.jsfx'),'--native-gfx-legacy','--out-ll',str(out/'fixture.ll'),'--out-h',str(out/'JSFXDSP.h'),'--out-obj',str(out/'fixture.obj'),'--meta',str(out/'meta.json'),'--opt','2','--target','x86_64-pc-windows-msvc']
subprocess.run(cmd,cwd=root,check=True)
(out/'native.cpp').write_text(r'''
#include "JSFXDSP.h"
#include "JsfxStateVariables.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL "<<#x<<" line "<<__LINE__<<std::endl;std::exit(1);}}while(0)
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){CHECK(n<=s->memN);}
static int idx(const char* name){for(auto& v:DSPJSFX_VARS)if(std::string(v.name)==name)return v.index;std::cerr<<name;std::exit(2);}
int main(){
 DSPJSFX_State s{};za::jsfx::StateVariables vars;vars.bind(s,DSPJSFX_VARS_COUNT);std::vector<DSPJSFX_Cell> heap(100000);s.mem=heap.data();s.memN=heap.size();s.srate=48000;jsfx_init(&s);
 auto set=[&](const char* n,double x){s.vars[idx(n)]=x;};auto get=[&](const char* n){return double(s.vars[idx(n)]);};
 auto make=[&](int n,int mode,double limit=0,int sources=1){for(auto& x:heap)x=0;jsfx_init(&s);set("source_count",sources);set("grain_count",n);set("total_seconds",n*.05);set("window_setting",0);set("passage_mode",mode);set("passage_run",mode==2);set("passage_sensitivity",.5);set("passage_limit",limit);set("quiet_mode",0);set("test_work",64);
  for(int src=0;src<sources;++src){int base=src*8;int count=n/sources;heap[base]=src;heap[base+2]=count*2400;heap[base+3]=48000;heap[base+4]=2;heap[base+5]=src*count;heap[base+6]=count;heap[64+src*8+2]=.2;}
  for(int i=0;i<n;++i){int g=4096+i*64,src=i/(n/sources);heap[g]=src;heap[g+1]=src;heap[g+2]=(i%(n/sources))*2400;heap[g+3]=2400;heap[g+4]=48000;heap[g+5]=2;heap[g+6]=.2;heap[g+7]=.4;heap[g+12]=.3;heap[g+13]=.4;heap[g+14]=.2;heap[g+15]=.1;heap[g+19]=.2;heap[g+21]=heap[g+22]=.2;heap[g+49]=1;for(int j=0;j<16;++j)heap[g+24+j]=.4;}
 };
 auto run=[&](){set("test_op",1);jsfx_block(&s);set("test_op",2);int steps=0;while(!get("merge_done") && ++steps<10000)jsfx_block(&s);CHECK(get("merge_done"));return steps;};
 make(240,2);int steps=run();CHECK(get("grain_count")==1);CHECK(heap[4096+3]==576000);CHECK(steps>=480);set("test_op",3);set("test_id",0);s.sliders[51]=80;jsfx_block(&s);CHECK(get("play_frames")==576000 && get("play_srcpos")==0);
 std::cout<<"PASS 12-second steady source is one intact Coherent grain; no hidden 610/680/1200ms cap; cooperative single-sensor work\n";
 make(240,2,2);run();CHECK(get("grain_count")==6);for(int i=0;i<6;++i)CHECK(heap[4096+i*64+3]<=96000);
 make(240,2,0,2);run();CHECK(get("grain_count")==2);CHECK(heap[4096+1]==0 && heap[4096+64+1]==1);CHECK(heap[4096+3]==288000 && heap[4096+64+3]==288000);
 std::cout<<"PASS optional limit, exact source coverage and no cross-file merging\n";
 make(240,2);for(int i=100;i<110;++i){int g=4096+i*64;heap[g+6]=.001;heap[g+7]=.002;}run();CHECK(get("grain_count")==3);CHECK(heap[4096+3]==240000);CHECK(heap[4096+64+2]==240000 && heap[4096+64+3]==24000);CHECK(heap[4096+128+2]==264000);
 make(240,2);for(int i=100;i<102;++i)heap[4096+i*64+6]=.001;run();CHECK(get("grain_count")==1);
 make(240,2);for(int i=120;i<240;++i){int g=4096+i*64;heap[g+6]=.8;heap[g+12]=.95;heap[g+13]=.05;}run();CHECK(get("grain_count")==2);CHECK(heap[4096+3]==288000);
 std::cout<<"PASS sustained pause / persistent timbre-change boundaries; brief syllabic dip retained\n";
 make(240,1);run();int autoN=int(get("grain_count"));double autoMax=0;for(int i=0;i<autoN;++i)autoMax=std::max(autoMax,double(heap[4096+i*64+3]));
 make(240,0);run();int aggressiveN=int(get("grain_count"));CHECK(aggressiveN>autoN*2);for(int i=0;i<aggressiveN;++i)CHECK(heap[4096+i*64+3]<=autoMax);
 std::cout<<"PASS Aggressive yields shorter units: "<<aggressiveN<<" vs Auto "<<autoN<<"; Coherent 1\n";
 make(1,2);run();CHECK(get("grain_count")==1 && heap[4096+3]==2400);
 make(241,2);heap[4096+240*64+3]=100;heap[2]=240*2400+100;run();CHECK(get("grain_count")==1 && heap[4096+3]==240*2400+100);
 make(240,2);for(int i=0;i<240;++i){int g=4096+i*64;heap[g+6]=0;heap[g+7]=0;}run();CHECK(get("grain_count")==1);CHECK(std::isfinite(double(heap[4096+45])) && std::isfinite(double(heap[4096+46])));
 std::cout<<"PASS short source, tiny tail, silent source and finite aggregate descriptors\n";
 set("target_base",2048);set("target_role",2080);set("h_active",0);set("path_count",0);set("test_op",4);heap[3005]=-1;heap[3213]=.5;heap[3212]=1;heap[3210]=10000;s.sliders[45]=0;s.sliders[49]=0;
 for(double baseline:{.2,.8}){for(int d=0;d<8;++d)heap[3024+d]=heap[3016+d]=baseline;s.sliders[30]=.5;jsfx_block(&s);CHECK(std::abs(double(heap[2049])-baseline)<1e-10);s.sliders[30]=1;jsfx_block(&s);CHECK(std::abs(double(heap[2049])-std::min(1.,baseline+.4))<1e-10);}
 s.sliders[32]=1;jsfx_block(&s);CHECK(heap[2051]==1);CHECK(std::abs(double(heap[2053])-.975)<1e-10);
 s.sliders[1]=0;jsfx_block(&s);double target=heap[2049];s.sliders[1]=1;jsfx_block(&s);CHECK(heap[2049]==target);
 std::cout<<"PASS Materials actual cp_target: Brightness50% retains identity .2/.8; Brightness100% targets .6/1; Roughness offsets Motion and Crest; Variation does not set a target\n";

 make(3,2);set("gap_active",1);set("test_id",0);int z=4096;heap[z+3]=48000;heap[z+6]=.00002;for(int j=0;j<16;++j)heap[z+24+j]=.00003;set("test_op",5);jsfx_block(&s);set("test_op",6);jsfx_block(&s);CHECK(get("test_result")>.99);heap[z+3]=480;jsfx_block(&s);CHECK(get("test_result")<.4);
 heap[64+2]=.0001;heap[7]=.001;heap[z+6]=.0001;for(int j=0;j<16;++j)heap[z+24+j]=.0003;set("test_op",5);jsfx_block(&s);set("test_op",6);jsfx_block(&s);CHECK(get("test_result")==0);
 for(int id=0;id<3;++id){heap[4096+id*64+3]=48000;heap[4096+id*64+62]=1;heap[4096+id*64+63]=1;heap[3600+id]=id;}heap[4096+2*64+62]=0;heap[3500]=0;heap[3501]=-1;set("candidate_n",1);set("test_op",9);jsfx_block(&s);CHECK(get("candidate_n")==1 && heap[3500]==2);
 heap[4096+2*64+62]=1;heap[3500]=0;set("candidate_n",1);jsfx_block(&s);CHECK(get("candidate_n")==1 && heap[3500]==0 && get("gap_fallbacks")>0);
 s.sliders[244]=1;set("target_base",2048);heap[2048]=.5;set("test_lane",0);set("test_lane_target",.05);heap[75001]=0;heap[75002]=0;set("test_op",10);jsfx_block(&s);CHECK(get("gap_active")==0);heap[75002]=2;jsfx_block(&s);CHECK(get("gap_active")==1);s.sliders[244]=0;jsfx_block(&s);CHECK(get("gap_active")==0);
 std::cout<<"PASS quiet cache: scale-relative whisper preservation, duration-aware gap cost, audible fallback recovery, all-quiet scope, explicit low Energy override and Allow bypass\n";
 set("auto_window_min",.01);set("max_grain_seconds",10);heap[4096+3]=480;heap[4096+64+3]=480000;heap[4096+128+3]=std::sqrt(.01*10)*48000;set("test_op",7);for(int id=0;id<3;++id){set("test_id",id);jsfx_block(&s);CHECK(std::abs(get("test_result")-(id==0?0:id==1?1:.5))<1e-10);set("test_op",8);jsfx_block(&s);set("test_op",7);}
 double supported=0;for(int i=848;i<864;++i)supported+=double(heap[70000+i]);CHECK(supported==3);s.sliders[39]=52;s.sliders[245]=0;set("test_op",11);set("test_id",0);jsfx_block(&s);CHECK(get("test_result")==52);s.sliders[245]=1;jsfx_block(&s);CHECK(get("test_result")==53);s.sliders[247]=3;set("test_op",12);jsfx_block(&s);CHECK(get("test_result")==161);
 set("auto_window_min",1);set("max_grain_seconds",1);set("test_op",7);jsfx_block(&s);CHECK(get("test_result")==.5);
 std::cout<<"PASS Length: logarithmic range, exact shortest/longest, uniform fallback, descriptor reservoir bounds and legacy map/filter extension slots\n";
 // Actual descriptor range, not 16-bin edges, is the editor's coordinate
 // transform. A narrow centroid range should occupy almost the entire graph.
 set("test_id",16);set("test_op",14);set("test_u",.021);jsfx_block(&s);set("test_u",.039);jsfx_block(&s);
 CHECK(std::abs(double(heap[71000+48])-.021)<1e-12 && std::abs(double(heap[71000+49])-.039)<1e-12);
 set("test_op",13);for(double u:{0.,.25,.5,.75,1.}){set("test_u",u);jsfx_block(&s);CHECK(std::abs(get("test_roundtrip")-u)<1e-10);CHECK(get("test_result")>.020 && get("test_result")<.040);}
 set("test_id",17);set("test_op",14);set("test_u",.3);jsfx_block(&s);set("test_op",13);set("test_u",.5);jsfx_block(&s);CHECK(std::isfinite(get("test_result")) && std::abs(get("test_result")-.3)<1e-12);
 set("test_id",18);jsfx_block(&s);CHECK(get("test_result")==.5);set("test_u",1);jsfx_block(&s);CHECK(get("test_result")==1);
 std::cout<<"PASS corpus Y scale: actual cached extrema, narrow range, inverse drag mapping, flat/silent fallback, no curve or scoring changes\n";

}
''')
cmd=['clang++','-std=c++20','-O1','-UNDEBUG','-I'+str(out),'-I'+str(root/'src'),str(out/'native.cpp'),str(out/'fixture.obj'),'-o',str(out/'native.exe')]
subprocess.run(cmd,cwd=root,check=True);subprocess.run([str(out/'native.exe')],cwd=root,check=True)
