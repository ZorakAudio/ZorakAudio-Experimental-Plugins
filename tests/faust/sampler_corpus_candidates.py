"""Real Corpus/Sample DSP islands, plus full-plugin experimental integrations.
Pinned sources and exact EEL reference functions are saved for reproducibility.
No recording is opened here. Candidates are not installed automatically.
"""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/faust-sections/sampler-corpus';BASE.mkdir(parents=True,exist_ok=True)
FIX=ROOT/'tests/faust/fixtures/sampler-corpus';FIX.mkdir(parents=True,exist_ok=True)
def source(key):
 p=ROOT/'plugins/Spectral'/key/'src'/(key+'.jsfx');f=FIX/(key+'.jsfx')
 if not f.exists():f.write_bytes(p.read_bytes())
 return f.read_text(encoding='utf-8')
def function(s,name):
 start=s.index('function '+name+'(');m=re.search(r'(?m)^function |^@',s[start+1:]);return s[start:start+1+m.start()] if m else s[start:]
def corpus():
 s=source('Corpus');start=s.index('td_enabled && slider45>0.000001 ? (',s.index('@sample'));end=s.index('output_smooth+=',start)
 original=s[start:end];init=[];pre=['cf_on=td_enabled && slider45>0.000001;cf_input_l=out_l;cf_input_r=out_r;'];post=[];dsp=[]
 states=['td_amount_rt','td_fast','td_slow','td_mix','td_dry_peak','td_block_trans','td_block_mix','td_block_guard','td_block_inpeak','td_block_outpeak']
 for name in states:init.append('cf_prev_'+name+'=0;');pre.append('cf_prev_'+name+'='+name+';')
 for i in range(3):
  pre.append(f'cf_p{i}=floor(td_pos_base[{i}]);cf_p{i}>=td_len{i} ? cf_p{i}=0;cf_off{i}={i}*1024+cf_p{i};cf_dl{i}=td_buf_l[cf_off{i}];cf_dr{i}=td_buf_r[cf_off{i}];')
  post.append(f'td_buf_l[cf_off{i}]=cf_wl{i};td_buf_r[cf_off{i}]=cf_wr{i};cf_p{i}+=1;cf_p{i}>=td_len{i} ? cf_p{i}=0;td_pos_base[{i}]=cf_p{i};')
 dsp+=['cf_amount=cf_prev_td_amount_rt+(slider45-cf_prev_td_amount_rt)*td_param_a;',
 'cf_amt=min(1,max(0,cf_amount));cf_abs=max(abs(cf_input_l),abs(cf_input_r));',
 'cf_fast=max(cf_abs,cf_prev_td_fast*td_fast_rel);cf_slow=cf_prev_td_slow+(cf_abs-cf_prev_td_slow)*td_slow_a;',
 'cf_gate=min(1,max(0,(cf_fast-0.00002)/0.0015));',
 'cf_trans=min(1,max(0,(cf_fast-cf_slow*1.12)/max(0.000001,cf_fast*0.82)))*cf_gate;',
 'cf_target=cf_amt*cf_trans;cf_mix=cf_prev_td_mix+(cf_target-cf_prev_td_mix)*select2(cf_target>cf_prev_td_mix,0.015,0.45);',
 'cf_peak=max(cf_abs,cf_prev_td_dry_peak*td_peak_rel);cf_g0=0.30+0.30*cf_amt;cf_g1=min(0.64,cf_g0+0.04);cf_g2=max(0.20,cf_g0-0.03);']
 for i in range(3):
  for ch in ['l','r']:
   inp=f'cf_input_{ch}' if i==0 else f'cf_y{ch}{i-1}'
   dsp.append(f'cf_y{ch}{i}=-cf_g{i}*{inp}+cf_d{ch}{i};cf_w{ch}{i}={inp}+cf_g{i}*cf_y{ch}{i};')
 dsp+=['cf_mix_now=min(1,max(0,cf_mix));cf_pl=cf_input_l*(1-cf_mix_now)+cf_yl2*cf_mix_now;cf_pr=cf_input_r*(1-cf_mix_now)+cf_yr2*cf_mix_now;',
 'cf_proc_peak=max(abs(cf_pl),abs(cf_pr));cf_ceiling=max(0.000000001,cf_peak*1.001);',
 'cf_guard0=select2(cf_proc_peak>cf_ceiling,1,min(1,cf_ceiling/cf_proc_peak));',
 'cf_goal=max(0.000000001,cf_peak*(1-0.28*cf_amt*cf_trans));',
 'cf_guard=min(1,max(0,select2((cf_trans>0.15)&(cf_proc_peak>cf_goal),cf_guard0,min(cf_guard0,cf_goal/cf_proc_peak))));',
 'out_l=select2(cf_on,cf_input_l,cf_pl*cf_guard);out_r=select2(cf_on,cf_input_r,cf_pr*cf_guard);']
 mapping={'td_amount_rt':'cf_amount','td_fast':'cf_fast','td_slow':'cf_slow','td_mix':'cf_mix','td_dry_peak':'cf_peak','td_block_trans':'max(cf_prev_td_block_trans,cf_trans)','td_block_mix':'max(cf_prev_td_block_mix,cf_mix_now)','td_block_guard':'max(cf_prev_td_block_guard,1-cf_guard)','td_block_inpeak':'max(cf_prev_td_block_inpeak,cf_abs)','td_block_outpeak':'max(cf_prev_td_block_outpeak,max(abs(out_l),abs(out_r)))'}
 for name,value in mapping.items():dsp.append(f'{name}=select2(cf_on,cf_prev_{name},{value});')
 # Work globals are published because subsequent code or future callers may inspect them.
 for name,value in {'td_transient':'cf_trans','td_guard_gain':'cf_guard','td_work_l':'cf_yl2','td_work_r':'cf_yr2'}.items():
  init.append('cf_prev_'+name+'=0;');pre.append('cf_prev_'+name+'='+name+';');dsp.append(f'{name}=select2(cf_on,cf_prev_{name},{value});')
 replacement='\n'.join(pre)+'\n@faust\n'+'\n'.join(dsp)+'\nprocess=spl0,spl1;\n@sample\ncf_on ? (\n'+'\n'.join(post)+'\n);\n'
 full=s[:start]+replacement+s[end:];full=full.replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
 kernel='desc:Corpus authentic transient diffusion qualification\nslider45:0.5<0,1,0.01>Amount\n@init\n'+function(s,'td_stage')+function(s,'td_prepare')+function(s,'td_clear')+'function cp_clamp(x lo hi) (min(hi,max(lo,x)););function cp_01(x) (min(1,max(0,x)););\ntd_pos_base=1;td_buf_l=10;td_buf_r=3082;td_enabled=1;td_prepare();td_clear();\n@slider\ncf_next=slider45>0.000001;cf_next && !td_enabled ? td_clear();td_enabled=cf_next;!td_enabled ? td_mix=0;\n@sample\nout_l=spl0;out_r=spl1;\n'+original+'spl0=out_l;spl1=out_r;\n'
 after=kernel.replace(original,replacement).replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
 return full,kernel,after

def sample():
 s=source('Sample');pre=['sf_input_l=outL;sf_input_r=outR;sf_on=(proc_hpf_hz>22 || proc_lpf_hz<min(19000,srate*0.47) || abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001);'];post=[];init=[];dsp=['sf_zap(x)=select2(abs(x)<DENORM_CUT,x,0);']
 for i in range(13):
  for j in range(5):
   name=f'sf_b{i}_{j}';pre.append(f'{name}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{j}];');init.append(name+'=0;')
  for ch,offset in [('l',5),('r',7)]:
   for z in [1,2]:
    prev=f'sf_prev_{i}_{ch}{z}';new=f'sf_next_{i}_{ch}{z}';init.extend([prev+'=0;',new+'=0;']);pre.append(f'{prev}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{offset+z-1}];');post.append(f'posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{offset+z-1}]={new};')
 for name in ['proc_hpf1_l','proc_hpf1_r','proc_lpf1_l','proc_lpf1_r']:
  prev='sf_prev_'+name;init.append(prev+'=0;');pre.append(prev+'='+name+';')
 enables={i:f'(proc_hpf_hz>22)&(proc_hpf_slope_idx>={i+1})' for i in range(4)}
 enables.update({4:'abs(proc_eq_band1_db)>0.000001',5:'abs(proc_eq_band2_db)>0.000001',6:'abs(proc_eq_band3_db)>0.000001'})
 enables.update({i:f'(proc_lpf_hz<min(19000,srate*0.47))&(proc_lpf_slope_idx>={i-6})' for i in range(7,11)})
 enables[11]='(proc_hpf_hz>22)&(posteq_hpf_res_db>0.0001)';enables[12]='(proc_lpf_hz<min(19000,srate*0.47))&(posteq_lpf_res_db>0.0001)'
 for i,expression in enables.items():dsp.append(f'sf_en{i}=sf_on & ({expression});')
 for ch,sign in [('l','+'),('r','-')]:
  dsp.append(f'sf_{ch}_in=sf_input_{ch}{sign}denorm_guard;')
  hp=f'sf_prev_proc_hpf1_{ch}+(sf_{ch}_in-sf_prev_proc_hpf1_{ch})*proc_hpf_coeff'
  dsp.append(f'proc_hpf1_{ch}=select2(sf_on & (proc_hpf_hz>22)&(proc_hpf_slope_idx<=0),sf_prev_proc_hpf1_{ch},sf_zap({hp}));')
  dsp.append(f'sf_{ch}_hp=select2((proc_hpf_hz>22)&(proc_hpf_slope_idx<=0),sf_{ch}_in,sf_{ch}_in-proc_hpf1_{ch});')
  current=f'sf_{ch}_hp'
  for i in [0,1,2,3,11,4,5,6,12]:
   dsp.append(f'sf_y{i}_{ch}=sf_b{i}_0*{current}+sf_prev_{i}_{ch}1;')
   dsp.append(f'sf_next_{i}_{ch}1=select2(sf_en{i},sf_prev_{i}_{ch}1,sf_zap(sf_b{i}_1*{current}-sf_b{i}_3*sf_y{i}_{ch}+sf_prev_{i}_{ch}2));')
   dsp.append(f'sf_next_{i}_{ch}2=select2(sf_en{i},sf_prev_{i}_{ch}2,sf_zap(sf_b{i}_2*{current}-sf_b{i}_4*sf_y{i}_{ch}));')
   following=f'sf_pass{i}_{ch}';dsp.append(f'{following}=select2(sf_en{i},{current},sf_zap(sf_y{i}_{ch}));');current=following
  lp=f'sf_prev_proc_lpf1_{ch}+({current}-sf_prev_proc_lpf1_{ch})*proc_lpf_coeff'
  dsp.append(f'proc_lpf1_{ch}=select2(sf_on & (proc_lpf_hz<min(19000,srate*0.47))&(proc_lpf_slope_idx<=0),sf_prev_proc_lpf1_{ch},sf_zap({lp}));')
  following=f'sf_lp_{ch}';dsp.append(f'{following}=select2((proc_lpf_hz<min(19000,srate*0.47))&(proc_lpf_slope_idx<=0),{current},proc_lpf1_{ch});');current=following
  for i in range(7,11):
   dsp.append(f'sf_y{i}_{ch}=sf_b{i}_0*{current}+sf_prev_{i}_{ch}1;')
   dsp.append(f'sf_next_{i}_{ch}1=select2(sf_en{i},sf_prev_{i}_{ch}1,sf_zap(sf_b{i}_1*{current}-sf_b{i}_3*sf_y{i}_{ch}+sf_prev_{i}_{ch}2));')
   dsp.append(f'sf_next_{i}_{ch}2=select2(sf_en{i},sf_prev_{i}_{ch}2,sf_zap(sf_b{i}_2*{current}-sf_b{i}_4*sf_y{i}_{ch}));')
   following=f'sf_pass{i}_{ch}';dsp.append(f'{following}=select2(sf_en{i},{current},sf_zap(sf_y{i}_{ch}));');current=following
  dsp.append(f"out{'L' if ch=='l' else 'R'}=select2(sf_on,sf_input_{ch},sf_zap({current}));")
 quiet='''denorm_near_silence(outL,outR) ? (
 posteq_filter_silent+=1;
 posteq_filter_silent==DENORM_ANALYZER_HOLD ? posteq_zero_filter_states();
 posteq_filter_silent>DENORM_ANALYZER_HOLD ? posteq_filter_silent=DENORM_ANALYZER_HOLD;
) : posteq_filter_silent=0;
'''
 replacement='\n'.join(pre)+'\n@faust\n'+'\n'.join(dsp)+'\nprocess=spl0,spl1;\n@sample\nsf_on ? (\n'+'\n'.join(post)+'\n'+quiet+');\n'
 full=s.replace('apply_processing_strip();',replacement+'posteq_char_active ? apply_posteq_character();\n(posteq_solo_target>0 || posteq_solo_mix>0.0001) ? apply_posteq_solo();',1).replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
 funcs='function clamp(x lo hi) (min(hi,max(lo,x)););\n'+''.join(function(s,n) for n in ['denorm_zap','denorm_near_silence','posteq_bq_identity','posteq_bq_set_peak','posteq_bq_set_highpass','posteq_bq_set_lowpass','posteq_bq_process_l','posteq_bq_process_r','posteq_zero_filter_states','apply_posteq_filter'])
 controls='''proc_hpf_hz=slider1;proc_lpf_hz=slider2;proc_hpf_slope_idx=slider3;proc_lpf_slope_idx=slider3;
proc_eq_band1_db=slider4;proc_eq_band2_db=-slider4*0.7;proc_eq_band3_db=slider4*0.4;
posteq_hpf_res_db=slider5;posteq_lpf_res_db=slider5;
proc_hpf_coeff=2*$pi*proc_hpf_hz/(2*$pi*proc_hpf_hz+srate);proc_lpf_coeff=2*$pi*proc_lpf_hz/(2*$pi*proc_lpf_hz+srate);
i=0;loop(4,posteq_bq_set_highpass(i,proc_hpf_hz,0.7071);posteq_bq_set_lowpass(7+i,proc_lpf_hz,0.7071);i+=1;);
posteq_bq_set_peak(4,300,0.7,proc_eq_band1_db);posteq_bq_set_peak(5,1600,1.1,proc_eq_band2_db);posteq_bq_set_peak(6,6000,0.8,proc_eq_band3_db);
posteq_bq_set_peak(11,proc_hpf_hz,0.7,posteq_hpf_res_db);posteq_bq_set_peak(12,proc_lpf_hz,0.7,posteq_lpf_res_db);
'''
 kernel='''desc:Sample authentic post-EQ qualification
slider1:20<20,1000,1>HP
slider2:20000<1000,20000,1>LP
slider3:0<0,4,1>Slope
slider4:0<-12,12,0.1>EQ
slider5:0<0,12,0.1>Resonance
@init
'''+funcs+'''DENORM_CUT=0.000000000000000001;DENORM_AUDIO_FLOOR=0.000000000001;DENORM_ANALYZER_HOLD=max(1024,floor(srate*0.25));POSTEQ_BQ_STRIDE=9;POSTEQ_BQ_COUNT=13;POSTEQ_CHAR_BQ_COUNT=0;posteq_bq_base=1;denorm_guard=0;
@slider
'''+controls+'\n@sample\noutL=spl0;outR=spl1;\n'+pre[0]+'\nsf_on ? apply_posteq_filter();\nspl0=outL;spl1=outR;\n'
 after=kernel.replace(pre[0]+'\nsf_on ? apply_posteq_filter();',replacement).replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
 return full,kernel,after
if __name__=='__main__':
 for key,generate in [('Corpus',corpus),('Sample',sample)]:
  out=BASE/key;out.mkdir(parents=True,exist_ok=True);full,old,new=generate()
  for name,contents in [('candidate.jsfx',full),('kernel-before.jsfx',old),('kernel-after.jsfx',new)]: (out/name).write_text(contents,encoding='utf-8')
  print(key,'integration and authentic kernel generated',flush=True)
 (FIX/'manifest.json').write_text(json.dumps({k:hashlib.sha256((FIX/(k+'.jsfx')).read_bytes()).hexdigest() for k in ['Corpus','Sample']},indent=2)+'\n')
