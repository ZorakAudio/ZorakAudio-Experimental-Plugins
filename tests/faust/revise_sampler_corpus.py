"""Experimental Corpus export pruning and isolated private-state Sample EQ.
Uses previously prepared candidates; never changes production sources.
"""
from pathlib import Path
import re,sys
r=Path.cwd();b=r/'build/faust-sections/sampler-corpus'
p=b/'Corpus/block-candidate.jsfx';s=p.read_text(encoding='utf-8');a=s.index('@faust');z=s.index('\n@sample',a);body=s[a:z]
for n in ['td_amount_rt','td_fast','td_slow','td_dry_peak','td_guard_gain','td_transient']:
 body=re.sub(r'\b'+n+r'\b','private_'+n,body)
(b/'Corpus/revised-candidate.jsfx').write_text(s[:a]+body+s[z:],encoding='utf-8')
s=(b/'Sample/kernel-before.jsfx').read_text(encoding='utf-8');sys.path.insert(0,str(r/'tests/faust'));from sampler_corpus_candidates import function
f=function(s,'apply_posteq_filter');start=f.index('  proc_hpf_hz > 22');end=f.index('  abs(proc_eq_band1_db)',start);f=f[:start]+f[end:];start=f.index('  proc_lpf_hz <');end=f.index('  outL =',start);f=f[:start]+f[end:];s=s.replace(function(s,'apply_posteq_filter'),f)
# Diagnostic isolated EQ: the reference executes the same three bands; HP/LP
# setup still exists but is deliberately not part of the measured processing.
s=s.replace('sf_on=(proc_hpf_hz>22 || proc_lpf_hz<min(19000,srate*0.47) || abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001);','sf_on=abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001;')
(b/'Sample/private-kernel-before.jsfx').write_text(s,encoding='utf-8')
idx=s.index('@sample');prefix=s[:idx];prefix+='@block\nsf_on=abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001;\n'
for i in [4,5,6]:
 for j in range(5):prefix+=f'q_b{i}_{j}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{j}];\n'
state=[f'z{i}_{ch}{j}' for i in [4,5,6] for ch in ['l','r'] for j in [1,2]]+['silent']
lines=[];nexts={};outputs=[]
for ch,inp in [('l','input_l'),('r','input_r')]:
 current=f'({inp}'+('+' if ch=='l' else '-')+'denorm_guard)'
 for i in [4,5,6]:
  en=f'(sf_on & (abs(proc_eq_band{i-3}_db)>0.000001))';y=f'y{i}_{ch}';lines.append(f'{y}=q_b{i}_0*{current}+z{i}_{ch}1;')
  for j,expr in [(1,f'q_b{i}_1*{current}-q_b{i}_3*{y}+z{i}_{ch}2'),(2,f'q_b{i}_2*{current}-q_b{i}_4*{y}')]:nexts[f'z{i}_{ch}{j}']=f'select2({en},z{i}_{ch}{j},q_zap({expr}))'
  nxt=f'x{i}_{ch}';lines.append(f'{nxt}=select2({en},{current},q_zap({y}));');current=nxt
 out=f'out_{ch}';lines.append(f'{out}=select2(sf_on,{inp},q_zap({current}));');outputs.append(out)
lines+=['near=(abs(out_l)+abs(out_r))<DENORM_AUDIO_FLOOR;','next_silent=select2(sf_on,silent,select2(near,0,min(DENORM_ANALYZER_HOLD,silent+1)));','reset=sf_on & near & (silent+1==DENORM_ANALYZER_HOLD);']
vals=[f'select2(reset,{nexts[n]},0)' for n in state[:-1]]+['next_silent']+outputs+['next_silent']
faust='@faust\nq_zap(x)=select2(abs(x)<DENORM_CUT,x,0);\nq_result=(spl0,spl1):(step~('+','.join('_' for _ in state)+')) with {\nstep('+','.join(state+['input_l','input_r'])+')='+','.join(vals)+' with {\n'+'\n'.join(lines)+'\n};\n};\n'
for i,n in enumerate(['q_left','q_right','posteq_filter_silent']):faust+=n+'=q_result:('+','.join('_' if j==len(state)+i else '!' for j in range(len(vals)))+');\n'
faust+='process=q_left,q_right;\n';(b/'Sample/private-kernel-after.jsfx').write_text(prefix+faust,encoding='utf-8')

