"""Sample three-band EQ integration, retaining HP/LP and reset ownership in EEL."""
from pathlib import Path
import sys,re,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from sampler_corpus_candidates import BASE,source,function
s=source('Sample');folder=BASE/'Sample'
old=function(s,'apply_posteq_filter');begin=old.index('(\n',old.index('local('))+2;body=old[begin:old.rfind(');')]
a=body.index('  abs(proc_eq_band1_db)');b=body.index('  proc_lpf_hz <',a);before=body[:a];after=body[b:]
rename=lambda x:re.sub(r'\bxL\b','sf_xL',re.sub(r'\bxR\b','sf_xR',re.sub(r'\bidx\b','sf_idx',re.sub(r'\bcount\b','sf_count',x))))
pre=['sf_on=(proc_hpf_hz>22 || proc_lpf_hz<min(19000,srate*0.47) || abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001);','sf_on ? (',rename(before),');sf_input_l=sf_xL;sf_input_r=sf_xR;'];post=[];init=[];dsp=['sf_zap(x)=select2(abs(x)<DENORM_CUT,x,0);']
for i in [4,5,6]:
 for j in range(5):
  name=f'sf_b{i}_{j}';init.append(name+'=0;');pre.append(f'{name}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{j}];')
 for ch,offset in [('l',5),('r',7)]:
  for z in [1,2]:
   prev=f'sf_prev_{i}_{ch}{z}';new=f'sf_next_{i}_{ch}{z}';init.extend([prev+'=0;',new+'=0;']);pre.append(f'{prev}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{offset+z-1}];');post.append(f'posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{offset+z-1}]={new};')
 dsp.append(f'sf_en{i}=sf_on & (abs(proc_eq_band{i-3}_db)>0.000001);')
for ch in ['l','r']:
 current=f'sf_input_{ch}'
 for i in [4,5,6]:
  dsp.append(f'sf_y{i}_{ch}=sf_b{i}_0*{current}+sf_prev_{i}_{ch}1;')
  dsp.append(f'sf_next_{i}_{ch}1=select2(sf_en{i},sf_prev_{i}_{ch}1,sf_zap(sf_b{i}_1*{current}-sf_b{i}_3*sf_y{i}_{ch}+sf_prev_{i}_{ch}2));')
  dsp.append(f'sf_next_{i}_{ch}2=select2(sf_en{i},sf_prev_{i}_{ch}2,sf_zap(sf_b{i}_2*{current}-sf_b{i}_4*sf_y{i}_{ch}));')
  nxt=f'sf_pass{i}_{ch}';dsp.append(f'{nxt}=select2(sf_en{i},{current},sf_zap(sf_y{i}_{ch}));');current=nxt
 dsp.append(f'sf_result_{ch}={current};')
replacement='\n'.join(pre)+'\n@faust\n'+'\n'.join(dsp)+'\nprocess=spl0,spl1;\n@sample\nsf_on ? (\n'+'\n'.join(post)+'\nsf_xL=sf_result_l;sf_xR=sf_result_r;\n'+rename(after)+'\n);\n'
full=s.replace('apply_processing_strip();',replacement+'posteq_char_active ? apply_posteq_character();\n(posteq_solo_target>0 || posteq_solo_mix>0.0001) ? apply_posteq_solo();',1).replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
kernel=(folder/'kernel-before.jsfx').read_text();needle='sf_input_l=outL;sf_input_r=outR;sf_on=(proc_hpf_hz>22 || proc_lpf_hz<min(19000,srate*0.47) || abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001);\nsf_on ? apply_posteq_filter();'
assert needle in kernel
new=kernel.replace(needle,replacement).replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
for filename in ['candidate.jsfx','kernel-after.jsfx']:
 if (folder/filename).exists():shutil.copy2(folder/filename,folder/('attempt13-'+filename))
(folder/'candidate.jsfx').write_text(full,encoding='utf-8');(folder/'kernel-after.jsfx').write_text(new,encoding='utf-8');print('Sample three-EQ-band candidate generated')
