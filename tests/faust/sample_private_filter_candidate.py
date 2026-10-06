"""Whole Sample post-EQ filter kernel with private FAUST histories.
Diagnostic block kernel, not a full Sample plugin migration.
"""
from pathlib import Path
b=Path('build/faust-sections/sampler-corpus/Sample');s=(b/'kernel-before.jsfx').read_text(encoding='utf-8');p=s[:s.index('@sample')]
p+='@block\nsf_on=(proc_hpf_hz>22 || proc_lpf_hz<min(19000,srate*0.47) || abs(proc_eq_band1_db)>0.0001 || abs(proc_eq_band2_db)>0.0001 || abs(proc_eq_band3_db)>0.0001);\n'
for i in range(13):
 for j in range(5):p+=f'q_b{i}_{j}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{j}];\n'
states=[f'z{i}_{ch}{j}' for i in range(13) for ch in ['l','r'] for j in [1,2]]+[f'{typ}_{ch}' for typ in ['hp','lp'] for ch in ['l','r']]+['silent'];lines=[];nexts={};outputs=[]
for ch,inp in [('l','input_l'),('r','input_r')]:
 current=f'({inp}'+('+' if ch=='l' else '-')+'denorm_guard)'
 en='sf_on & (proc_hpf_hz>22) & (proc_hpf_slope_idx<=0)';n=f'hp_{ch}';nexts[n]=f'select2({en},{n},q_zap({n}+({current}-{n})*proc_hpf_coeff))';lines.append(f'onehp_{ch}=select2({en},{current},{current}-({nexts[n]}));');current=f'onehp_{ch}'
 order=list(range(4))+[11,4,5,6,12]+list(range(7,11))
 for i in order:
  if i<4:active=f'(proc_hpf_hz>22)&(proc_hpf_slope_idx>{i})'
  elif i==11:active='(proc_hpf_hz>22)&(posteq_hpf_res_db>0.0001)'
  elif i in [4,5,6]:active=f'abs(proc_eq_band{i-3}_db)>0.000001'
  elif i==12:active='(proc_lpf_hz<min(19000,srate*0.47))&(posteq_lpf_res_db>0.0001)'
  else:active=f'(proc_lpf_hz<min(19000,srate*0.47))&(proc_lpf_slope_idx>{i-7})'
  en=f'sf_on & ({active})';y=f'y{i}_{ch}';lines.append(f'{y}=q_b{i}_0*{current}+z{i}_{ch}1;')
  for j,expr in [(1,f'q_b{i}_1*{current}-q_b{i}_3*{y}+z{i}_{ch}2'),(2,f'q_b{i}_2*{current}-q_b{i}_4*{y}')]:nexts[f'z{i}_{ch}{j}']=f'select2({en},z{i}_{ch}{j},q_zap({expr}))'
  nxt=f'x{i}_{ch}';lines.append(f'{nxt}=select2({en},{current},q_zap({y}));');current=nxt
 en='sf_on & (proc_lpf_hz<min(19000,srate*0.47)) & (proc_lpf_slope_idx<=0)';n=f'lp_{ch}';nexts[n]=f'select2({en},{n},q_zap({n}+({current}-{n})*proc_lpf_coeff))';lines.append(f'onelp_{ch}=select2({en},{current},{nexts[n]});');out=f'out_{ch}';lines.append(f'{out}=select2(sf_on,{inp},q_zap(onelp_{ch}));');outputs.append(out)
lines+=['near=(abs(out_l)+abs(out_r))<DENORM_AUDIO_FLOOR;','next_silent=select2(sf_on,silent,select2(near,0,min(DENORM_ANALYZER_HOLD,silent+1)));','reset=sf_on & near & (silent+1==DENORM_ANALYZER_HOLD);']
vals=[f'select2(reset,{nexts[n]},0)' for n in states[:-1]]+['next_silent']+outputs
faust='@faust\nq_zap(x)=select2(abs(x)<DENORM_CUT,x,0);\nq_result=(spl0,spl1):(step~('+','.join('_' for _ in states)+')) with {\nstep('+','.join(states+['input_l','input_r'])+')='+','.join(vals)+' with {\n'+'\n'.join(lines)+'\n};\n};\n'
for i,n in enumerate(['q_left','q_right']):faust+=n+'=q_result:('+','.join('_' if j==len(states)+i else '!' for j in range(len(vals)))+');\n'
faust+='process=q_left,q_right;\n';(b/'privatefull-kernel-before.jsfx').write_text(s,encoding='utf-8');(b/'privatefull-kernel-after.jsfx').write_text(p+faust,encoding='utf-8')
