from pathlib import Path
b=Path('build/faust-sections/perceptual');s=(b/'kernel-cache.jsfx').read_text(encoding='utf-8');i=s.index('function posteq_topology_shape(');j=s.index('function posteq_char_mode(',i);old=s[i:j];old=old.replace('posteq_topology_shape(','posteq_exact_shape(',1)
lookup='''function posteq_topology_shape(x topo) local(ax index frac base) (
 ax=abs(x);
 topo==2 && ax<=64 ? (
  index=floor(ax*128);frac=ax*128-index;base=x<0 ? za_shape_neg : za_shape_pos;
  index>=8192 ? base[8192] : base[index]+(base[index+1]-base[index])*frac;
 ) : posteq_exact_shape(x,topo);
);
'''
s=s[:i]+old+lookup+s[j:];s=s.replace('DENORM_CUT=','za_shape_pos=2048;za_shape_neg=12288;za_shape_i=0;loop(8193,za_shape_x=za_shape_i/128;za_shape_pos[za_shape_i]=posteq_exact_shape(za_shape_x,2);za_shape_neg[za_shape_i]=posteq_exact_shape(-za_shape_x,2);za_shape_i+=1;);\nDENORM_CUT=',1);(b/'kernel-lut.jsfx').write_text(s,encoding='utf-8')
# Independent filter histories, using the installed FAUST filter primitives.
base=Path('build/faust-sections/sampler-corpus/Sample');s=(base/'privatefull-kernel-before.jsfx').read_text(encoding='utf-8');p=s[:s.index('@sample')];p+='@block\n'
for i in range(13):
 for j in range(5):p+=f'q_b{i}_{j}=posteq_bq_base[{i}*POSTEQ_BQ_STRIDE+{j}];\n'
f='''@faust
import("stdfaust.lib");
q_bq(b0,b1,b2,a1,a2,on)=fi.tf2(select2(on,1,b0),select2(on,0,b1),select2(on,0,b2),select2(on,0,a1),select2(on,0,a2));
q_hp=proc_hpf_hz>22;q_lp=proc_lpf_hz<min(19000,srate*0.47);
q_hp1(x)=x-(x:*(proc_hpf_coeff):fi.pole(1-proc_hpf_coeff));
q_lp1=*(proc_lpf_coeff):fi.pole(1-proc_lpf_coeff);
'''
# Explicit branches select audio, not reset callbacks. The continuously advancing
# internal histories differ on bypass/re-enable; this is an experimental design.
for i in range(13):
 if i<4:on=f'q_hp & (proc_hpf_slope_idx>{i})'
 elif i==11:on='q_hp & (posteq_hpf_res_db>0.0001)'
 elif i in [4,5,6]:on=f'abs(proc_eq_band{i-3}_db)>0.000001'
 elif i==12:on='q_lp & (posteq_lpf_res_db>0.0001)'
 else:on=f'q_lp & (proc_lpf_slope_idx>{i-7})'
 f+=f'q_f{i}=q_bq('+','.join(f'q_b{i}_{j}' for j in range(5))+f',({on}));\n'
f+='q_first(x)=select2(q_hp & (proc_hpf_slope_idx<=0),x,x:q_hp1);\nq_last(x)=select2(q_lp & (proc_lpf_slope_idx<=0),x,x:q_lp1);\nq_chain=q_first:'+':'.join('q_f'+str(i) for i in list(range(4))+[11,4,5,6,12]+list(range(7,11)))+':q_last;\nprocess=q_chain,q_chain;\n';(b/'kernel-faust-filters.jsfx').write_text(p+f,encoding='utf-8')
