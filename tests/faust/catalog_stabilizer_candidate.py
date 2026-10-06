from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'build/catalog-faust-audit/SpectralStabilizer'
s=(base/'baseline.jsfx').read_text(encoding='utf-8');init=[];imports=[];exports=[];dsp=[]
for i in range(12):
 for j in range(5):
  name=f'st_b{i}_{j}';init.append(name+'=0;');imports.append(f'{name}=mem[ofs_bq+{i*5+j}];')
 name=f'st_target{i}';init.append(name+'=1;');imports.append(f'{name}=mem[ofs_gt+{i}];')
 for field,array in [('env','envlin'),('log','envlog'),('d1','d1'),('d2','d2'),('motion','mot'),('gain','gs')]:
  name=f'st_{field}{i}';init.append(name+('=1;' if field=='gain' else '=0;'));exports.append(f'mem[ofs_{array}+{i}]={name};')
 args=','.join(f'st_b{i}_{j}' for j in range(5))
 dsp += [f'st_left{i}=st_biquad({args},spl0);',f'st_right{i}=st_biquad({args},spl1);',
  f'st_env{i}=st_smooth(env_a,0.5*(st_left{i}*st_left{i}+st_right{i}*st_right{i}));',
  f'st_log{i}=log(st_env{i}+0.000000000001);',f"st_d1{i}=st_log{i}-st_log{i}';",f"st_d2{i}=st_d1{i}-st_d1{i}';",
  f'st_minst{i}=min(1,max(0,abs(st_d1{i})*6+abs(st_d2{i})*3+(st_d1{i}>onset_thr)*0.75));',
  f'st_motion{i}=st_smooth(mot_a,st_minst{i});',f'st_gain{i}=st_gain_smooth(st_target{i});',
  f'st_apply{i}=(1+action_amt*(st_gain{i}-1))-1;',
  f'st_add_l{i}=st_apply{i}*st_left{i};',f'st_add_r{i}=st_apply{i}*st_right{i};']
def ordered_sum(ch):
 out='spl'+str(0 if ch=='l' else 1)
 for i in range(12):out='('+out+f'+st_add_{ch}{i})'
 return out
faust='''@faust
import("stdfaust.lib");
// Fixed bank, exact transposed biquad and EMA operation order.
st_biquad(b0,b1,b2,a1,a2,x)=x:(step~(_,_)):(!,!,_) with {
 step(z1,z2,value)=b1*value-a1*y+z2,b2*value-a2*y,y with {y=b0*value+z1;};
};
st_smooth(a,x)=x:(step~_):(!,_) with {
 step(previous,value)=next,next with {next=a*previous+(1-a)*value;};
};
st_gain_smooth(x)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {
  previous=select2((1')==0,old,1);
  a=select2(value<previous,rel_a,atk_a);
  next=a*previous+(1-a)*value;
 };
};
'''+ '\n'.join(dsp)+'\nprocess='+ordered_sum('l')+','+ordered_sum('r')+';\n'
faust+='\n@block\n// Publish current block detector values for the existing display.\n'+'\n'.join(exports)+'\n'
s=s.replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
s=s.replace('@block\n','@block\n// Previous block detector state for DoSG control and display.\n'+'\n'.join(exports)+'\n',1)
s=re.sub(r'(?ms)^@sample[^\n]*\n.*?(?=^@|\Z)',lambda _: '\n'.join(imports)+'\n\n'+faust+'\n',s)
(base/'candidate.jsfx').write_text(s,encoding='utf-8');print('Stabilizer candidate generated')
