"""Preserve original CMD policy and audio arithmetic in a full-block Faust port."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
s=(root/'plugins/Spectral/CMD/src/CrossMixDeclutter.jsfx').read_text()
init='cmd_epoch=0; cmd_phase_seed=breathPhase;\n'
for i in range(24):init+=f'cmd_acc{i}=0; cmd_cut{i}=0; cmd_som{i}=0;\n'
for v in ['total','low','contact','visc','onset']:init+=f'cmd_sum_{v}=0;\n'
s=s.replace('\n@slider\n','\n'+init+'\n@slider\n',1)
pre='cmd_epoch+=1;\n'
for i in range(24):pre+=f'accum[{i}]=cmd_acc{i};\n'
for v in ['total','low','contact','visc','onset']:pre+=f'{v}_accum=cmd_sum_{v};\n'
s=s.replace('\n@block\n','\n@block\n'+pre,1)
a=s.index('\n@sample\n');b=s.index('\n@gfx',a)
imports='\n'
for i in range(24):
 for dest,src in [('b0','b0c'),('b2','b2c'),('a1','a1c'),('a2','a2c'),('fc','fc'),('targetcut','targetCut'),('targetsom','targetSom')]:imports+=f'cmd_{dest}{i}={src}[{i}];\n'
f='''@faust block
cmd_bp(on,b0,b2,a1,a2,x)=x:(step~(_,_,_,_)):(!,!,!,!,_) with {
 step(x1,x2,y1,y2,value)=nx1,nx2,ny1,ny2,out with {
 raw=b0*value+b2*x2-a1*y1-a2*y2;
 nx1=select2(on,x1,value);nx2=select2(on,x2,x1);
 ny1=select2(on,y1,raw);ny2=select2(on,y2,y1);
 out=select2(on,0,raw);
 };
};
cmd_pole(c,lo,hi,on,x)=x:(step~_):(!,_) with {
 step(previous,value)=next,next with {next=select2(on,0,min(hi,max(lo,previous+c*(value-previous))));};
};
cmd_sum(x)=x:(step~(_,_)):(!,!,_) with {
 step(oldEpoch,previous,value)=cmd_epoch,next,next with {next=select2(cmd_epoch!=oldEpoch,previous,0)+value;};
};
cmd_phase(x)=x:(step~(_,_)):(!,!,_) with {
 step(previous,started,value)=next,1,next with {
 raw=select2(started,cmd_phase_seed,previous)+value;
 next=select2(raw>2*pi,raw,raw-2*pi);
 };
};
cmd_follow(c,x)=x:(step~_):(!,_) with { step(previous,value)=next,next with {next=previous+c*(value-previous);}; };
cmd_fast=cmd_follow(fast_env_coeff,max(abs(spl0),abs(spl1)));
cmd_sum_onset=cmd_sum(max(0,cmd_fast-cmd_fast'));
cmd_breath=0.70+0.30*sin(cmd_phase(breath_base_inc*(0.045+0.420*bus_arousal+0.220*bus_thrust)));
'''
for i in range(24):
 f+=f'cmd_on{i}=active_nb>{i};\n'
 for ch,inp in [('l','spl0'),('r','spl1')]:
  args=f'cmd_on{i},cmd_b0{i},cmd_b2{i},cmd_a1{i},cmd_a2{i}'
  f+=f'cmd_band_{ch}{i}=cmd_bp({args},cmd_bp({args},{inp}));\n'
 f+=f'cmd_energy{i}=select2(cmd_on{i},0,.5*(cmd_band_l{i}*cmd_band_l{i}+cmd_band_r{i}*cmd_band_r{i}));\n'
 f+=f'cmd_safe{i}=select2(abs(cmd_energy{i})<1000000000,0,cmd_energy{i});\n'
 f+=f'cmd_acc{i}=cmd_sum(cmd_safe{i});\n'
 f+=f'cmd_cut{i}=cmd_pole(cut_smooth,0,.95,cmd_on{i},cmd_targetcut{i});\ncmd_som{i}=cmd_pole(som_smooth,-.25,.25,cmd_on{i},cmd_targetsom{i});\n'
 f+=f'cmd_gain{i}=-cmd_cut{i}+cmd_som{i}*cmd_breath;\n'
for kind,condition in [('total','1'),('low','cmd_fc{i}<260'),('contact','(cmd_fc{i}>=1100)&(cmd_fc{i}<=8000)'),('visc','(cmd_fc{i}>=320)&(cmd_fc{i}<=6200)')]:
 terms=[f'cmd_safe{i}' if kind=='total' else f'select2({condition.format(i=i)},0,cmd_safe{i})' for i in range(24)]
 f+=f"cmd_sum_{kind}=cmd_sum("+'+'.join(terms)+');\n'
for ch,inp in [('l','spl0'),('r','spl1')]:f+=f'cmd_out_{ch}={inp}+'+'+'.join(f'cmd_gain{i}*cmd_band_{ch}{i}' for i in range(24))+';\n'
f+='''cmd_width=cmd_pole(width_smooth,-.25,.25,1,targetWidth);
cmd_sat=cmd_pole(sat_smooth,0,.35,1,targetSat);
cmd_mid=.5*(cmd_out_l+cmd_out_r);
cmd_side=.5*(cmd_out_l-cmd_out_r)*(1+cmd_width);
cmd_left=cmd_mid+cmd_side;cmd_right=cmd_mid-cmd_side;
cmd_shape(x)=select2(cmd_sat>.00001,x,x*(1+cmd_sat)/(1+cmd_sat*abs(x)));
process=min(8,max(-8,cmd_shape(cmd_left)*out_gain)),min(8,max(-8,cmd_shape(cmd_right)*out_gain));
'''
post='\n@block\n// Publish the final smoothing values to the original UI arrays.\n'
for i in range(24):post+=f'smoothCut[{i}]=cmd_cut{i}; smoothSom[{i}]=cmd_som{i};\n'
s=s[:a]+imports+'\n'+f+post+s[b:]
p=root/'plugins/Spectral/CMD/src/CrossMixDeclutterFaust.jsfx';p.write_text(s,encoding='utf-8',newline='\n')
print('Generated original-design CMD full-block port; original source unchanged')
