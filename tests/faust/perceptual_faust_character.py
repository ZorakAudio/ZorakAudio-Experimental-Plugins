from pathlib import Path
b=Path('build/faust-sections/perceptual');s=(b/'kernel-cache.jsfx').read_text(encoding='utf-8');s=s[:s.index('@sample')];s=s.replace('@init\n','@init\nposteq_char_wet=0;\n',1);s+='@block\n'
names=['hpf','b1','b2','b3','lpf']
for i,n in enumerate(names):
 s+=f'q_mode{i}=posteq_{n}_char;q_drive{i}=posteq_{n}_drive;\n'
 for j in range(5):s+=f'q_c{i}_{j}=posteq_char_bq_base[{i}*9+{j}];\n'
f='''@faust
import("stdfaust.lib");
q_zap(x)=select2(abs(x)<DENORM_CUT,x,0);
q_bq(b0,b1,b2,a1,a2,on,x)=x:(step~(_,_)):(!,!,_) with {
 step(z1,z2,value)=next1,next2,y with {
  y=b0*value+z1;
  next1=select2(on,z1,q_zap(b1*value-a1*y+z2));
  next2=select2(on,z2,q_zap(b2*value-a2*y));
 };
};
q_trans(x)=y/(1+abs(y)*0.62) with {y=x+0.18*x*x-0.10*x*x*x;};
q_diode(x)=x/sqrt(1+x*x*1.45)+0.055*(x*x/(1+abs(x)*1.8)-0.18*x);
q_shape(x,mode)=select2(mode==1,q_diode(x),q_trans(x));
'''
for i in range(5):
 f+=f'q_amp{i}=exp(q_drive{i}*0.11512925464970229);q_inv{i}=1/max(q_amp{i},0.000001);q_wet{i}=min(1,max(0,(q_drive{i}/24)^0.72));\n'
 f+=f'q_f{i}=q_bq('+','.join(f'q_c{i}_{j}' for j in range(5))+f',(q_mode{i}>0)&(q_drive{i}>0.0001));\n'
 for ch in ['l','r']:
  f+=f'q_bp{i}{ch}=spl{0 if ch=="l" else 1}:q_f{i};\nq_r{i}{ch}=select2((q_mode{i}>0)&(q_drive{i}>0.0001),0,(q_shape(q_bp{i}{ch}*q_amp{i},q_mode{i})*q_inv{i}-q_bp{i}{ch})*q_wet{i}'+(f'*select2(q_mode{i}==2,0.62,0.78)' if i>=3 else '')+');\n'
for ch in ['l','r']:
 f+=f'q_out{ch}=spl{0 if ch=="l" else 1}+'+'+'.join(f'q_r{i}{ch}' for i in range(5))+';\n'
f+='posteq_char_wet='+ 'max('*4+','.join(f'max(abs(q_r{i}l),abs(q_r{i}r))*8)' if i else f'max(abs(q_r{i}l),abs(q_r{i}r))*8' for i in range(5))+';\n'
# Use explicit nesting rather than the string above's ambiguous commas.
f=f[:f.index('posteq_char_wet=')];meter='max(abs(q_r4l),abs(q_r4r))*8'
for i in reversed(range(4)):meter=f'max(max(abs(q_r{i}l),abs(q_r{i}r))*8,{meter})'
f+='posteq_char_wet='+meter+';\nprocess=q_outl,q_outr;\n@block\nza_character_block_barrier=0;\n@sample\noutL=spl0;outR=spl1;apply_output_harmonics();spl0=outL;spl1=outR;\n'
(b/'kernel-faust-character.jsfx').write_text(s+f,encoding='utf-8')
