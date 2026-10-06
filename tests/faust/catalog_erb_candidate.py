"""ERBTilt whole-bank candidate; retained only after null/performance checks."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'build/catalog-faust-audit/ERBTilt'
s=(base/'baseline.jsfx').read_text(encoding='utf-8')
imports=[];exports=[];init=[];dsp=[]
for i in range(16):
    for field,array in [('gain','gBand'),('freq','fcBand')]+([('a','lpA')] if i<15 else []):
        name=f'et_{field}{i}';init.append(name+'=0;');imports.append(f'{name}={array}[{i}];')
    for field,array in [('pre','preE'),('post','postE'),('fast','envF'),('slow','envS')]:
        name=f'et_{field}{i}';init.append(name+'=0;');exports.append(f'{array}[{i}]={name};')
for ch in ['l','r']:
    for i in range(15):dsp.append(f'et_lp_{ch}{i}=et_lp(et_a{i},spl{0 if ch=="l" else 1});')
    for i in range(16):
        band=(f'et_lp_{ch}0' if i==0 else f'et_lp_{ch}{i}-et_lp_{ch}{i-1}' if i<15 else f'spl{0 if ch=="l" else 1}-et_lp_{ch}14')
        dsp.append(f'et_band_{ch}{i}={band};')
for i in range(16):
    dsp+= [f'et_pre{i}=et_pole(energy_smooth,0.5*(et_band_l{i}*et_band_l{i}+et_band_r{i}*et_band_r{i}));',
           f'et_y_l{i}=et_band_l{i}*et_gain{i};',f'et_y_r{i}=et_band_r{i}*et_gain{i};',
           f'et_amp{i}=0.5*(abs(et_y_l{i})+abs(et_y_r{i}));',
           f'et_fast{i}=et_guarded(rough_fast,et_freq{i}>rough_start_hz,et_amp{i});',
           f'et_slow{i}=et_guarded(rough_slow,et_freq{i}>rough_start_hz,et_amp{i});',
           f'et_sat{i}=guard*max(0,(et_fast{i}-et_slow{i})/(et_slow{i}+0.000001))*6;',
           f'et_out_l{i}=select2(et_freq{i}>rough_start_hz,et_y_l{i},et_y_l{i}/(1+et_sat{i}*abs(et_y_l{i})));',
           f'et_out_r{i}=select2(et_freq{i}>rough_start_hz,et_y_r{i},et_y_r{i}/(1+et_sat{i}*abs(et_y_r{i})));',
           f'et_post{i}=et_pole(energy_smooth,0.5*(et_out_l{i}*et_out_l{i}+et_out_r{i}*et_out_r{i}));']
def ordered_sum(ch):
    result='0.0'
    for i in range(16):result='('+result+f'+et_out_{ch}{i})'
    return result
faust='''@faust
// Whole fixed ERB bank; EEL retains coefficient setup, compensation and UI.
et_lp(a,x)=x:(step~_):(!,_) with {
    step(previous,value)=next,next with { next=(1-a)*value+a*previous; };
};
et_pole(c,x)=x:(step~_):(!,_) with {
    step(previous,value)=next,next with { next=previous+(value-previous)*c; };
};
et_guarded(c,on,x)=x:(step~_):(!,_) with {
    step(previous,value)=next,next with { next=previous+select2(on,0,(value-previous)*c); };
};
'''+ '\n'.join(dsp)+'\nprocess='+ordered_sum('l')+'*global_gain,'+ordered_sum('r')+'*global_gain;\n'
faust+='\n@block\n// Publish current block detector values for the existing display.\n'+'\n'.join(exports)+'\n'
s=s.replace('@init\n','@init\n'+'\n'.join(init)+'\n',1)
s=s.replace('@block\n','@block\n// Publish previous block energy/follower state before compensation.\n'+'\n'.join(exports)+'\n',1)
# Fetch controls after existing coefficient rebuild and compensation.
s=re.sub(r'(?ms)^@sample[^\n]*\n.*?(?=^@|\Z)',lambda _: '\n'.join(imports)+'\n\n'+faust+'\n',s)
(base/'candidate.jsfx').write_text(s,encoding='utf-8')
print('ERBTilt candidate generated')
