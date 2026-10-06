"""Generate a guarded full-block character candidate, retaining the EEL fallback."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/faust-sections/sample-character'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT / 'plugins/Spectral/Sample/src/Sample.jsfx')
parser.add_argument('--output', type=Path, default=BASE / 'Sample-Faust-Character.jsfx')
args = parser.parse_args()
source = args.source.read_text(encoding='utf-8')

def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError('Sample integration boundary changed; audit before regenerating: ' + old)
    source = source.replace(old, new, 1)

replace_once('tags:instrument sampler ui', 'tags:instrument sampler ui\noptions:za_faust_quantum=256')

# Synchronize EEL histories on handoff and at host boundaries. An epoch makes
# both ordinary EQ clears and silent-tail clears visible at the exact frame.
for name in ('reset_processing_buffers', 'posteq_zero_filter_states'):
    start = source.index('function ' + name + '(')
    body = source.index('\n(', start) + 2
    source = source[:body] + '\n  za_character_epoch += 1;' + source[body:]

replace_once('  posteq_char_active ? apply_posteq_character();',
             '  posteq_char_active && !za_character_block ? apply_posteq_character();')

# Everything after character must be a linear, constant output trim plus dry
# passthrough. Otherwise retain the original chronological EEL sample path.
prepare = '''
@block
// Only commute character residual past the final constant trim and dry sum.
za_character_previous=za_character_block;
za_character_block = posteq_char_active && !posteq_analyzer_active && posteq_solo_target==0
  && posteq_solo_mix<=0.0001 && !v71_contour_active && !sc_running
  && !dc_latency_active && v71_total_delay==0
  && dc_switch_gain==1 && dc_switch_target==1 && dc_output_warm_left<=0
  && (denorm_silence_samps+min(samplesblock,256)<DENORM_SILENCE_RESET_SAMPS
      || denorm_silence_samps>=DENORM_SILENCE_RESET_SAMPS);
za_character_seed_now=!za_character_previous && za_character_block;
za_character_epoch_start=za_character_epoch;
za_character_trim=out_trim;
za_character_sample_start=block_sample_index;
za_character_silence_start=denorm_silence_samps;
'''
names = ['hpf', 'b1', 'b2', 'b3', 'lpf']
controls = ''
for i, name in enumerate(names):
    controls += f'za_mode{i}=posteq_{name}_char; za_drive{i}=posteq_{name}_drive;\n'
    for j in range(5):
        controls += f'za_c{i}_{j}=posteq_char_bq_base[{i}*9+{j}];\n'
prepare += 'za_character_previous && !za_character_block && za_character_epoch==za_character_epoch_completed ? (\n'
for i in range(5):
    for ch, offset in [('l', 5), ('r', 7)]:
        for k in range(2):
            prepare += f'posteq_char_bq_base[{i}*9+{offset+k}]=za_result{i}{ch}{k};\n'
prepare += ');\nza_character_seed_now ? (\n'
for i in range(5):
    for ch, offset in [('l', 5), ('r', 7)]:
        for k in range(2):
            prepare += f'za_seed{i}{ch}{k}=posteq_char_bq_base[{i}*9+{offset+k}];\n'
prepare += ');\n'
replace_once('@sample\n', controls + prepare + '\n@sample\n')
begin = source.index('(!active_voice_seen && denorm_near_silence(outL, outR)) ? (')
end = source.index('\noutL = denorm_zap(outL);', begin)
source = source[:begin] + '!za_character_block ? (\n' + source[begin:end] + '\n);\n' + source[end:]
# The eligible path carries raw sampler audio through the existing stereo ports.
# FAUST performs character, then the original trim and untouched dry sum.
begin = source.index('dc_latency_active ? apply_clean_decrust();', source.index('@sample'))
end = source.index('// Publish pending source-derived work', begin)
source = source[:begin] + '!za_character_block ? (\n' + source[begin:end] + '\n) : (spl0=outL;spl1=outR;);\n' + source[end:]


faust = '''
@block
// The prefix is complete. Its explicit sample streams were captured by the
// runtime; this boundary does not add a callback or another block of latency.
za_character_boundary=0;
@faust block when za_character_block
import("stdfaust.lib");
qc_first=(block_sample_index==za_character_sample_start+1)&za_character_seed_now;
qc_zap(x)=select2(abs(x)<DENORM_CUT,x,0);
qc_trans(x)=y/(1+abs(y)*0.62) with {y=x+0.18*x*x-0.10*x*x*x;};
qc_diode(x)=x/sqrt(1+x*x*1.45)+0.055*(x*x/(1+abs(x)*1.8)-0.18*x);
qc_shape(x,mode)=select2(mode==1,qc_diode(x),qc_trans(x));
qc_state(b0,b1,b2,a1,a2,on,seed1,seed2,x)=
 (x,za_character_epoch,qc_first):(step~(_,_,_)) with {
 step(z1,z2,previous,value,epoch,first)=next1,next2,epoch,y with {
  changed=select2(first,epoch!=previous,epoch!=za_character_epoch_start);
  old1=select2(changed,select2(first,z1,seed1),0);
  old2=select2(changed,select2(first,z2,seed2),0);
  y=b0*value+old1;
  next1=select2(on,old1,qc_zap(b1*value-a1*y+old2));
  next2=select2(on,old2,qc_zap(b2*value-a2*y));
 };
};
'''
for i in range(5):
    faust += f'qc_on{i}=za_character_block&(za_mode{i}>0)&(za_drive{i}>0.0001);\n'
    faust += f'qc_amp{i}=exp(za_drive{i}*0.11512925464970229);qc_inv{i}=1/max(qc_amp{i},0.000001);qc_wet{i}=min(1,max(0,(za_drive{i}/24)^0.72));\n'
    for ch, sign in [('l', '+'), ('r', '-')]:
        coefficients = ','.join(f'za_c{i}_{j}' for j in range(5))
        faust += f'qc_s{i}{ch}=qc_state({coefficients},qc_on{i},za_seed{i}{ch}0,za_seed{i}{ch}1,spl{0 if ch=="l" else 1}{sign}denorm_guard);\n'
        faust += f'qc_bp{i}{ch}=qc_s{i}{ch}:(!,!,!,_);\n'
        faust += f'za_result{i}{ch}0=qc_s{i}{ch}:(_,!,!,!);za_result{i}{ch}1=qc_s{i}{ch}:(!,_,!,!);\n'
        faust += f'qc_raw{i}{ch}=select2(qc_on{i},0,(qc_shape(qc_bp{i}{ch}*qc_amp{i},za_mode{i})*qc_inv{i}-qc_bp{i}{ch})*qc_wet{i});\n'
        air = f'*select2(za_mode{i}==2,0.62,0.78)' if i >= 3 else ''
        faust += f'qc_res{i}{ch}=qc_zap(qc_raw{i}{ch}{air});\n'
for ch in ['l', 'r']:
    # Preserve the original left-to-right additions, followed by denormal zap.
    col = f'spl{0 if ch=="l" else 1}'
    for i in range(5):
        col = f'({col}+qc_res{i}{ch})'
    faust += f'qc_col{ch}=qc_zap({col});\n'
    faust += f'qc_out{ch}=dry_in_{ch}+qc_col{ch}*za_character_trim;\n'
meter = 'max(abs(qc_raw4l),abs(qc_raw4r))*8'
for i in reversed(range(4)):
    meter = f'max(max(abs(qc_raw{i}l),abs(qc_raw{i}r))*8,{meter})'
faust += f'za_character_meter={meter};\nprocess=qc_outl,qc_outr;\n@block\n'
faust = faust.replace('process=qc_outl,qc_outr;', '''za_character_out_l=qc_coll;za_character_out_r=qc_colr;
qc_quiet=(active_voice_seen==0)&((abs(qc_coll)+abs(qc_colr))<DENORM_AUDIO_FLOOR);
za_character_silence=(qc_quiet,qc_first):(step~_) with {
 step(previous,quiet,first)=select2(quiet,0,min(DENORM_SILENCE_RESET_SAMPS,select2(first,previous,za_character_silence_start)+1));
};
process=qc_outl,qc_outr;''')
faust += 'za_character_block && (block_sample_index>za_character_sample_start) ? (\n'
faust += ' block_sample_index==samplesblock ? (\n'
for i in range(5):
    for ch, offset in [('l', 5), ('r', 7)]:
        for k in range(2):
            faust += f' posteq_char_bq_base[{i}*9+{offset+k}]=za_result{i}{ch}{k};\n'
faust += ' );\n posteq_char_active ? (posteq_char_wet=za_character_meter;outL=za_character_out_l;outR=za_character_out_r;);\n denorm_silence_samps=za_character_silence;\n);\n(block_sample_index>za_character_sample_start) ? za_character_epoch_completed=za_character_epoch;\n'
replace_once('@gfx 1380 760', faust + '\n@gfx 1380 760')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(source, encoding='utf-8')
