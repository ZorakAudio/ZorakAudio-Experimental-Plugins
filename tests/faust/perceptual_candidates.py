from pathlib import Path
import sys,re
R=Path.cwd();sys.path.insert(0,str(R/'tests/faust'));from sampler_corpus_candidates import function
B=R/'build/faust-sections/perceptual';B.mkdir(exist_ok=True);s=(R/'tests/faust/fixtures/sampler-corpus/Sample.jsfx').read_text(encoding='utf-8');(B/'Sample-before.jsfx').write_text(s,encoding='utf-8')
cache='// Cache control-only character math. Key on the live value so GUI and slider\n// changes take effect immediately, including edits within a processing buffer.\nfunction posteq_cached_character(slot drive) (\n'
for i in range(5):
 cache+=('  ' if i==0 else ' : ')+f'slot=={i} ? (\n  drive!=za_char_drive{i} ? (za_char_drive{i}=drive;za_char_amp{i}=db2amp(drive);za_char_inv{i}=1/max(za_char_amp{i},0.000001);za_char_wet{i}=clamp01((drive/24)^0.72););\n  posteq_cached_amp=za_char_amp{i};posteq_cached_inv=za_char_inv{i};posteq_cached_wet=za_char_wet{i};\n )'
cache+=';\n);\n'
init='\n'.join(f'za_char_drive{i}=-1;' for i in range(5))+'\nza_harm_amount=-1;\n'
t=s.replace('function posteq_apply_character_band(',cache+'\nfunction posteq_apply_character_band(',1)
old='    driveamp = db2amp(drive);\n    invdrive = 1 / max(driveamp, 0.000001);\n    wet = clamp01((drive / 24) ^ 0.72);';new='    posteq_cached_character(slot,drive);\n    driveamp=posteq_cached_amp;invdrive=posteq_cached_inv;wet=posteq_cached_wet;';assert old in t;t=t.replace(old,new,1)
a=t.index('    h2 = 0.30 * (a ^ 0.92);');z=t.index('\n    prevL = 0;',a)
t=t[:a]+'''    // These powers depend only on Amount, not band, audio or sample rate.
    a != za_harm_amount ? (
      za_harm_amount=a;
      za_harm_h2=0.30*(a^0.92);za_harm_h3=0.22*(a^1.14);
      za_harm_h4=0.13*harm_smoothstep(0.30,0.88,a);
      za_harm_h5=0.12*harm_smoothstep(0.44,0.96,a);
      za_harm_h7=0.075*harm_smoothstep(0.70,1.00,a);
      za_harm_mix=0.22+0.78*(a^1.22);za_harm_wet=clamp01(0.20+0.88*(a^1.18));
    );
    h2=za_harm_h2;h3=za_harm_h3;h4=za_harm_h4;h5=za_harm_h5;h7=za_harm_h7;
''' + t[z:]
t=t.replace('env * (0.22 + 0.78 * (a ^ 1.22)) * trans_guard * bright_guard','env * za_harm_mix * trans_guard * bright_guard',1).replace('wet = clamp01(0.20 + 0.88 * (a ^ 1.18));','wet = za_harm_wet;',1).replace('@init\n','@init\n'+init,1);(B/'Sample-cache.jsfx').write_text(t,encoding='utf-8')
helpers=['clamp','clamp01','denorm_zap','db2amp','posteq_char_bq_set_band','posteq_char_bq_process_l','posteq_char_bq_process_r','posteq_topology_shape','posteq_char_mode','posteq_char_drive','posteq_apply_character_band','apply_posteq_character','harm_xover_freq','harm_update_coeffs','harm_smoothstep','harm_band_weight','harm_hi_weight','harm_ultra_weight','harm_dc_l','harm_dc_r','harm_safe','apply_output_harmonics','reset_output_harmonics_buffers']
for label,text in [('before',s),('cache',t)]:
 kernel='desc:Sample nonlinear character and harmonic qualification\nslider1:1<0,2,1>Character\nslider2:12<0,24,0.5>Drive\nslider3:0.5<0,1,0.01>Harmonics\n@init\n'+(init+cache if label=='cache' else '')+'\n'.join(function(text,n) for n in helpers)
 kernel+='\nDENORM_CUT=0.000000000000000001;denorm_guard=0;POSTEQ_CHAR_BQ_STRIDE=9;posteq_char_bq_base=1;HARM_BANDS=8;HARM_XOVERS=7;\n'
 for i,name in enumerate(['harm_lpa_l_base','harm_lpb_l_base','harm_lpa_r_base','harm_lpb_r_base','harm_band_l_base','harm_band_r_base','harm_env_base','harm_dcx_l_base','harm_dcy_l_base','harm_dcx_r_base','harm_dcy_r_base','harm_coeff_base']):kernel+=f'{name}={100+i*16};\n'
 kernel+='harm_env_att_coeff=1-exp(-6.90775527898214/max(1,floor(0.005*srate)));harm_env_rel_coeff=1-exp(-6.90775527898214/max(1,floor(0.090*srate)));harm_update_coeffs();\n'
 kernel+='posteq_char_bq_set_band(0,120,0.7);posteq_char_bq_set_band(1,300,0.7);posteq_char_bq_set_band(2,1600,1.1);posteq_char_bq_set_band(3,6000,0.8);posteq_char_bq_set_band(4,12000,0.8);\n'
 kernel+='@slider\nposteq_hpf_char=slider1;posteq_b1_char=slider1;posteq_b2_char=slider1;posteq_b3_char=slider1;posteq_lpf_char=slider1;posteq_hpf_drive=slider2;posteq_b1_drive=slider2;posteq_b2_drive=slider2;posteq_b3_drive=slider2;posteq_lpf_drive=slider2;output_harmonics_amt=slider3;\n@sample\noutL=spl0;outR=spl1;apply_posteq_character();apply_output_harmonics();spl0=outL;spl1=outR;\n'
 (B/('kernel-'+label+'.jsfx')).write_text(kernel,encoding='utf-8')
print('Generated cache candidate and authentic nonlinear/harmonic kernels')
