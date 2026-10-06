"""Generate panner variants; keep the production optimization and frozen experimental shell separate."""
from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'plugins/Spatialization/3DPanner/src/3DPanner.jsfx'
original=base.read_text(encoding='utf-8')
def section(text,name):
 match=re.search(r'(?m)^@'+name+r'\b',text)
 if not match:raise ValueError('Missing section '+name)
 return match.start()
def once(s,old,new):
 if s.count(old)!=1:raise ValueError('Integration boundary changed: '+old[:100])
 return s.replace(old,new,1)
# Preserve arithmetic and update order; give LLVM constant filter offsets.
if 'i=0;loop(5,st[5+i]' in original:
 fast=once(original,'  p7_filter_count>0 ? (i=0;loop(5,st[5+i]+=(st[i]-st[5+i])*p7_coeff_k;i+=1;););',
  '  p7_filter_count>0 ? (\n'+''.join(f'    st[{5+i}]+=(st[{i}]-st[{5+i}])*p7_coeff_k;\n' for i in range(5))+'  );')
 fast=once(fast,'  b=0;loop(6,y=p7_filter(st+32+b*12,y);b+=1;);', ''.join(f'  y=p7_filter(st+{32+b*12},y);\n' for b in range(6)))
 start=fast.index('  p7_actor=0;\n  loop(2,',section(fast,'sample'))
 end=fast.index('  p7_out_l=p7_direct_l+p7_er_l;',start)
 render=''
 for actor in range(2):
  for kind in range(7):
   offset=(actor*14+kind*2)*128
   render+=f'  p7_yl=p7_path_tick(p7_states+{offset},p7_buf_{"l" if actor==0 else "r"},p7_wpos);\n'
   render+=f'  p7_yr=p7_path_tick(p7_states+{offset+128},p7_buf_{"l" if actor==0 else "r"},p7_wpos);\n'
   render+=f'  p7_{"direct" if kind==0 else "er"}_l+=p7_yl;p7_{"direct" if kind==0 else "er"}_r+=p7_yr;\n'
 fast=fast[:start]+render+fast[end:]
else:
 # The production source now already includes the qualified Fast specialization.
 if 'p7_yl=p7_path_tick(p7_states+0,p7_buf_l,p7_wpos);' not in original:raise ValueError('Unknown optimized panner shape')
 fast=original
fast=once(fast,original.splitlines()[0],'desc:Hyperreal 3D Panner Fast (Preserved Renderer)')
fast=fast.replace('Hyperreal 3D Panner V7.1.2','Hyperreal 3D Panner Fast')
# Keep the separately qualified experimental designs on their frozen shell.
reference=ROOT/'tests/faust/reference/3DPanner-before-fast.jsfx'
if reference.exists():original=reference.read_text(encoding='utf-8')
# The new instrument reuses the interaction/control shell, not either old DSP.
head=original[:section(original,'sample')]
seed=head.index('// Initialize from saved targets before processing the first sample.')
head=head[:seed]+'''// New FAUST renderer owns smoothing and all audio histories.
seed_pending=0;samples_since_init+=samplesblock;
p71_sm_az=p71_target_az;p71_sm_el=tgt_elev*90;p71_sm_range=dist_m;
p7_sm_qw=p7_t_qw;p7_sm_qx=p7_t_qx;p7_sm_qy=p7_t_qy;p7_sm_qz=p7_t_qz;
p7_m00=1-2*(p7_t_qy*p7_t_qy+p7_t_qz*p7_t_qz);
p7_m01=2*(p7_t_qx*p7_t_qy+p7_t_qw*p7_t_qz);p7_m02=2*(p7_t_qx*p7_t_qz-p7_t_qw*p7_t_qy);
p7_m10=2*(p7_t_qx*p7_t_qy-p7_t_qw*p7_t_qz);p7_m11=1-2*(p7_t_qx*p7_t_qx+p7_t_qz*p7_t_qz);p7_m12=2*(p7_t_qy*p7_t_qz+p7_t_qw*p7_t_qx);
p7_m20=2*(p7_t_qx*p7_t_qz+p7_t_qw*p7_t_qy);p7_m21=2*(p7_t_qy*p7_t_qz-p7_t_qw*p7_t_qx);p7_m22=1-2*(p7_t_qx*p7_t_qx+p7_t_qy*p7_t_qy);
p7_rotate(p7_t_x,p7_t_y,p7_t_z);
np_lat=p7_rx/max(.2,dist_m);np_front=p7_ry/max(.2,dist_m);np_up=p7_rz/max(.2,dist_m);
np_shadow_l=exp(-2*$pi*(1600+13000*(1-max(0,np_lat)*tgt_throw)*(1-.35*max(0,-np_front)*tgt_throw))/srate);
np_shadow_r=exp(-2*$pi*(1600+13000*(1-max(0,-np_lat)*tgt_throw)*(1-.35*max(0,-np_front)*tgt_throw))/srate);
np_air=exp(-2*$pi*(1800+15000/(1+dist_m*.08*p7_t_air))/srate);
np_occ=exp(-2*$pi*(450+15000*(1-tgt_occ))/srate);
p7_room_w=4+24*(tgt_rsize*tgt_rsize*(.72+.28*tgt_rsize));p7_room_d=5+27*(tgt_rsize*tgt_rsize*(.72+.28*tgt_rsize));
p7_room_h=p7_t_room_height;p7_ear_height=p7_t_ear_height;
p7_source_x=p7_t_x;p7_source_y=p7_t_y;p7_source_z=p7_t_z;
np_clearance=min(min(p7_room_w*.5-abs(p7_t_x),p7_room_d*.5-abs(p7_t_y)),min(p7_ear_height+p7_t_z,p7_room_h-p7_ear_height-p7_t_z));
np_room_weight=p7_model ? p3d_clamp(np_clearance/.3,0,1) : 1;
p7_boundary=np_clearance<.3;
p7_requested_r=dist_m;p7_center_r=dist_m;p71_extent=tgt_width*1.92;
'''
faust=(ROOT/'tests/faust/panner_renderer.dsp').read_text(encoding='utf-8')
new=head+'\n@faust block\n'+faust+'\n'+original[section(original,'gfx'):]
new=re.sub(r'(?m)^kemar\[\d+\] = .*\n', '', new)
new=re.sub(r'(?m)^// #HELP:.*\n', '', new)
new=new.replace('// #TOOLTIP: Headphone-focused perceptual object panner with local position and room controls.', '// #HELP: New FAUST perceptual renderer. Artistic/Physical are approximate cue models, not the original KEMAR renderer.\n// #HELP: Same canvas, automation and manager shell; new smoothed timing, shadow, comb-pinna, six early taps and eight late feedback combs.\n// #HELP: Travel time is bounded to 65534 samples. This renderer deliberately sounds different.\n// #TOOLTIP: New FAUST headphone panner; original controls with a new audio model.')
new=once(new,original.splitlines()[0],'desc:Hyperreal FAUST Panner (New Renderer)')
new=new.replace('Hyperreal 3D Panner V7.1.2','Hyperreal FAUST Panner')
# Replace historical fitted-HRTF claims in the reused interaction shell.
for old,new_text in {
 'Artistic | corrected DSP | local controls':'FAUST Artistic | designed spatial cues',
 'Physical | ear paths, spectral fit and 3D room':'FAUST Physical | timing, coloration and room cues',
 'V7.1.2 Artistic | original distance mapping in metres.':'FAUST Artistic | distance control in metres.',
 'Headphones | KEMAR fit: -40 to +90 deg':'Headphones | designed pinna coloration',
 'Physical: fitted spectra and geometric timing.':'Physical: distance loss and bounded travel timing.',
 'Three variants of one KEMAR fit, not people.':'Three designed coloration profiles, not people.',
 'KEMAR spectral fit with floor/ceiling paths.':'Designed pinna cues with floor/ceiling taps.',
 'Physical adds 3D paths; Pan Curve softens center.':'Two cue models; Pan Curve shapes Artistic placement.',
}.items():new=once(new,old,new_text)

# Hybrid: retain the original Artistic signal path, replace only Physical audio.
hybrid=original
start=hybrid.index('p7_running ? (',section(hybrid,'sample'))
end=hybrid.index('p7_was_running=p7_running;',start)
hybrid=hybrid[:start]+hybrid[end:]
start=hybrid.index('p7_running ? (',hybrid.index('p7_out_l=0;',section(hybrid,'sample')))
end=hybrid.index('p7_wpos=(p7_wpos+1)&p7_mask;',start)
hybrid=hybrid[:start]+hybrid[end:]
hybrid=once(hybrid,'out_l=art_l*(1-p7_blend)+p7_out_l*p7_blend;','out_l=art_l;')
hybrid=once(hybrid,'out_r=art_r*(1-p7_blend)+p7_out_r*p7_blend;','out_r=art_r;')
# Keep the independent Artistic late field warm through renderer transitions.
hybrid=once(hybrid,'  sv_in=sv_in*(1-p7_blend)+(p7_send_l+p7_send_r)*p7_late_gain*p71_sm_room_weight*p7_blend;','  // Artistic excitation stays independent of Physical mode.')
hybrid=once(hybrid,'  sv_side_in=sv_side_in*(1-p7_blend)+(p7_send_l-p7_send_r)*p7_late_gain*p71_sm_room_weight*p7_blend;','')
# Use private target geometry: do not overwrite Artistic smoother histories.
prep=head[head.index('p7_m00=',head.index('// New FAUST renderer')):]
hybrid=hybrid[:section(hybrid,'sample')]+prep+'\n'+hybrid[section(hybrid,'sample'):]
physical=faust[:faust.index('// Existing display bindings')]
physical=physical.replace('spl0','in_l').replace('spl1','in_r')
physical=once(physical,'np_phys=np_s(p7_model);','np_phys=1;')
physical=re.sub(r'np_side=.*?;', 'np_side=np_bound(-1,1,np_s(np_lat)+np_s(tgt_motion)*.035*os.osc(.17));',physical)
physical=re.sub(r'np_ahead=.*?;', 'np_ahead=np_bound(-1,1,np_s(np_front));',physical)
physical=re.sub(r'np_gain=.*?;', 'np_gain=(1-.65*np_occlusion)*pow(max(.2,np_dist),-np_s(p7_t_distance_gain));',physical)

physical+='process=select2(p7_blend<=0,spl0*(1-p7_blend)+np_output_l*p7_blend,spl0),select2(p7_blend<=0,spl1*(1-p7_blend)+np_output_r*p7_blend,spl1);\n'
at=section(hybrid,'gfx')
hybrid=hybrid[:at]+'@block\n// Explicit buffer boundary after the original Artistic sample pass.\n@faust block\n'+physical+'\n'+hybrid[at:]
hybrid=once(hybrid,original.splitlines()[0],'desc:Hyperreal Hybrid Panner (Original Artistic / FAUST Physical)')
hybrid=hybrid.replace('Hyperreal 3D Panner V7.1.2','Hyperreal Hybrid Panner')
for old,new_text in {
 'Physical | ear paths, spectral fit and 3D room':'FAUST Physical | designed timing, coloration and room',
 'Frequency-scale variant of the KEMAR spectral fit; not an individual listener measurement.':'Frequency-scale variant of the designed FAUST Physical coloration.',
 'Physical adds 3D paths; Pan Curve softens center.':'FAUST Physical cues; Pan Curve softens Artistic center.',
 'Headphones | KEMAR fit: -40 to +90 deg':'Headphones | designed Physical coloration',
 'Physical: fitted spectra and geometric timing.':'Physical: new FAUST distance and travel cues.',
 'Three variants of one KEMAR fit, not people.':'Three designed coloration profiles, not people.',
 'KEMAR spectral fit with floor/ceiling paths.':'FAUST pinna cues with floor/ceiling taps.',
}.items():hybrid=once(hybrid,old,new_text)
hybrid=re.sub(r'(?m)^// #HELP:.*\n','',hybrid)
hybrid=hybrid.replace('// #TOOLTIP:', '// #HELP: Original Artistic audio; new FAUST Physical renderer. Both histories stay warm for mode changes.\n// #HELP: Physical deliberately sounds different. No fitted KEMAR rendering. Local controls only.\n// #TOOLTIP:',1)

for slug,name,code,source in [('HyperrealFast','Hyperreal Panner Fast','HpFt',fast),('HyperrealFaust','Hyperreal FAUST Panner','HpFa',new),('HyperrealHybrid','Hyperreal Hybrid Panner','HpHy',hybrid)]:
 p=ROOT/'plugins/Spatialization'/slug;p.mkdir(exist_ok=True);(p/'src').mkdir(exist_ok=True)
 (p/'src'/f'{slug}.jsfx').write_text(source,encoding='utf-8')
 (p/'plugin.json').write_text(json.dumps({'name':name,'slug':slug,'pluginCode':code,'bundleId':'com.zorakaudio.experimental.'+slug.lower(),'clapId':'com.zorakaudio.experimental.'+slug.lower(),'clapFeatures':['audio-effect'],'pluginType':'jsfx','nativeGfx':'legacy','entry':f'src/{slug}.jsfx'},indent=2)+'\n',encoding='utf-8')
 print(slug,'generated',flush=True)


