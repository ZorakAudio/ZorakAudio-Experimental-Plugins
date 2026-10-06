from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
s=(root/'plugins/Spectral/CMD/src/CrossMixDeclutter.jsfx').read_text()
s=s.replace(s.splitlines()[0],'desc:CMD Flow - Cooperative Mix Engine (FAUST)')
s=s.replace('in_pin:left input','options:idle=always\noptions:gfx_hz=30\n\nin_pin:left input',1)
s=s.replace('slider11:12<8,24,4>ERB Bands / CPU','slider11:12<12,12,1>-Fixed 12-band engine')
a=s.index('  new_active_nb =');b=s.index('  width_cap =',a)
s=s[:a]+'  active_nb = 12;\n'+s[b:]
s=s.replace('NB = 24;','NB = 12;')
s=s.replace('desired = max_cut_coeff * pressure', 'desired = manifold_amt * max_cut_coeff * pressure')
s=s.replace('targetWidth = clamp(targetWidth + pistonTargetWidth, -JND_WIDTH_LIMIT, JND_WIDTH_LIMIT);', 'targetWidth = manifold_amt * clamp(targetWidth + pistonTargetWidth, -JND_WIDTH_LIMIT, JND_WIDTH_LIMIT);')
s=s.replace('fc_i = hz_from_erb(erb_c);', 'fc_i = i==0 ? 40 : (i==11 ? 20000 : 80*200^((i-.5)/10));')
s=s.replace('gmem_attach_size(#bus_name, GMEM_SIZE);','sprintf(#flow_namespace,"ZorakAudio.CMD.Flow.%s",#bus_name);\ngmem_attach_size(#flow_namespace, GMEM_SIZE);')
s=s.replace('comm_join("ZorakAudio.CrossMixSomaticBus")','comm_join("ZorakAudio.CMD.Flow")')
s=s.replace('Cross-Mix Somatic Bus v4 Manifold','CMD Flow | Full-block FAUST')
s=s.replace('Blue=local ERB activity.','Blue=local broad-band activity.')
init='cf_energy_c=1-exp(-1/(.010*srate));\ncf_smooth_c=1-exp(-1/(.040*srate));\ncf_breath=1;\n'
for i in range(12):
 init+=f'cf_e{i}=0;\n'
for i in range(11):
 # broad logarithmic partition; low-pass differences telescope to original input
 hz=80*(16000/80)**(i/10)
 init+=f'cf_a{i}=1-exp(-2*$pi*min({hz:.12g},srate*.42)/srate);\n'
s=s.replace('@slider\n',init+'\n@slider\n',1)
a=s.index('@sample');b=s.index('@gfx',a)
pre='''// Summaries from the preceding FAUST block feed peer policy once per block.
cf_energy_c=1-exp(-1/(.010*srate));
cf_smooth_c=1-exp(-1/(.040*srate));
cf_breath_phase += samplesblock/srate*(.045+.420*bus_arousal+.220*bus_thrust)*2*$pi;
cf_breath_phase -= floor(cf_breath_phase/(2*$pi))*2*$pi;
cf_breath=.70+.30*sin(cf_breath_phase);
total_accum=0; low_accum=0; contact_accum=0; visc_accum=0;
'''
for i in range(12):
 pre+=f'accum[{i}]=max(0,cf_e{i})*samplesblock; total_accum+=accum[{i}];\n'
 if i<3:pre+=f'low_accum+=accum[{i}];\n'
 if 5<=i<=10:pre+=f'contact_accum+=accum[{i}];\n'
 if 3<=i<=9:pre+=f'visc_accum+=accum[{i}];\n'
pre+='onset_accum=max(0,total_accum-cf_previous_total); cf_previous_total=total_accum;\n'
s=s.replace('@block\n','@block\n'+pre,1)
imports=''
for i in range(12):imports+=f'cf_tcut{i}=targetCut[{i}]; cf_tsom{i}=targetSom[{i}]; smoothCut[{i}]+=(1-exp(-samplesblock/(.040*srate)))*(targetCut[{i}]-smoothCut[{i}]); smoothSom[{i}]+=(1-exp(-samplesblock/(.040*srate)))*(targetSom[{i}]-smoothSom[{i}]);\n'
s=s[:s.index('@sample')] if False else s
# locate again after inserting block body
# Shared policy runs at about 50 Hz; audio interpolation remains per sample.
a=s.index('setup_from_sliders();',s.index('@block'))
b=s.index('@sample',a)
s=s[:a]+'cf_policy_time += samplesblock/srate;\ncf_policy_time >= .020 ? (\ncf_policy_time = cf_policy_time-floor(cf_policy_time/.020)*.020;\n'+s[a:b]+'\n);\n'+s[b:]
s=s.replace('@slider\n','@slider\ncf_policy_time=.020;\n',1)
s=s.replace('STALE_TICKS = MAX_SLOTS * 128;', 'STALE_TICKS = MAX_SLOTS * 16;')
a=s.index('@sample');b=s.index('@gfx',a)
s=s[:a]+imports+'\n__AUDIO__\n'+s[b:]
f='''@faust block
cf_pole(c,x)=x:(step~_):(!,_) with {
 step(previous,value)=next,next with { next=previous+(value-previous)*c; };
};
'''
eel='@sample\n'
for ch,inp in [('l','spl0'),('r','spl1')]:
 for i in range(11):
  f+=f'cf_lp_{ch}{i}=cf_pole(cf_a{i},{inp});\n'
  eel+=f'cf_lp_{ch}{i}+=( {inp}-cf_lp_{ch}{i})*cf_a{i};\n'
 for i in range(12):
  expr=f'cf_lp_{ch}0' if i==0 else f'{inp}-cf_lp_{ch}10' if i==11 else f'cf_lp_{ch}{i}-cf_lp_{ch}{i-1}'
  f+=f'cf_b_{ch}{i}={expr};\n';eel+=f'cf_b_{ch}{i}={expr};\n'
for i in range(12):
 energy=f'.5*(cf_b_l{i}*cf_b_l{i}+cf_b_r{i}*cf_b_r{i})'
 f+=f'cf_e{i}=cf_pole(cf_energy_c,{energy});\ncf_cut{i}=cf_pole(cf_smooth_c,min(.95,max(0,cf_tcut{i})));\ncf_som{i}=cf_pole(cf_smooth_c,min(.25,max(-.25,cf_tsom{i})));\ncf_gain{i}=1-cf_cut{i}+cf_som{i}*cf_breath;\n'
 eel+=f'cf_e{i}+=({energy}-cf_e{i})*cf_energy_c;\ncf_cut{i}+=(min(.95,max(0,cf_tcut{i}))-cf_cut{i})*cf_smooth_c;\ncf_som{i}+=(min(.25,max(-.25,cf_tsom{i}))-cf_som{i})*cf_smooth_c;\ncf_gain{i}=1-cf_cut{i}+cf_som{i}*cf_breath;\n'
for ch in ['l','r']:
 expr='+'.join(f'cf_b_{ch}{i}*cf_gain{i}' for i in range(12))
 f+=f'cf_out_{ch}={expr};\n';eel+=f'cf_out_{ch}={expr};\n'
f+='cf_width=cf_pole(cf_smooth_c,targetWidth);\ncf_sat=cf_pole(cf_smooth_c,min(.35,max(0,targetSat)));\n'
eel+='cf_width+=(targetWidth-cf_width)*cf_smooth_c;\ncf_sat+=(min(.35,max(0,targetSat))-cf_sat)*cf_smooth_c;\n'
common=['cf_mid=.5*(cf_out_l+cf_out_r);','cf_side=.5*(cf_out_l-cf_out_r)*(1+cf_width);','cf_left=cf_mid+cf_side;','cf_right=cf_mid-cf_side;','spl0=min(8,max(-8,out_gain*cf_left*(1+cf_sat)/(1+cf_sat*abs(cf_left))));','spl1=min(8,max(-8,out_gain*cf_right*(1+cf_sat)/(1+cf_sat*abs(cf_right))));']
f+='\n'.join(common).replace('spl0=','cf_audio_l=').replace('spl1=','cf_audio_r=')+'\nprocess=cf_audio_l,cf_audio_r;\n';eel+='\n'.join(common)+'\n'
folder=root/'tests/faust/experiments/CMDFlow'
(folder/'src/CMDFlow.jsfx').write_text(s.replace('__AUDIO__',f))
fixture=root/'tests/faust/fixtures/CMDFlowEquivalent.jsfx';fixture.write_text(s.replace('__AUDIO__',eel))
manifest={'name':'CMD Flow','slug':'CMDFaust','pluginCode':'CMF2','bundleId':'com.zorakaudio.experimental.cmdflow','clapId':'com.zorakaudio.experimental.cmdflow','clapFeatures':['audio-effect'],'pluginType':'jsfx','entry':'src/CMDFlow.jsfx','nativeGfx':'legacy'}
(folder/'plugin.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated separate CMD Flow and equivalent EEL reference')
