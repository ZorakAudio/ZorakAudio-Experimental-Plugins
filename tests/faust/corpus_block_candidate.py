"""Second Corpus candidate: private Faust histories and audio-only block boundary."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from sampler_corpus_candidates import BASE,source
s=source('Corpus');folder=BASE/'Corpus'
faust='''@faust
import("stdfaust.lib");
cf_on=(td_enabled!=0)&(slider45>0.000001);
cf_reset=(cf_epoch!=cf_epoch')|((1')==0);
cf_blockreset=(cf_serial!=cf_serial')|((1')==0);
cf_age=1:(step~_):(!,_) with {step(previous,value)=next,next with {next=select2(cf_reset,previous+1,0);};};
cf_slew(a,x)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2(cf_reset,old,0);next=select2(cf_on,previous,previous+(value-previous)*a);};
};
cf_peak(a,x)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2(cf_reset,old,0);next=select2(cf_on,previous,max(value,previous*a));};
};
cf_ar(x)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2(cf_reset,old,0);a=select2(value>previous,0.015,0.45);next=select2(cf_on,0,previous+(value-previous)*a);};
};
cf_max(x)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2(cf_blockreset|cf_reset,old,0);next=select2(cf_on,previous,max(previous,value));};
};
cf_hold(x,initial)=x:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2(cf_reset,old,initial);next=select2(cf_on,previous,value);};
};
cf_ap(n,g,x)=x:(step~de.delay(1023,int(n)-1)):(!,_) with {
 step(delayed,value)=next,y with {
  d=select2(cf_age>=n,0,delayed);y=-g*value+d;next=select2(cf_on,0,value+g*y);
 };
};
td_amount_rt=cf_slew(td_param_a,slider45);
cf_amt=min(1,max(0,td_amount_rt));cf_abs=max(abs(spl0),abs(spl1));
td_fast=cf_peak(td_fast_rel,cf_abs);td_slow=cf_slew(td_slow_a,cf_abs);
cf_gate=min(1,max(0,(td_fast-0.00002)/0.0015));
cf_trans=min(1,max(0,(td_fast-td_slow*1.12)/max(0.000001,td_fast*0.82)))*cf_gate;
td_transient=cf_hold(cf_trans,0);
td_mix=cf_ar(cf_amt*cf_trans);td_dry_peak=cf_peak(td_peak_rel,cf_abs);
cf_g=0.30+0.30*cf_amt;
cf_wl=spl0:cf_ap(td_len0,cf_g):cf_ap(td_len1,min(0.64,cf_g+0.04)):cf_ap(td_len2,max(0.20,cf_g-0.03));
cf_wr=spl1:cf_ap(td_len0,cf_g):cf_ap(td_len1,min(0.64,cf_g+0.04)):cf_ap(td_len2,max(0.20,cf_g-0.03));
td_work_l=cf_hold(cf_wl,0);td_work_r=cf_hold(cf_wr,0);
cf_mix=min(1,max(0,td_mix));cf_pl=spl0*(1-cf_mix)+cf_wl*cf_mix;cf_pr=spl1*(1-cf_mix)+cf_wr*cf_mix;
cf_peakout=max(abs(cf_pl),abs(cf_pr));cf_ceiling=max(0.000000001,td_dry_peak*1.001);
cf_guard0=select2(cf_peakout>cf_ceiling,1,min(1,cf_ceiling/cf_peakout));
cf_goal=max(0.000000001,td_dry_peak*(1-0.28*cf_amt*cf_trans));
cf_guard=min(1,max(0,select2((cf_trans>0.15)&(cf_peakout>cf_goal),cf_guard0,min(cf_guard0,cf_goal/cf_peakout))));
td_guard_gain=cf_hold(cf_guard,1);
cf_ol=select2(cf_on,spl0,cf_pl*cf_guard);cf_or=select2(cf_on,spl1,cf_pr*cf_guard);
td_block_trans=cf_max(cf_trans);td_block_mix=cf_max(cf_mix);td_block_guard=cf_max(1-cf_guard);
td_block_inpeak=cf_max(cf_abs);td_block_outpeak=cf_max(max(abs(cf_ol),abs(cf_or)));
process=cf_ol,cf_or;
'''
# Extracted reference keeps all original arithmetic and ring operations.
old=(folder/'kernel-before.jsfx').read_text();old=old.replace('function td_clear() (','function td_clear() (\ncf_epoch+=1;')
old=old.replace('@sample\n','@block\ncf_serial+=1;td_block_trans=0;td_block_mix=0;td_block_guard=0;td_block_inpeak=0;td_block_outpeak=0;\n@sample\n',1)
new=old[:old.index('@sample')]+faust
(folder/'block-kernel-before.jsfx').write_text(old);(folder/'block-kernel-after.jsfx').write_text(new)
# Full integration must also own output gain smoothing so no EEL scalar return
# introduces sample fusion. Preserve the original gain initialization through a zone.
start=s.index('td_enabled && slider45>0.000001 ? (',s.index('@sample'));end=s.index('// Audible continuity',start)
body=faust.replace('process=cf_ol,cf_or;', '''cf_gain=output_target:(step~_):(!,_) with {
 step(old,value)=next,next with {previous=select2((1')==0,old,cf_initial_gain);next=previous+(value-previous)*0.002;};
};
output_smooth=cf_gain;
cf_gl=cf_ol*cf_gain;cf_gr=cf_or*cf_gain;cf_outputpeak=max(abs(cf_gl),abs(cf_gr));
output_peak=cf_outputpeak;
cf_catch=select2((slider20>=0.5)&(cf_outputpeak>0.98),1,0.98/cf_outputpeak);
process=cf_gl*cf_catch,cf_gr*cf_catch;''')
full=s[:start]+'spl0=out_l;spl1=out_r;clock+=1;\n'+body+'\n@sample\nout_l=spl0;out_r=spl1;\n'+s[end:]
# Preserve the pre-limit peak published by the original; audio and peak are
# distinct outputs. The full pipeline remains sample fused for shared EEL state.
full=full.replace('spl0=out_l;spl1=out_r;clock+=1;\nza_keep_awake=', 'spl0=out_l;spl1=out_r;\n@block\nza_keep_awake=',1)
full=full.replace('function td_clear() (','function td_clear() (\ncf_epoch+=1;')
full=full.replace('@block\ncg_retire();','@block\ncf_serial+=1;cg_retire();',1)
full=full.replace('clock=0;output_smooth=cp_amp(slider7);','clock=0;output_smooth=cp_amp(slider7);cf_initial_gain=output_smooth;',1)
# Counter accumulation exported at block end, rather than scalar feedback per sample.
full=full.replace('process=cf_gl*cf_catch,cf_gr*cf_catch;', '''cf_limited=(slider20>=0.5)&(cf_outputpeak>0.98);
cf_limit_count=cf_limited:(step~_):(!,_) with {step(old,value)=next,next with {previous=select2(cf_blockreset,old,0);next=previous+value;};};
process=cf_gl*cf_catch,cf_gr*cf_catch;''')
full=full.replace('@block\nza_keep_awake=', '@block\nlimited_samples+=cf_limit_count;\nza_keep_awake=',1)
(folder/'block-candidate.jsfx').write_text(full,encoding='utf-8');print('Corpus block candidate generated')
