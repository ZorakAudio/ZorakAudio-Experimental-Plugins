from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'build/catalog-faust-audit/DPT'
s=(base/'baseline.jsfx').read_text(encoding='utf-8')
s=s.replace('@init\n','@init\nza_sleep_ready=0;za_dpt_zero_run=0;dpt_shl=0;dpt_shr=0;dpt_dfl=0;dpt_dfr=0;gsh=0;gdf=0;\n',1)
block='''@block
za_dpt_settled = (pan_s + (pan_t-pan_s)*slew_g) === pan_s &&
  (nat_s + (nat_t-nat_s)*slew_g) === nat_s;
za_dpt_sh = pan_s >= 0 ? dpt_shl : dpt_shr;
za_dpt_df = pan_s >= 0 ? dpt_dfl : dpt_dfr;
za_dpt_filters_settled = (za_dpt_sh + gsh*(0-za_dpt_sh)) === za_dpt_sh &&
  (za_dpt_df + gdf*(0-za_dpt_df)) === za_dpt_df;
za_sleep_ready = za_dpt_zero_run >= BUF_LEN && za_dpt_settled &&
  (mode === 0 || za_dpt_filters_settled);

@faust
import("stdfaust.lib");
dpt_pole(c,on,x)=x:(step~_):(!,_) with {
 step(previous,value)=next,next with {next=select2(on,previous,previous+c*(value-previous));};
};
pan_s=dpt_pole(slew_g,1,pan_t);
nat_s=dpt_pole(slew_g,1,nat_t);
dpt_x=0.5*(spl0+spl1);
za_dpt_zero_run=dpt_x:(step~_):(!,_) with {
 step(previous,value)=next,next with {next=select2(value==0,0,min(BUF_LEN,previous+1));};
};
dpt_abs=abs(pan_s);
dpt_p=dpt_abs*dpt_abs*(3-2*dpt_abs);
dpt_gl0=sqrt(0.5*(1-pan_s));
dpt_gr0=sqrt(0.5*(1+pan_s));
dpt_farleft=pan_s>=0;
dpt_fill=0.45*nat_s*dpt_abs;
dpt_glbase=select2(dpt_farleft,dpt_gl0,dpt_gl0+dpt_fill*(1-dpt_gl0));
dpt_grbase=select2(dpt_farleft,dpt_gr0+dpt_fill*(1-dpt_gr0),dpt_gr0);
dpt_norm=sqrt(dpt_glbase*dpt_glbase+dpt_grbase*dpt_grbase)+EPS;
dpt_gl=dpt_glbase/dpt_norm;
dpt_gr=dpt_grbase/dpt_norm;
dpt_itd=min(64,floor(0.00066*dpt_p*nat_s*srate+0.5));
dpt_d1=min(256,floor((0.0009+0.0007*dpt_abs)*srate+0.5));
dpt_d2=min(256,floor((0.0019+0.0011*dpt_abs)*srate+0.5));
dpt_xd=dpt_x:de.delay(64,int(dpt_itd));
dpt_x1=dpt_x:de.delay(256,int(dpt_d1));
dpt_x2=dpt_x:de.delay(256,int(dpt_d2));
dpt_dfin=0.6*dpt_x1+0.4*dpt_x2;
dpt_lpcoef(fc)=w/(w+srate) with {f=min(max(fc,40),0.49*srate);w=6.283185307179586*f;};
gsh=dpt_lpcoef(max(700,18000/(1+12*nat_s*(dpt_abs*dpt_abs))));
gdf=dpt_lpcoef(1400+1600*(1-dpt_abs));
dpt_dfgain=0.22*nat_s*dpt_p;
dpt_shl=dpt_pole(gsh,(mode!=0)&dpt_farleft,dpt_xd);
dpt_shr=dpt_pole(gsh,(mode!=0)&(pan_s<0),dpt_xd);
dpt_dfl=dpt_pole(gdf,(mode!=0)&dpt_farleft,dpt_dfin);
dpt_dfr=dpt_pole(gdf,(mode!=0)&(pan_s<0),dpt_dfin);
dpt_headl=select2(dpt_farleft,dpt_gl*dpt_x,dpt_gl*dpt_shl+dpt_dfgain*dpt_dfl);
dpt_headr=select2(dpt_farleft,dpt_gr*dpt_shr+dpt_dfgain*dpt_dfr,dpt_gr*dpt_x);
process=min(8,max(-8,select2(mode==0,dpt_headl,dpt_gl0*dpt_x)*out_gain)),
        min(8,max(-8,select2(mode==0,dpt_headr,dpt_gr0*dpt_x)*out_gain));

'''
s=re.sub(r'(?ms)^@sample[^\n]*\n.*?(?=^@|\Z)',lambda _:block,s)
(base/'faust-candidate.jsfx').write_text(s,encoding='utf-8');print('DPT FAUST candidate generated')
