"""Response checks for the new renderer's live audio controls, using one source."""
from pathlib import Path
import subprocess,array,json,math,argparse
p=argparse.ArgumentParser();p.add_argument('--label',choices=['faust','hybrid'],default='faust');args=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/panner'/args.label;OUT=BASE/'controls';OUT.mkdir(exist_ok=True)
RECORDING=r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
contexts={'physical':(1,{}),'artistic':(0,{}),'stereo':(3,{}),'bed':(4,{}),'room':(1,{7:0,12:1}),'late':(1,{7:0,27:.8})}
cases=[('lateral','physical',6,-.55),('front_rear','physical',8,-.72),('distance','physical',7,.55),('far_distance','physical',7,1),('throw','physical',9,1),('mono_size','physical',10,.9),('push_out','physical',11,1),('room','room',12,.1),('occlusion','physical',13,.8),('motion','physical',14,1),('trim','physical',15,6),('automation_safe','physical',16,0),('pan_curve','artistic',17,3),('smoothing','physical',18,200),('room_size','room',19,.9),('elevation','physical',21,.65),('source_stereo','physical',22,1),('source_bed','physical',22,2),('source_dual','physical',22,3),('input_width','stereo',23,.15),('bed_anchor','bed',24,.95),('late_off','late',27,0),('space_size','late',28,.9),('protect','late',29,0),('role_vocal','late',32,1),('role_ambience','late',32,5),('model','physical',36,0),('pinna_profile','physical',37,2),('distance_gain','physical',38,0),('travel','physical',39,1),('air','physical',40,0),('yaw','physical',41,60),('pitch','physical',42,45),('roll','physical',43,60),('room_height','room',44,12),('ear_height','room',45,.4)]
def run(name,context,changes):
 case,defaults=contexts[context];controls=defaults|changes;p=OUT/(name+'.json');args=[str(BASE/'host.exe'),RECORDING,str(p),'48000','256',str(case)]
 for slider,value in controls.items():args.extend([str(slider),str(value)])
 subprocess.run(args,check=True);a=array.array('f');a.frombytes(Path(str(p)+'.pcm.bin').read_bytes());return a,json.loads(p.read_text())
references={k:run('base-'+k,k,{}) for k in contexts};rows=[]
for name,context,slider,value in cases:
 before=references[context][0];after,measurement=run(name,context,{slider:value});error=max(abs(x-y) for x,y in zip(before,after));rms=math.sqrt(sum((x-y)**2 for x,y in zip(before,after))/len(before));assert error>1e-8,('Control has no tested response',name,error)
 if name=='trim':
  scale=10**(6/20);assert max(abs(y-x*scale) for x,y in zip(before,after))<2e-6,'Final trim is not the expected gain'
 rows.append({'control':name,'context':context,'slider':slider,'value':value,'maximum_difference':error,'rms_difference':rms,'measurement':measurement});(OUT/'results.json').write_text(json.dumps(rows,indent=2));print('CONTROL PASS',name,rms,flush=True)
print('PASS',len(rows),'audio control responses',flush=True)
