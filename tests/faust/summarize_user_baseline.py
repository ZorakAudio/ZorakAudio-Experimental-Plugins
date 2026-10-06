# Exact pasted-baseline comparison and cooperative idle audit.
from pathlib import Path
import json,hashlib,shutil,csv,math
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(r'C:\Users\LouisJenkinsCS\Documents\Codex\2026-10-05\can-x20\outputs');OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'build/faust-sections/user-baseline';PROFILE=BASE/'profile'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()
before=json.loads((PROFILE/'before-active.json').read_text());after=json.loads((PROFILE/'after-active.json').read_text())
a=PROFILE/'before-active.json.pcm.bin';b=PROFILE/'after-active.json.pcm.bin';assert a.stat().st_size==b.stat().st_size==before['frames']*8
assert sha(a)==sha(b);assert before['sleep_blocks']==after['sleep_blocks']==0
old=json.loads((BASE/'idle-before-fix.json').read_text());new=json.loads((BASE/'idle-final.json').read_text())
assert all(not r['different_samples'] and not r.get('sleep_blocks',0) for r in new['cases'])
speed=before['processBlock_seconds']/after['processBlock_seconds'];saved=100*(1-1/speed)
source=json.loads((BASE/'source-comparison.json').read_text());source['hash_normalization']='UTF-8 source text with normalized line endings';source['pasted_fixture_byte_sha256']=sha(ROOT/'tests/faust/fixtures/EasyExpander-user.jsfx')
report={'platform':'Windows x64','source_comparison':source,'input_sha256':'1f44c88399234d13891fb9bd29eac84108262334c7e041111dabc89267c7e8c5','baseline':'Exact supplied EasyExpander source compiled through the repository AOT pipeline','after':'Matching EasyExpander Faust, rebuilt with the same explicit-permission-only host','settings':'Identical default sliders; 48 kHz stereo; 256-frame blocks; offline; no sleep on either side','before':before,'after':after,'speedup':speed,'processing_time_saved_percent':saved,'equivalence':{'float_samples':before['frames']*2,'different_samples':0,'max_error':0,'rms_error':0,'pcm_sha256':sha(a)},'kernel_profiles':json.loads((BASE/'kernel/results.json').read_text()),'idle_before_fix':old,'idle_final':new,'validation':{'cooperative_permission_expiration_wake_keep_awake_tasks_and_legacy_state_migration':'passed','compiler_contracts':8,'structured_task_regression_tests':11,'build_mode_regression_tests':7,'source_resolver_tests':25,'source_resolver_environment':'Canonical workspace temp directory; default 8.3 TEMP spelling caused an expected-path comparison mismatch on initial run.'},'scope':'Processor processing time, not a timed REAPER render; one final paired full-recording run and three-trial median kernel runs. No native REAPER JSFX performance claim.'}
(OUT/'EasyExpander-user-baseline-and-idle-results.json').write_text(json.dumps(report,indent=2))
with (OUT/'EasyExpander-user-baseline-and-idle-results.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['case','before_seconds','after_seconds','speedup','different_output_samples']);w.writerow(['full_processor_active',before['processBlock_seconds'],after['processBlock_seconds'],speed,0])
 for row in report['kernel_profiles']:w.writerow([f"kernel_{row['before']['sample_rate']}_{row['before']['block_size']}",row['before']['median_seconds'],row['after']['median_seconds'],row['speedup'],0])
problem=next(row for row in old['cases'] if row['fixture']=='supplied_recording' and not row['offline'] and row['mode']==0)
text=f'''# EasyExpander source comparison and sleep audit

The supplied source and the earlier repository baseline have identical generated
LLVM. Differences are help/tooltip comments and the final newline, with no change
to executable DSP. The Faust example now keeps the supplied EEL text.

Previous threshold-based Auto Sleep was not equivalent to active processing:
{problem['sleep_blocks']} blocks reported sleeping; {problem['different_samples']:,}
output samples differed; peak error {problem['max_error']:.9f}
(about {20*math.log10(problem['max_error']):.1f} dBFS).
{problem['different_samples_while_awake']:,} differences were in blocks reported
awake, consistent with detector/gain state frozen during sleep. The first
difference was at frame {problem['first_different_frame']} ({problem['first_different_frame']/48000:.3f}s).
Offline processing had the same failure before the fix.

The host now has one realtime policy: active until a fresh za_sleep_ready=1
plugin grant, with exact silence and no pending work. The status badge has no
mode selector. Legacy saved settings are ignored and removed. Unmodified plugins
stay active. Offline processing always advances DSP. See Cooperative-Sleep.md
for the conditional readiness promise and wake rules.

All {len(new['cases'])} final null comparisons passed on the full recording and
the quiet/recovery fixture derived from it: zero changed samples, including all
legacy selector values in realtime and offline modes. Cooperative fixture checks
also passed fresh/stale/no-grant, tiny input, parameter wake, keep-awake, retained
task result, serialized-state migration and offline processing.

## Matched continuously active comparison

| Processor | Processing time |
|---|---:|
| Supplied EasyExpander, AOT | {before['processBlock_seconds']:.3f} s |
| Matching EasyExpander Faust | {after['processBlock_seconds']:.3f} s |

{speed:.2f}x faster, {saved:.1f}% less processing time. All {before['frames']*2:,}
float output samples bit-identical. Default sliders, 48 kHz stereo, 256-frame
blocks, offline and zero slept blocks on both sides. Editor/rate-reset checks
passed. This is one paired full-recording run; kernel results use three-trial
medians across 48/96 kHz and 64/256/1024 frames.

These measure built processors, not a timed REAPER render or a comparison to
REAPER's native EEL JIT. Decoding and output dumping are excluded from processing
time. Full render speed includes those and other host costs. Earlier bit-identical
results compared two builds using the same Auto Sleep policy, so they did not
establish equivalence to continuously active DSP.

Install the new VST3/CLAP package to receive this policy. Other plugins must be
rebuilt; already-loaded/older binaries retain their previous sleep behavior.
The authorized recording was only read; no other audio input file was loaded.
'''
(ROOT/'docs/validation/EasyExpander-Sleep-Audit.md').write_text(text,encoding='utf-8');(OUT/'EasyExpander-Sleep-Audit.md').write_text(text,encoding='utf-8')
shutil.copy2(ROOT/'docs/Cooperative-Sleep.md',OUT/'Cooperative-Sleep.md')
p=ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md';s=p.read_text(encoding='utf-8');s+=f'\nUpdated matched active-processing result: {before["processBlock_seconds"]:.3f} s -> {after["processBlock_seconds"]:.3f} s ({speed:.2f}x), with {before["frames"]*2:,} bit-identical float samples. See EasyExpander-Sleep-Audit.md and the accompanying JSON report.\n';p.write_text(s,encoding='utf-8')
print(json.dumps({'speedup':speed,'before_seconds':before['processBlock_seconds'],'after_seconds':after['processBlock_seconds'],'null_cases':len(new['cases']),'bit_identical_samples':before['frames']*2},indent=2))
