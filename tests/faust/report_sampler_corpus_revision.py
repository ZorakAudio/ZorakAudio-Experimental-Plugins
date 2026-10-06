from pathlib import Path
import json,struct,math,csv,hashlib,shutil
r=Path.cwd();base=r/'build/faust-sections/sampler-corpus';rows=[]
def audio(a,b):
 n=d=0;err=0
 with a.open('rb') as x,b.open('rb') as y:
  while raw:=x.read(1048576):
   other=y.read(len(raw));assert len(other)==len(raw);u=struct.unpack('<'+'f'*(len(raw)//4),raw);v=struct.unpack('<'+'f'*len(u),other)
   for q,w in zip(u,v):assert math.isfinite(q) and math.isfinite(w);n+=1;d+=q!=w;err=max(err,abs(q-w))
  assert y.read(1)==b''
 return {'samples':n,'different':d,'maximum_error':err}
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 a=base/'Corpus/revised-reference'/f'{rate}-{block}.json';b=base/'Corpus/revised-host'/a.name;before=json.loads(a.read_text());after=json.loads(b.read_text());traces=[]
 for p in [a,b]:
  with p.with_suffix('.json.state.csv').open() as f:traces.append([{k:v for k,v in row.items() if k not in ['td_fast','td_slow']} for row in csv.DictReader(f)])
 row={'rate':rate,'block':block,'before':before,'after':after,'speedup':before['process_seconds']/after['process_seconds'],'audio':audio(a.with_suffix('.json.pcm.bin'),b.with_suffix('.json.pcm.bin')),'retained_state_trace_equal':traces[0]==traces[1]};rows.append(row);assert row['audio']['maximum_error']==0;assert row['retained_state_trace_equal'];print(rate,block,row['speedup'])
a=base/'Corpus/clip-before/48000-256.json';b=base/'Corpus/revised-host/clip.json';clip={'before':json.loads(a.read_text()),'after':json.loads(b.read_text()),'audio':audio(a.with_suffix('.json.pcm.bin'),b.with_suffix('.json.pcm.bin'))};assert clip['audio']['maximum_error']==0;assert clip['before']['limited_samples']==clip['after']['limited_samples'];assert clip['before']['maximum_pre_limit_peak']==clip['after']['maximum_pre_limit_peak']
sample=json.loads((base/'Sample/private-results.json').read_text());manifest=json.loads((r/'tests/faust/fixtures/sampler-corpus/manifest.json').read_text());unchanged={k:hashlib.sha256((r/'plugins/Spectral'/k/'src'/(k+'.jsfx')).read_bytes()).hexdigest()==h for k,h in manifest.items()};assert all(unchanged.values())
data={'Corpus':rows,'Corpus_clip':clip,'Sample_private_EQ':sample,'production_sources_unchanged':unchanged,'Sample_whole_filter_compile':'120-second invocation timeout, line 297, inferred imports 30, exports 0; no runtime result'}
out=Path(r'C:\Users\LouisJenkinsCS\Documents\Codex\2026-10-05\can-x20\outputs');(out/'Corpus-Sample-Revised-Audit-Measurements.json').write_text(json.dumps(data,indent=2));dest=r/'docs/validation/corpus-sample-faust';(dest/'revised-measurements.json').write_text(json.dumps(data,indent=2))
text='''# Corpus and Sample revised FAUST audit — 6 October 2026

## Revised conclusion

Sample's three-band EQ is a successful block-kernel candidate: private histories and a coherent audio boundary give 2.67–4.36x measured speedup, with bit-identical tested audio. The earlier external-state bridge regression therefore did not establish that this computation was unsuitable for FAUST.

Corpus's full-plugin export-pruned candidate preserves tested audio and shows modest gains at larger buffers, but is still sample-fused. It does not establish a full-block migration. Its previously tested isolated private-delay block kernel already demonstrated up to 1.80x speedup. The missing piece remains the full-plugin boundary around voicing, continuity and meters.

Neither production plugin is migrated in this pass. This is a renewed implementation audit backed by revised candidates and measurements, not a claim that the complete sampler or analysis engines have been ported.

The proposed @faust block / @faust sample syntax has not been implemented. This pass verifies actual execution through generated stage metadata and runtime call-count assertions. It must not be read as testing a new explicit-mode compiler contract.

## Sample: genuinely block-based EQ

The revised isolated candidate moves all twelve EQ history values into a FAUST feedback recurrence. It imports fifteen coefficients once per host buffer through an EEL @block stage; it no longer imports previous states or exports next states. The generated stage has two audio inputs, two audio outputs, 23 control zones and zero scalar exports. There are no intermediate EEL sample stages to force fusion.

The reference is the original three-band EQ arithmetic and RAM state implementation extracted from Sample. Both sides exclude HP/LP processing for this diagnostic; the reference was narrowed as well as the candidate. Consequently these measurements are an apples-to-apples comparison of the three EQ bands, not a comparison against the earlier complete post-EQ-filter timing.

| Initial case | 48 kHz / 64 | 48 kHz / 256 | 96 kHz / 1024 |
|---|---:|---:|---:|
'''
for case in range(4):
 vals=[x['speedup'] for x in sample if x['case']==case];text+=f"| {case} | {vals[0]:.2f}x | {vals[1]:.2f}x | {vals[2]:.2f}x |\n"
text+='''
The initial cases include bypass and active EQ. All sequences subsequently change controls, bypass, re-enable and change gain again, retaining state across slider changes. They use stereo tones and impulses, a long quiet interval, 1e-9 input and subsequent recovery. HP/LP/resonance settings in the reused control driver do not affect this three-band-only kernel; those variations must not be presented as additional filter coverage.

Twelve cases cover 3,072,000 frames and 6,144,000 float output samples. All are finite and bit-identical to the extracted reference; nonzero output is explicitly required. Runtime assertions require zero scalar FAUST calls and exactly frames/buffer block calls. The quiet counter freezes during whole-strip bypass; disabled bands retain their histories; the original denormal cutoff and quiet-threshold state clear are implemented inside the recurrence.

Timings use three-trial medians, with state-dump collection in a separate fourth trial. The final matrix was rerun after concurrent compilation completed. Earlier exploratory timing values were discarded. Very short kernel timings remain sensitive to machine load; the magnitude and consistency of this result are more useful than individual decimal places.

Private histories are not directly compared to EEL RAM snapshots. Audio tests exercise them across changes and recovery, but do not prove every possible state trajectory. The extracted fixture uses a zero denormal guard, whereas production Sample alternates a tiny guard every sample. This difference is matched between the two kernel implementations, but still needs integration work in the complete plugin.

## Sample: whole post-EQ filter attempt

A second new candidate includes all thirteen stereo biquad slots, HP/LP one-pole states, three EQ bands, HP/LP resonance, inactive-path history retention and the quiet reset. It uses a coupled private-state recurrence with an audio-only boundary. Coefficients would be read at @block, rather than bridged on every sample.

FAUST generation exceeded the existing 120-second per-invocation build limit at section line 297, after discovering 30 imports and zero scalar exports. No usable candidate object or runtime result was generated. The reference compiled. The 30-import count is the partial discovery state when compilation timed out, not the complete number of required controls.

This repeats the earlier whole-filter compile difficulty with a different, private-state graph. Thus external state publication was not the only obstacle. It is a graph construction/build-complexity problem requiring further work; it is not evidence that a compiled thirteen-filter DSP would run slowly. Raising the timeout was not used to hide the issue.

## Sample: what full integration must preserve

The source still requires more than placing the isolated EQ after voice generation:

* HP stages precede EQ; LP stages follow it. EEL-owned filter histories and per-sample clears cross that boundary in the previous candidate.
* Character, solo and Contour processing follow the filter chain. The spectrum analyzer can read both pre-strip and post-Contour audio.
* denorm_guard_tick alternates every sample. It cannot become a block-constant scalar snapshot.
* Quiet-state reset, periodic scalar flushing and global silent-tail reset can affect processing state. A private FAUST implementation must receive those reset events at their original sample positions, or own the equivalent reset decision itself.
* active_voice_seen and dry-input values originate earlier in the sample chain and are consumed later. Separating those stages into whole-block loops would require per-frame streams, rather than ordinary scalar variables that retain only the last sample.

The appropriate architecture is a coherent private-state post-processing chain with an explicit reset contract and buffered per-frame values where necessary. Merely inserting a block barrier or disabling fusion is not a correctness fix. The kernel gain is a strong reason to pursue that architecture, but it is not yet a measured end-to-end Sample speedup. A new full Sample host benchmark was not performed because no equivalent full block candidate was produced.

## Corpus: export pruning in the complete plugin

The revised full candidate keeps td_amount_rt, td_fast, td_slow, td_dry_peak, td_guard_gain and td_transient private. Source inspection finds their external occurrences are clear/initialization assignments rather than consumers in this candidate. Actual block statistics, td_mix, output gain, pre-limit peak and limiter count remain exported.

The result has nine scalar exports plus two audio outputs, rather than the corrected preceding candidate's fifteen scalar exports plus two audio outputs. This reduces output-stream materialization and per-sample publication. The preceding historical normal-host metadata predates the pre-limit-peak export correction; it must not be used as the exact corrected-candidate channel count.

Generated metadata still marks sample -> FAUST -> sample as fused. No runtime dispatch or compiler-fusion rule is changed. This candidate still calls FAUST once per sample, so it is a pruning experiment, not the promised full-block design.

The loaded-bank comparison uses the original pinned Corpus and the revised candidate, with fresh paired timings after compilation and Sample profiling completed. Each run processes 32 seconds of output with MIDI overlaps, releases, retriggers, diffusion changes and gain changes. Both load only the user-authorized recording; Corpus reaches ready state with 1,888 grains. Preparation and file loading are excluded from processBlock timing.

| Rate / buffer | Original seconds | Revised seconds | Change in CPU time |
|---|---:|---:|---:|
'''
for row in rows:
 a=row['before']['process_seconds'];z=row['after']['process_seconds'];text+=f"| {row['rate']} / {row['block']} | {a:.5f} | {z:.5f} | {(z/a-1)*100:+.1f}% |\n"
text+='''
These are single paired host measurements, not statistical medians. The 64-sample result is effectively unchanged; the larger-buffer runs improve by roughly 8%. This is encouraging but insufficient to install the candidate or promise a repeatable whole-plugin gain.

All 12,288,000 normal float output samples are bit-identical. Retained state traces match after excluding td_fast and td_slow, which deliberately no longer publish to EEL. Text snapshots use six-digit precision and intermittent sampling, so trace equality is not a bitwise proof of all private state. The same benchmark also checks editor creation, release/reprepare at another rate and absence of DSP/GFX faults; it does not test interactive UI behavior exhaustively.

A separate +12 dB clipping run compares another 3,072,000 samples bit-identically against the previously captured original clipping reference. Both report 3,673 limited samples, pre-limit maximum 2.35429 and output ceiling 0.98. That comparison is for correctness; its old reference timing is not used as a new paired performance result.

## Corpus: why it still cannot batch as written

The original sample loop generates voices and per-frame coverage counts, performs diffusion and gain/limiting, then evaluates audible continuity and accumulates meters. The last stage uses both post-processing audio and values such as sample_coverage, sample_startedvoices, sample_lowvoices and voice_list_n from that exact frame. output_peak also means the pre-limit peak, which cannot be reconstructed from a clipped output sample.

A block migration therefore needs per-frame coverage/voice metadata and pre-limit peaks as internal streams, or must move the dependent bookkeeping into the same coherent block DSP. Clock advancement must stay aligned with voice generation. RAM allocation and task-arena preservation need review if new scratch streams are introduced. Renaming unused exports addresses none of these dependencies.

The prior isolated private-delay block test remains applicable to the diffusion computation; this pass did not rerun or inflate that measurement. Source indexing, feature extraction, structure and PE tasks were not rewritten or reprofiled here. Their existing defer infrastructure is separate from the FAUST audio stage.

## Validation and reproducibility

Production source hashes still match the pinned manifest for both plugins. No plugin manifest, installed binary, sleep grant, production DSP source or JoepVanlier source is changed. The Corpus executable uses the existing test-only runner identity and is not a distributable plugin. No audio file other than the authorized FLAC was loaded. Sample tests use generated RAM signals and comparison scripts read only generated dumps.

Toolchain and precision follow the prior audit: clang 21, FAUST 2.81.2 / LLVM 17, O2 AOT compilation, double DSP and float audio outputs. Nine compiler contract tests pass in this pass. The test driver adds a block-call assertion mode for the private Sample candidates; no production runtime optimization is introduced.

Reproduction requires the previous prepared sampler/corpus candidates. Run tests/faust/revise_sampler_corpus.py to create the export-pruned Corpus and three-EQ kernels, then tests/faust/profile_private_sample.ps1 for the numerical/performance matrix. tests/faust/sample_private_filter_candidate.py creates the whole-filter experiment; profile_private_sample.ps1 -WholeFilter attempts its build and currently reports the compiler limit. The comparison helper can also consume existing results without rebuilding. Corpus uses build_sampler_corpus.py and sampler_corpus_host with the authorized path, as in the first audit.

## Decision

The renewed audit changes Sample's assessment: its private-state block EQ is demonstrably promising, and the previous bridge's slowdown was an integration cost. It does not yet qualify a full Sample migration. The complete filter graph's build complexity and production reset/sideband semantics remain concrete work items.

Corpus's export reduction removes avoidable work and preserves tested rendering, but meaningful block execution requires a broader boundary design. Keep its production implementation until that design is compiled, its actual block call counts are verified, and complete audio plus meter/continuity/reset comparisons pass. Neither plugin should be classified as inherently unsuitable for FAUST on the basis of the earlier candidates.
'''
(out/'Corpus-Sample-Revised-Audit.md').write_text(text,encoding='utf-8');(dest/'historical-revised-audit.txt').write_text(text,encoding='utf-8');shutil.copyfile(base/'privatefull-compile.log',dest/'sample-private-whole-filter-compile.log')
print('Saved revised audit and measurements; production hashes unchanged.')
