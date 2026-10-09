"""Publish the complete native WDL timing matrix into the plugin help pages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from pluginlib import discover_plugins, read_plugin_readme

GFX = {'joep_reabee', 'joep_stereospectrumsplit', 'joep_saikemultispectralanalyzer',
       'joep_saikemultispectralanalyzer_mk2', 'joep_saikemultispectralanalyzer_old'}
LIMITED = {
    'joep_saike_bric_a_brac': 'No sample files were loaded; this measures the empty-slot baseline, not loaded texture playback.',
    'joep_sequencedfx': 'Default playback/pattern state; this does not measure an active chain of sequenced effects.',
    'joep_ripple': 'The default pattern is empty; this does not establish active MIDI-sequencing performance.',
    'joep_saike_routing_utility': 'Default monitor selection may output silence; active monitor routing needs a separate workload.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report = json.loads(args.results.read_text(encoding='utf-8'))
    specs = [s for s in discover_plugins(root) if s.category == 'JoepVanlier']
    rows = {row['plugin']: row for row in report['plugins']}
    if len(rows) != len(report['plugins']) or set(rows) != {s.slug for s in specs}:
        raise ValueError('Publication requires one result for every JoepVanlier plugin, without duplicates.')
    method = report['method']
    if 'native x64' not in method['reference']:
        raise ValueError('Do not publish interpreter results as native WDL JIT comparisons.')
    for row in rows.values():
        if row['status'] != 'MEASURED':
            continue
        cases = row.get('trials', [])
        if len(cases) != 2 or {c['block_size'] for c in cases} != {64, 512}:
            raise ValueError(f"{row['plugin']}: expected two disjoint buffer cases")
        for case in cases:
            if case['status'] not in {'EXACT', 'WITHIN_TOLERANCE', 'DIVERGES'}:
                raise ValueError(f"{row['plugin']}: unrecognized output status")
            if len(case['trials']) != method['trials']:
                raise ValueError(f"{row['plugin']}: incomplete trial coverage")
            for trial in case['trials']:
                if (trial['backend'] != 'WDL native x64 SSE JIT'
                        or trial['block_size'] != case['block_size']
                        or trial['rate'] != method['sample_rate']
                        or trial['fp_per_call']):
                    raise ValueError(f"{row['plugin']}: inconsistent reference/settings")
    matrix, mismatches, counts = [], [], dict(exact=0, within_tolerance=0, diverges=0, not_comparable=0)
    ratio = lambda slug: next(c['speedup_median'] for c in rows[slug]['trials'] if c['block_size'] == 512)
    highlights = (
        f"LLVM is not uniformly faster. At 512 frames, the measured ratios are "
        f"{ratio('joep_amaranth'):.2f}× for Amaranth, {ratio('joep_filther'):.2f}× for Filther, "
        f"{ratio('joep_tonestacks'):.2f}× for ToneStacks and {ratio('joep_stereomanipulator'):.2f}× "
        f"for StereoManipulator. Wahriffic is {ratio('joep_wahriffic'):.2f}×: "
        f"its LLVM audio path takes {1 / ratio('joep_wahriffic'):.2f} times the WDL time. "
        "These are workload-specific results, not a promise about every setting."
    ) if all(rows[s]['status'] == 'MEASURED' for s in (
        'joep_amaranth', 'joep_filther', 'joep_tonestacks', 'joep_stereomanipulator', 'joep_wahriffic')) else ''
    stack_note = method.get('protosynth_fixture_note', '')
    for spec in specs:
        row = rows[spec.slug]
        section = f"Measured {report['date']} on {method['cpu']} (Windows x64), at 48 kHz with default controls. "
        section += f"{method['trials']} serial trials per buffer, {method['seconds_per_trial']:g} seconds of generated audio after one second of warmup, alternating engine order. "
        section += 'The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.\n\n'
        section += 'These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin\'s complete callback time. No algorithm simplification was made.\n\n'
        cases = row.get('trials', [])
        if row['status'] != 'MEASURED' or {case['block_size'] for case in cases} != {64, 512}:
            counts['not_comparable'] += 1
            section += '**No validated speedup claim.** The native comparison did not complete: ' + row.get('error', 'Incomplete buffer coverage').replace('\n', ' ') + '\n\n'
            matrix.append(f'| [{spec.key}](../plugins/JoepVanlier/{spec.key}/README.md) | — | — | Incomplete comparison |')
        else:
            status = 'DIVERGES' if any(c['status'] == 'DIVERGES' for c in cases) else ('WITHIN_TOLERANCE' if any(c['status'] == 'WITHIN_TOLERANCE' for c in cases) else 'EXACT')
            counts[{'DIVERGES': 'diverges', 'WITHIN_TOLERANCE': 'within_tolerance', 'EXACT': 'exact'}[status]] += 1
            if status == 'DIVERGES':
                mismatches.append(spec.key)
                section += '**No validated speedup claim: output differs from native WDL.** The raw timings below are diagnostic only; they do not demonstrate an equivalent faster implementation. '
                section += f"Maximum absolute float-output error: {max(c['max_error'] for c in cases):.6g}; maximum relative RMS error: {max(c['relative_rms_error'] for c in cases):.6g}; MIDI differences: {max(c['midi_differences'] for c in cases)}. The cause is not resolved by this documentation/timing audit.\n\n"
            elif status == 'EXACT':
                section += '**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.\n\n'
            else:
                section += f"**Observed output: within 2e-7 absolute float error, with matching MIDI.** Not bit-identical; maximum error {max(c['max_error'] for c in cases):.6g}.\n\n"
            section += '| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |\n| --- | --- | --- | --- | --- |\n'
            for case in sorted(cases, key=lambda c: c['block_size']):
                section += f"| {case['block_size']} | {case['wdl_us_per_sample']:.4f} | {case['llvm_us_per_sample']:.4f} | {case['speedup_median']:.2f}× | {case['speedup_min']:.2f}–{case['speedup_max']:.2f}× |\n"
            section += '\nA frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.\n\n'
            if spec.slug in GFX:
                section += '**Graphics-dependent workload:** this omits the analysis/display or simulation in `@gfx`. No complete analyzer or interactive-effect speedup is established.\n\n'
            if spec.slug in LIMITED:
                section += '**Workload limit:** ' + LIMITED[spec.slug] + '\n\n'
            by_block = {c['block_size']: c for c in cases}
            note = {'EXACT': 'Exact audio/MIDI', 'WITHIN_TOLERANCE': 'Within tolerance', 'DIVERGES': '**Output mismatch; no speedup claim**'}[status]
            if spec.slug in GFX:
                note += '; GFX excluded'
            if spec.slug in LIMITED:
                note += '; baseline only'
            matrix.append(f"| [{spec.key}](../plugins/JoepVanlier/{spec.key}/README.md) | {by_block[64]['speedup_median']:.2f}× | {by_block[512]['speedup_median']:.2f}× | {note} |")
        section += 'See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.\n\n'
        text = read_plugin_readme(spec.readme_path)
        before, after = text.split('## Performance comparison\n\n', 1)
        after = after.split('## Attribution and source\n\n', 1)[1]
        spec.readme_path.write_text(before + '## Performance comparison\n\n' + section + '## Attribution and source\n\n' + after, encoding='utf-8', newline='\n')
    destination = root / 'docs/validation/joep-performance'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    mismatch_note = (
        'Output discrepancies against native WDL (' + ', '.join(mismatches) + ') are recorded rather than hidden behind their timings. '
        "The earlier Linux qualification used WDL's portable backend, so an exact result there is not evidence of equivalence to this native backend. "
        'This audit does not resolve which backend/semantic detail causes a discrepancy.'
    ) if mismatches else 'No audio/MIDI discrepancy was observed in this workload.'
    text = f'''# JoepVanlier: native WDL/EEL2 JIT versus LLVM DSP

This {report['date']} comparison covers all {len(specs)} buildable JoepVanlier entries without simplifying their JSFX algorithms. It compares the vendored **native x64 SSE WDL JIT** to the LLVM DSP kernel used by the native JUCE plugins. It does **not** compare complete REAPER and JUCE host callbacks.

{counts['exact']} entries matched float audio and MIDI exactly in all measured trials; {counts['within_tolerance']} were within tolerance; {counts['diverges']} differed; {counts['not_comparable']} did not complete a comparable measurement. A mismatch is a correctness finding, not a validated performance improvement.

## Method

- Windows x64, {method['cpu']}; {method['compiler']}.
- 48 kHz; 64- and 512-frame buffers; default source controls.
- {method['trials']} trials per buffer, {method['seconds_per_trial']:g} seconds measured after one second of warmup.
- Serial, below-normal priority; alternating engine order per block and trial.
- Identical continuous synthetic input and short MIDI events; no external audio files loaded.
- Cached optimized LLVM IR is checked against current expanded JSFX/imports and re-emitted with `-O2 -Xclang -disable-llvm-passes`. This retains the existing IR optimizer/inliner decisions and optimizes final machine code without adding fast-math.
- WDL uses NOFPSTATE with a scoped FP environment, avoiding a floating-point environment transition per sample. LLVM calls its block entry point; WDL runs its original block/sample sections with required channel marshalling.
- Timed: the DSP audio sections and required sample/buffer marshalling. Excluded: compilation, initialization, generated signal construction, transport/MIDI/control setup, output checks, GFX, JUCE host callback and DAW.
- Timing uses C++ `steady_clock` elapsed durations, including any scheduling interruption; the raw `*_cpu_seconds` field names do not mean process CPU accounting. Paired order and trial ranges help expose measurement variability.

The ratio is the median of paired **WDL time / LLVM time**. Above 1 favours LLVM; below 1 favours WDL. Leaf READMEs report microseconds per frame and the trial range. Close ratios should be treated as similar performance, not a universal speed claim. Tiny kernels are more sensitive to timer/marshalling overhead.

## Results

| Plugin | Ratio, 64 frames | Ratio, 512 frames | Output and scope |
| --- | --- | --- | --- |
''' + '\n'.join(matrix) + f'''

## Interpretation and limits

Exact matching means the observed finite float values and short MIDI messages matched for these inputs and defaults; positive and negative zero are not distinguished. It is not qualification of every preset, rate, sample bank, automation path, custom serialization, interactive state or DAW. Silent output and empty default patterns/slots provide limited active-workload evidence. The arpeggiator fixture creates the same four active steps in both guests so MIDI generation is actually exercised.

The spectral analyzers and ReaBee do substantial work or drive audio state from `@gfx`; that work is absent here. Their ratios describe the measured audio path only. Bric-a-brac has no samples loaded, SEQS is at its default playback/pattern state, and Ripple's default pattern is empty. Instruments receive short note events, rather than an exhaustive maximum-polyphony arrangement.

{mismatch_note}

{highlights}

The previously reported **12.9× Amaranth complete-callback improvement** compares our old and optimized production callbacks. It includes removed host-variable lookup overhead. It is **not** a WDL speedup and is not interchangeable with this kernel comparison. Amaranth's source algorithm is unchanged.

{stack_note}

## Evidence and reproduction

[All raw trials and fingerprints](validation/joep-performance/results.json) are retained in the repository. The runner is `tests/jsfx_showcase/benchmark_joep_native.py`, using `joep_native_benchmark.cpp` and the original shared numeric/slider/runtime helpers. It requires Windows x64 Clang, cached regular build IR/headers/source, and a native WDL static library (including the SSE assembly from `src/WDL/eel2/asm-nseel-x64-sse.asm`). A portable WDL library must not be substituted.

```text
python tests/jsfx_showcase/benchmark_joep_native.py --out build/joep-readme-performance --wdl-lib <native-wdl.lib> --trials 5 --seconds 4
python scripts/update_joep_performance.py build/joep-readme-performance/results.json
python scripts/check_plugin_readmes.py
```

The publication script requires complete, disjoint coverage of every JoepVanlier entry; it will not publish a partial matrix as the complete catalog. Documentation is embedded when plugins build. Updating the source README does not replace help in already installed binaries.
'''
    (root / 'docs/Joep-Performance.md').write_text(text, encoding='utf-8', newline='\n')
    print('Published', len(specs), 'per-plugin comparisons:', counts)


if __name__ == '__main__':
    main()
