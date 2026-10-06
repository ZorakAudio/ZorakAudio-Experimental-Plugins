"""Interleaved complete-plugin qualification. Loads only the authorized FLAC."""
from pathlib import Path
import array
import json
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/faust-sections/sample-character'
RECORDING = r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
rows = []
routes = '--routes' in sys.argv
plain = '--plain' in sys.argv
for rate, block in ([(48000, 256), (96000, 1024)] if routes else [(48000, 64), (48000, 256), (96000, 1024)]):
    times = {'cache': [], 'faust': []}
    trials = 1 if routes or plain else 3
    for trial in range(trials):
        paths = {}
        for label in (['cache', 'faust'] if trial % 2 == 0 else ['faust', 'cache']):
            paths[label] = BASE / f'host-{label}' / f'{rate}-{block}-{trial}{"-routes" if routes else "-plain" if plain else ""}.json'
            subprocess.run([str(BASE / f'host-{label}' / 'host.exe'), 'Sample', RECORDING,
                            str(paths[label]), str(rate), str(block),
                            *([] if plain else ['--character-routes' if routes else '--character'])], check=True)
            result = json.loads(paths[label].read_text())
            times[label].append(result['process_seconds'])
            if label == 'faust':
                assert result['faust_scalars'] == 0
                assert result['faust_blocks'] <= result['callbacks'] * ((block+255)//256)
                assert result['character_fast_frames'] == result['faust_blocks'] * min(block,256)
                if plain:
                    assert result['character_fast_frames'] == 0
                else:
                    assert result['character_fast_frames'] > 0, 'Block path never activated'
        audio = {}
        for label, path in paths.items():
            audio[label] = array.array('f')
            audio[label].frombytes(Path(str(path) + '.pcm.bin').read_bytes())
        assert len(audio['cache']) == len(audio['faust'])
        maximum = max(abs(x-y) for x,y in zip(audio['cache'], audio['faust']))
        # Full output includes extremely tiny denormal-tail differences; reject
        # a perceptibly material discrepancy rather than claiming bit-null.
        assert maximum < 1e-7, (rate, block, maximum)
        rows.append({'rate': rate, 'block': block, 'trial': trial, 'routes': routes, 'plain': plain,
                     'samples': len(audio['cache']), 'maximum_error': maximum,
                     'cache': json.loads(paths['cache'].read_text()),
                     'faust': json.loads(paths['faust'].read_text())})
        print(f'PASS {rate}/{block} trial {trial} max error {maximum:.3g}', flush=True)
    print(f'{rate}/{block}: cache {statistics.median(times["cache"]):.6f}, '
          f'FAUST {statistics.median(times["faust"]):.6f}', flush=True)
(BASE / ('routes-results.json' if routes else 'plain-results.json' if plain else 'host-results.json')).write_text(json.dumps(rows, indent=2))
