"""Restore pinned pre-migration references and current candidates for the audit tests."""
from pathlib import Path
import json, hashlib, shutil
ROOT=Path(__file__).resolve().parents[2]
fixtures=ROOT/'tests/faust/fixtures/catalog'
entries={
 'ADS':'plugins/Ambience/ADS/src/ADS.jsfx',
 'SaliencePush':'plugins/Spatialization/SaliencePush/src/SP.jsfx',
 'DPT':'plugins/Spatialization/DPT/src/DPT.jsfx',
 'DDT':'plugins/Spatialization/DDT/src/DDT.jsfx',
 'ERBTilt':'plugins/Spectral/ERBTilt/src/ERB Tilt Faust.jsfx',
 'SpectralStabilizer':'plugins/Spectral/SpectralStabilizer/src/Spectral Stabilizer Faust.jsfx',
}
for key,digest in json.loads((fixtures/'manifest.json').read_text()).items():
    source=fixtures/(key+'.jsfx')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest, key+' baseline changed'
    dest=ROOT/'build/catalog-faust-audit'/key;dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest/'baseline.jsfx')
    shutil.copy2(ROOT/entries[key],dest/('sleep-candidate.jsfx' if key in ['ADS','SaliencePush'] else 'candidate.jsfx'))
print('Prepared six pinned references and current candidates; no audio loaded.')
