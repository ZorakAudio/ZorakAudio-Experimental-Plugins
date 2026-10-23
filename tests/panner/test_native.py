"""Check Artistic native controls/state against an already built Windows panner."""
from pathlib import Path
import argparse,subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,default=ROOT/'build/windows/3DPanner')
p.add_argument('--screenshots',type=Path,default=ROOT/'build/panner-proximity/screenshots')
args=p.parse_args();build=args.build.resolve();screens=args.screenshots.resolve()
cache=build/'CMakeCache.txt'
if not cache.exists():raise SystemExit('Build Hyperreal 3D Panner first with scripts/build.py.')
previous=next((line.split('=',1)[1] for line in cache.read_text().splitlines() if line.startswith('CMAKE_PROJECT_INCLUDE:')),None)
try:
    subprocess.run(['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(build),'-DCMAKE_PROJECT_INCLUDE='+str(Path(__file__).parent/'native_probe.cmake')],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','panner_proximity_native','--parallel','4'],check=True)
    subprocess.run([str(build/'panner_proximity_native.exe'),str(screens)],check=True)
finally:
    restore=['-U','CMAKE_PROJECT_INCLUDE'] if previous is None else ['-DCMAKE_PROJECT_INCLUDE='+previous]
    subprocess.run(['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(build),*restore],check=True)
