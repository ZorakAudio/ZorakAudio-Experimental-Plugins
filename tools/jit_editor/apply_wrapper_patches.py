"""Restore the JIT-only wrapper hooks when configuring a fresh submodule checkout."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
for name,relative in [('juce','libs/JUCE'),('clap','libs/clap-juce-extensions')]:
 root=ROOT/relative;patch=Path(__file__).parent/'wrapper-patches'/(name+'.patch')
 command=['git','-C',str(root),'apply','--whitespace=nowarn']
 if subprocess.run(command+['--reverse','--check',str(patch)],capture_output=True).returncode==0:continue
 checked=subprocess.run(command+['--check',str(patch)],capture_output=True,text=True)
 if checked.returncode:raise SystemExit('JIT wrapper patch conflicts with local changes in '+relative+'; preserve local edits and merge '+str(patch)+' manually.\n'+checked.stderr)
 subprocess.run(command+[str(patch)],check=True)
