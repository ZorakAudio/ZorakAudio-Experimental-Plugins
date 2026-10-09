"""Exercise the actual Linux compiler process supervisor, including parent death."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

LAUNCHER = None

def alive(pid):
    try:
        # A terminated zombie is not executing compiler code.
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False

@unittest.skipUnless(sys.platform == 'linux', 'Linux process supervisor')
class SupervisorTests(unittest.TestCase):
    def test_exit_code_and_argument_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='za-supervisor-') as directory:
            target = Path(directory) / 'argument Ω with spaces.txt'
            code = 'import sys;from pathlib import Path;Path(sys.argv[1]).write_text(sys.argv[2]);raise SystemExit(7)'
            child = subprocess.run([str(LAUNCHER), str(os.getpid()), sys.executable, '-c', code,
                                    str(target), 'literal $() " Ω'], start_new_session=True)
            self.assertEqual(child.returncode, 7)
            self.assertEqual(target.read_text(), 'literal $() " Ω')

    def test_parent_death_stops_worker_and_descendant(self):
        with tempfile.TemporaryDirectory(prefix='za-supervisor-') as directory:
            record = Path(directory) / 'pids.json'
            worker = ('import os,subprocess,json,time;from pathlib import Path;'
                      'p=subprocess.Popen(["/bin/sleep","120"]);'
                      f'Path({str(record)!r}).write_text(json.dumps([os.getpid(),p.pid]));time.sleep(120)')
            parent = ('import os,subprocess,time;from pathlib import Path;'
                      f'p=subprocess.Popen([{str(LAUNCHER)!r},str(os.getpid()),{sys.executable!r},"-c",{worker!r}],start_new_session=True);'
                      f'\nwhile not Path({str(record)!r}).exists():time.sleep(.01)\nos._exit(0)')
            process = subprocess.Popen([sys.executable, '-c', parent])
            pids = []
            try:
                process.wait(timeout=10)
                pids = json.loads(record.read_text())
                deadline = time.monotonic() + 5
                while any(alive(pid) for pid in pids) and time.monotonic() < deadline: time.sleep(.02)
                self.assertFalse(any(alive(pid) for pid in pids), 'Orphan compiler process remained alive')
            finally:
                if process.poll() is None: process.kill(); process.wait()
                for pid in pids:
                    if alive(pid): os.kill(pid, signal.SIGKILL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher', type=Path, required=True)
    args, rest = parser.parse_known_args()
    LAUNCHER = args.launcher.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
