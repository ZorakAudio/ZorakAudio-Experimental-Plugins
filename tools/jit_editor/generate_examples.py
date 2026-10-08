"""Embed the documented, tested example sources into the plugin UI."""
from pathlib import Path
import json, sys
folder=Path(__file__).with_name('examples')
examples=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
lines=['#pragma once','struct JitExample { const char* name; const char* mode; const char* source; };','inline constexpr JitExample jitExamples[] = {']
for example in examples:
    source=(folder/example['file']).read_text(encoding='utf-8')
    lines.append('    {'+', '.join(json.dumps(v,ensure_ascii=False) for v in (example['name'],example['mode'],source))+'},')
lines.append('};')
Path(sys.argv[1]).write_text('\n'.join(lines)+'\n',encoding='utf-8')
