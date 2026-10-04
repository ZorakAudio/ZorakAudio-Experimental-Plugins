"""Audit full Sample and extract its pure drawing helpers for native/EEL parity.

The generated scene is a renderer qualification, not a port of Sample's @gfx.
The drawing scene excludes input and publications; the full source has an explicit native contract.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsp_jsfx_aot as compiler


def children(value):
    if isinstance(value, compiler.Node):
        return vars(value).values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def audit(source):
    pipeline = compiler.prepare_jsfx_pipeline(source, include_gfx=True)
    functions = pipeline['fn_defs']
    reachable = set()
    calls = defaultdict(list)
    indexes = []

    def walk(node, context='gfx', writing=False):
        if isinstance(node, compiler.Call):
            if node.fn in functions:
                if node.fn not in reachable:
                    reachable.add(node.fn)
                    walk(functions[node.fn].body, node.fn)
            else:
                calls[node.fn].append({'line': node.span.line, 'args': len(node.args), 'context': context})
        if isinstance(node, compiler.Index):
            indexes.append({'line': node.span.line, 'write': writing,
                            'expression': compiler._node_to_jsfx_text(node), 'context': context})
        if isinstance(node, compiler.Assign):
            walk(node.target, context, True)
            walk(node.value, context)
            return
        for child in children(node):
            walk(child, context)

    walk(pipeline['programs']['gfx'])
    usage = compiler.analyze_gfx_var_sync(source, compiler.collect_user_vars(pipeline['programs'], functions))
    contract = compiler.parse_native_gfx_contract(source, compiler.collect_user_vars(pipeline['programs'], functions))
    compiler.validate_native_gfx_prototype(pipeline['programs'], functions, contract)
    gate = None
    return {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'reachable_helper_count': len(reachable), 'builtins': dict(sorted(calls.items())),
            'indexed_access': indexes,
            'scalar_write_candidates': sorted(usage['gfx_writes'] & (usage['audio_reads'] | usage['audio_writes'])),
            'first_native_rejection': gate}


def drawing_fixture(source):
    sections = compiler.extract_sections(source)
    programs = {name: compiler.Parser(text, base_line=line).parse_program()
                for name, (text, line) in sections.items() if name in ('init', 'slider', 'block', 'sample', 'gfx')}
    functions, _ = compiler.extract_function_defs(programs)
    selected = set()

    def add(name):
        if name in selected:
            return
        selected.add(name)

        def walk(node):
            if isinstance(node, compiler.Call) and node.fn in functions:
                add(node.fn)
            for child in children(node):
                walk(child)
        walk(functions[name].body)

    for name in ('ui_panel', 'ui_inner_panel', 'ui_led', 'ui_hmeter',
                 'ui_draw_header_logo', 'draw_curve_line', 'draw_center_buf'):
        add(name)
    def original_definition(name):
        # Preserve literal spelling and definition order for portable EEL.
        # The compiler's debug AST printer may use scientific notation, which
        # the portable EEL parser does not accept. Scan to the outer semicolon,
        # ignoring comments and quoted text when balancing parentheses.
        match = re.search(r'^\s*function\s+' + re.escape(name) + r'\s*\(', source,
                          re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError('Missing original helper: ' + name)
        start = match.start()
        i, depth, mode = start, 0, ''
        while i < len(source):
            ch = source[i]
            pair = source[i:i+2]
            if mode == 'line':
                if ch == '\n': mode = ''
            elif mode == 'block':
                if pair == '*/': mode = ''; i += 1
            elif mode in ('"', "'"):
                if ch == '\\': i += 1
                elif ch == mode: mode = ''
            elif pair == '//': mode = 'line'; i += 1
            elif pair == '/*': mode = 'block'; i += 1
            elif ch in ('"', "'"): mode = ch
            elif ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif ch == ';' and depth == 0:
                return source[start:i+1].strip()
            i += 1
        raise ValueError('Unterminated original helper: ' + name)

    definitions = '\n'.join(original_definition(name)
                            for name in sorted(selected, key=lambda name: functions[name].span.line))
    # Only ENV_VIS_SHAPE is an external constant read by these pure helpers.
    # Copy its original assignment rather than inventing an initialization rule.
    constant = re.search(r'^ENV_VIS_SHAPE\s*=.*?;', source, re.MULTILINE).group()
    scene = r'''
gfx_set(0.025,0.033,0.046,1);gfx_rect(0,0,gfx_w,gfx_h);
ui_panel(10,10,gfx_w-20,84);
ui_draw_header_logo(54,52,23);
gfx_setfont(1,"Arial",25,0);gfx_x=98;gfx_y=29;
gfx_set(0.92,0.97,1,1);gfx_drawstr("SAMPLE / NATIVE DRAWING QUALIFICATION");
gfx_setfont(1,"Arial",14,0);gfx_x=98;gfx_y=65;
gfx_set(0.63,0.72,0.84,1);gfx_drawstr("Original Sample helpers; input, DSP state and bank loading are excluded.");
ui_panel(10,104,gfx_w*0.29-15,gfx_h-120);
ui_panel(gfx_w*0.29+5,104,gfx_w*0.42-15,gfx_h-120);
ui_panel(gfx_w*0.71+5,104,gfx_w*0.29-15,gfx_h-120);
gfx_setfont(1,"Arial",16,0);gfx_set(0.88,0.94,1,1);
gfx_x=28;gfx_y=120;gfx_drawstr("METERS / LINE GRADIENTS");
ui_led(32,172,1,0.18,0.90,0.44);
ui_hmeter(28,214,gfx_w*0.29-52,14,0.62,0.16,0.62,1);
ui_hmeter(28,280,gfx_w*0.29-52,14,0.36,0.20,0.86,0.58);
ui_hmeter(28,346,gfx_w*0.29-52,14,0.82,0.64,0.40,1);
gfx_setfont(1,"Arial",14,0);gfx_set(0.74,0.80,0.88,1);
gfx_x=28;gfx_y=190;gfx_drawstr("Pitch confidence");
gfx_x=28;gfx_y=256;gfx_drawstr("Average brightness");
gfx_x=28;gfx_y=322;gfx_drawstr("Average attack focus");
gx=gfx_w*0.29+22;gy=168;gw=gfx_w*0.42-50;gh=gfx_h-220;
ui_inner_panel(gx,gy,gw,gh);
gfx_setfont(1,"Arial",16,0);gfx_set(0.88,0.94,1,1);
gfx_x=gx;gfx_y=120;gfx_drawstr("ENVELOPE / ORIGINAL CURVE HELPERS");
gfx_set(0.36,0.78,1,1);
draw_curve_line(gx+14,gy+gh-18,gx+gw*0.2,gy+18,100);
gfx_line(gx+gw*0.2,gy+18,gx+gw*0.4,gy+18);
draw_curve_line(gx+gw*0.4,gy+18,gx+gw*0.65,gy+gh*0.55,-65);
gfx_line(gx+gw*0.65,gy+gh*0.55,gx+gw*0.8,gy+gh*0.55);
draw_curve_line(gx+gw*0.8,gy+gh*0.55,gx+gw-14,gy+gh-18,-100);
gfx_setfont(1,"Arial",14,0);sprintf(#ui_buf,"Measured / centered: %.0f x %.0f",gw,gh);
draw_center_buf(gx,gy+gh+12,gw);
tx=gfx_w*0.71+24;tw=gfx_w*0.29-50;
gfx_setfont(1,"Arial",16,0);gfx_set(0.88,0.94,1,1);
gfx_x=tx;gfx_y=120;gfx_drawstr("TEXT OUTPUT LVALUES / AA");
gfx_setfont(1,"Arial",14,0);sprintf(#ui_buf,"First line\nSecond line is wider");
gfx_measurestr(#ui_buf,mw,mh);
gfx_rect(tx,168,mw,mh,0);gfx_x=tx;gfx_y=168;gfx_drawstr(#ui_buf);
gfx_measurestr(#ui_buf);gfx_measurestr(#ui_buf,only_width);
gfx_measurestr(#ui_buf,alias,alias);
sprintf(#ui_buf,"Width %.2f | Height %.2f",only_width,alias);draw_center_buf(tx,236,tw);
gfx_setfont(0);sprintf(#ui_buf,"Bitmap\nfont metrics");gfx_measurestr(#ui_buf,bw,bh);
gfx_rect(tx,280,bw,bh,0);gfx_x=tx;gfx_y=280;gfx_drawstr(#ui_buf);
gfx_line(tx+0.9,340.8,tx+tw-0.1,370.9,0);
gfx_line(tx+0.9,380.8,tx+tw-0.1,410.9,0.5);
gfx_line(tx+0.9,420.8,tx+tw-0.1,450.9);
'''
    return ('desc:Sample native drawing qualification (not the full plugin)\n@init\n' + definitions
            + '\n' + constant + '\n@gfx 1380 760\n' + scene), sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'plugins/Spectral/Sample/src/Sample.jsfx')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    report = audit(source)
    fixture, helpers = drawing_fixture(source)
    compiler.compile_jsfx_to_ir(fixture, native_gfx_prototype=True)
    report['drawing_fixture_helpers'] = helpers
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sample-gfx-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'SampleDrawingQualification.jsfx').write_text(fixture)
    print("Full Sample: native contract validated")
    print(f"Audit: {report['reachable_helper_count']} helpers, {len(report['indexed_access'])} indexed access sites")
    print(f"Drawing fixture: {len(helpers)} original helpers; {args.out}")


if __name__ == '__main__':
    main()
