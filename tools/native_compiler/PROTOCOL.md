# Native frontend protocol v1

This protocol covers source resolution, section extraction, tokens and unlowered syntax trees. It does not compile DSP, validate builtin services, infer pins/sliders, specialize receivers, plan Faust execution, optimize or emit LLVM. A successful frontend result is not a successful plugin compilation. Both recursive parser calls and AST depth are bounded at 512, including long left-associated expressions that do not recurse deeply during parsing.

The library entry point is `za::compiler::frontend(request, cancellation)`. The helper accepts two absolute paths: `jsfx_frontend request.json result.json`. It returns exit 0 for success, 1 for a structured frontend failure, and 2 for invalid command arguments, unreadable/malformed request files or failed result-file writes. Compile away from the audio thread. The Windows helper reserves 8 MiB of stack; direct library callers must provide a compiler thread with that stack budget. Source and expanded text are limited to 4 MiB, import chains to 32 units, recursive parser calls to 512. These are implementation resource limits, not new JSFX language rules.

## Request

```json
{
  "protocol": 1,
  "stage": "frontend",
  "source": "@sample\nspl0 *= .5;\n",
  "resolve": true,
  "sourcePath": "D:/package/effect.jsfx",
  "packageRoot": "D:/package",
  "searchRoots": [],
  "snippet": false
}
```

`source` is authoritative, allowing edited text with an existing or not-yet-saved origin. `resolve` defaults to false. When true, `sourcePath` is required; the package root defaults to its parent. Additional search roots are explicit, not automatically the whole repository. Imports execute in postorder with diamond deduplication. Root sections override imported fallbacks, including empty root sections. Imported `@init` bodies concatenate in postorder. Configuration defaults and WDL preprocessing execute per unit. Windows preprocessing preserves the reference helper's text-mode CRLF expansion.

Unknown request fields and incorrectly typed options fail with a protocol diagnostic. Optional flags must be booleans, paths nonempty strings, search roots an array of nonempty strings, and `baseLine` a positive 32-bit integer. A new compilation stage/options schema requires an explicit protocol extension; v1 does not silently ignore purported compiler settings.

`snippet: true` parses one EEL expression/statement stream instead of extracting JSFX sections. `baseLine` defaults to 1 and controls its first source line. JSFX section extraction uses the production compiler's current rules: repeated section bodies concatenate; the resolver handles mixed Faust stages before extraction. Faust bodies are returned intact with `opaque: true`; they are not parsed as EEL or validated as Faust.

## Result

Every library result includes `protocol: 1`, `backend: "cpp-experimental"`, `stage: "frontend"`, and boolean `ok`. Successful results add:

- `resolution`: exact expanded `text`, postorder absolute `dependencies`, and `sectionSources`, mapping each section to its contributing source units.
- `sections`: ordered objects with `name`, exact `source`, first `line`, and either `tokens`/`ast` or `opaque: true` for Faust.
- `phaseSeconds`: resolution and lexer/parser durations, excluding helper startup and JSON serialization.

Tokens have `kind`, `text`, and `{line, col}` spans. Decoded string token text is **lowercase hexadecimal UTF-8 bytes**; all other token text is the production lexer spelling. This avoids losing embedded NULs in native strings. AST nodes have the production dataclass type name in `type`, its span, and the corresponding named fields. Ephemeral node IDs are omitted. `Num.value` is exactly 16 lowercase hexadecimal digits representing the IEEE binary64 bits in big-endian order. `StrLit.value` uses the same UTF-8 hex encoding as string tokens. Optional nodes are null; arrays retain evaluation order. Function/call cell-parameter fields remain present even when empty, allowing a later lowering phase to use the same schema.

Failures add `diagnostics`, an ordered array with `severity`, `phase`, `message`, `file`, `line`, `column`. Lexer/parser spans are positions in the **expanded section stream**, not an exact remapping to the original imported file. `sectionSources` supplies coarse provenance; precise source maps remain later work. Resolver messages include the offending file/import chain. Unknown locations use 0. Native failures never invoke Python as a fallback.

Cancellation is cooperative between units and syntax operations. WDL preprocessing is synchronous and cannot be interrupted by that flag; an invoking process manager must impose a timeout and terminate the helper for a stuck preprocessing program. The editor's existing isolated compiler process remains the model for later integration. The library serializes WDL resolution calls, because its host hooks assume one preprocessing operation at a time. Independent parsing calls can run concurrently. This is not a sandbox for untrusted code.

## Compatibility gate

`Reference.py` invokes the existing production Python classes unchanged. `Check.py` compares exact expanded text, dependency and section ownership order, token spelling and spans, and normalized AST fields. Syntax errors compare phase, primary message and location. Resolver errors currently compare error categories, because their search-detail formatting differs. Timing fields and backend labels are not equivalence fields.

Run `Check.py --all-plugins` to discover every JSFX entry through `plugin.json`; no pre-existing build report is required. `--freeze-reference file.zip` captures relocation-normalized requests and expected frontend results, with hashes of the actual production sources. Generated fixtures use a fixed seed. Update a frozen reference deliberately when production semantics change; do not silently accept a native mismatch.
