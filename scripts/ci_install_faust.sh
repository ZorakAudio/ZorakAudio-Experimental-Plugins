#!/usr/bin/env bash
# Build the pinned upstream compiler natively, before universal2 plugin flags.
# Both required backends are enabled explicitly; no rolling distro Faust.
set -euo pipefail
version=2.81.2
checksum=c91afe17cc01f1f75e4928dc2d2971dd83b37d10be991dda7e8b94ffab1f1ac9
prefix="$RUNNER_TEMP/faust-prefix"
source_root="$RUNNER_TEMP/faust-source"
archive="$RUNNER_TEMP/faust-$version.tar.gz"
curl --fail --location --retry 3 \
  "https://github.com/grame-cncm/faust/releases/download/$version/faust-$version.tar.gz" \
  --output "$archive"
python - "$archive" "$checksum" <<'PY'
import hashlib, sys
from pathlib import Path
actual = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit('Faust source checksum mismatch: ' + actual)
PY
mkdir -p "$source_root"
tar -xzf "$archive" -C "$source_root" --strip-components=1
if [[ "$RUNNER_OS" == macOS ]]; then
  llvm_config="$(brew --prefix llvm@18)/bin/llvm-config"
else
  llvm_config=/usr/bin/llvm-config-18
fi
cmake -S "$source_root/build" -B "$source_root/ci-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCPP_BACKEND=COMPILER -DLLVM_BACKEND=COMPILER \
  -DINCLUDE_LLVM=ON -DLLVM_CONFIG="$llvm_config" -DLINK_LLVM_STATIC=OFF \
  -DINCLUDE_EXECUTABLE=ON -DINCLUDE_STATIC=OFF -DINCLUDE_DYNAMIC=OFF \
  -DINCLUDE_OSC=OFF -DINCLUDE_HTTP=OFF -DINCLUDE_EMCC=OFF -DINCLUDE_WASM_GLUE=OFF
cmake --build "$source_root/ci-build" --target faust --parallel 2
cmake --install "$source_root/ci-build"
echo "$prefix/bin" >> "$GITHUB_PATH"
echo "JSFX_FAUST_COMPILER=$prefix/bin/faust" >> "$GITHUB_ENV"
"$prefix/bin/faust" --version
