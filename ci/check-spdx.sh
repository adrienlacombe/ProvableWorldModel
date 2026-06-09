#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX header gate. Fails if any first-party source file lacks the
# `SPDX-License-Identifier: Apache-2.0` line in its first few lines.
#
# Scope (docs/legacy-stark/spec/09-release-and-versioning.md, RFC-0015): every first-party
# Rust/Python source file and every first-party Cargo manifest, plus the CI
# shell scripts. Vendored files under `third_party/` retain their upstream
# license headers and are excluded.
set -euo pipefail

spdx='SPDX-License-Identifier: Apache-2.0'
status=0
count=0

# Tracked first-party source files: Rust, Python, Cargo manifests, CI scripts.
# `third_party/` is excluded — vendored code keeps its upstream headers.
# A while-read over a process substitution keeps `status` in this shell (no
# subshell pipe) and stays portable to the macOS bash 3.2 a contributor may run.
while IFS= read -r f; do
  [ -z "$f" ] && continue
  count=$((count + 1))
  if ! head -n 5 -- "$f" | grep -qF "$spdx"; then
    echo "missing SPDX header: $f"
    status=1
  fi
done < <(
  git ls-files \
    | grep -Ev '^third_party/' \
    | grep -E '\.(rs|py)$|(^|/)Cargo\.toml$|^ci/.*\.sh$'
)

if [ "$status" -ne 0 ]; then
  echo "SPDX header check FAILED: add '// $spdx' (or '# $spdx') to the files above."
  exit 1
fi

echo "SPDX header check passed ($count files)."
