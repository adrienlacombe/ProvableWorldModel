#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a release: bump the workspace version and roll the changelog.

Automates the mechanical, error-prone parts of cutting a release
(``docs/legacy-stark/spec/09-release-and-versioning.md``):

1. Validates the target version is semver-conformant and a forward bump.
2. Sets ``[workspace.package].version`` and every ``[workspace.dependencies]``
   first-party ``pwm-*`` version in the root ``Cargo.toml`` to the new version.
3. Renames the ``## [Unreleased]`` changelog section to ``## [<version>] - <date>``
   and opens a fresh, empty ``## [Unreleased]`` above it.

It does NOT commit, tag, or push — it prepares the tree for review. After
running, review the diff, run the gates, commit, and tag ``v<version>``; the
release workflow validates and packages on the tag.

Usage:
    prepare-release.py <version> [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT_CARGO = Path("Cargo.toml")
CHANGELOG = Path("CHANGELOG.md")

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
FIRST_PARTY = ["pwm-core", "pwm-export", "pwm-prover", "pwm-verifier", "pwm-testkit"]


def parse_core(version: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    assert m is not None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def current_workspace_version(cargo: str) -> str:
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', cargo)
    if not m:
        raise SystemExit("error: could not find [workspace.package] version in Cargo.toml")
    return m.group(1)


def bump_cargo(cargo: str, version: str) -> str:
    # [workspace.package] version (the first top-level `version = "..."`).
    cargo, n = re.subn(r'(?m)^version = "[^"]+"', f'version = "{version}"', cargo, count=1)
    if n != 1:
        raise SystemExit("error: failed to update [workspace.package] version")
    # Each first-party [workspace.dependencies] entry's version field.
    for crate in FIRST_PARTY:
        pattern = re.compile(
            rf'(?m)^({re.escape(crate)} = \{{ path = "[^"]+", version = ")[^"]+(" \}})'
        )
        cargo, c = pattern.subn(rf"\g<1>{version}\g<2>", cargo)
        if c != 1:
            raise SystemExit(f"error: failed to update workspace dependency version for {crate}")
    return cargo


def roll_changelog(text: str, version: str, date: str) -> str:
    if f"## [{version}]" in text:
        raise SystemExit(f"error: CHANGELOG already has a '## [{version}]' section")
    marker = "## [Unreleased]"
    if marker not in text:
        raise SystemExit("error: CHANGELOG has no '## [Unreleased]' section to roll")
    replacement = f"## [Unreleased]\n\n## [{version}] - {date}"
    return text.replace(marker, replacement, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a release tree.")
    parser.add_argument("version", help="target version, e.g. 0.2.0 (no leading v)")
    parser.add_argument("--date", help="release date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    version = args.version[1:] if args.version.startswith("v") else args.version
    if not SEMVER_RE.match(version):
        raise SystemExit(f"error: '{version}' is not a semver version (MAJOR.MINOR.PATCH)")

    if args.date:
        date = args.date
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise SystemExit("error: --date must be YYYY-MM-DD")
    else:
        # Imported lazily so the failure path above does not depend on the clock.
        from datetime import date as _date

        date = _date.today().isoformat()

    cargo = ROOT_CARGO.read_text(encoding="utf-8")
    old = current_workspace_version(cargo)
    if parse_core(version) <= parse_core(old) and version == old:
        raise SystemExit(f"error: {version} is not a forward bump from {old}")

    ROOT_CARGO.write_text(bump_cargo(cargo, version), encoding="utf-8")
    CHANGELOG.write_text(
        roll_changelog(CHANGELOG.read_text(encoding="utf-8"), version, date), encoding="utf-8"
    )

    print(f"prepared release {old} -> {version} ({date})")
    print("next steps:")
    print("  1. review the diff and the new changelog section")
    print("  2. run the gates: cargo build && cargo test --workspace && bash ci/check-spdx.sh")
    print(f"  3. python3 ci/check-changelog.py {version}")
    print(f"  4. commit, then tag: git tag v{version} && git push origin v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
