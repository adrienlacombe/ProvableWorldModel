#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Changelog and semver gate for the release path.

Enforces the changelog discipline from
``docs/spec/09-release-and-versioning.md#changelog`` (INV-REL-10): no release
tag is cut without a matching, non-empty ``CHANGELOG.md`` section whose version
equals the tag, and the version must be semver-conformant.

Usage:
    check-changelog.py <version>            # validate the release section exists
    check-changelog.py --extract <version>  # print that section's body (release notes)

``<version>`` may be given with or without a leading ``v``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")

# Semantic Versioning 2.0.0 (the official regex, simplified to the fields we use).
SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)


def normalize_version(raw: str) -> str:
    return raw[1:] if raw.startswith("v") else raw


def is_semver(version: str) -> bool:
    return SEMVER_RE.match(version) is not None


def find_section(text: str, version: str) -> str | None:
    """Return the body of the ``## [<version>]`` section, or None if absent.

    The body runs from the heading to the next ``## `` heading (or EOF), with the
    heading line and the trailing link-reference lines excluded.
    """
    lines = text.splitlines()
    heading = re.compile(rf"^##\s+\[{re.escape(version)}\](?:\s+-\s+\S+)?\s*$")
    start = None
    for i, line in enumerate(lines):
        if heading.match(line):
            start = i
            break
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip("\n")


def has_content(body: str) -> bool:
    """True if the section carries at least one entry (a bullet under a section)."""
    return any(line.strip().startswith("-") for line in body.splitlines())


def main(argv: list[str]) -> int:
    extract = False
    args = argv[1:]
    if args and args[0] == "--extract":
        extract = True
        args = args[1:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2

    version = normalize_version(args[0])

    if not is_semver(version):
        print(f"error: '{version}' is not a semver version (MAJOR.MINOR.PATCH)", file=sys.stderr)
        return 1
    if version.lower() == "unreleased":
        print("error: cannot release the 'Unreleased' section; give a concrete version", file=sys.stderr)
        return 1
    if not CHANGELOG.exists():
        print(f"error: {CHANGELOG} not found", file=sys.stderr)
        return 1

    text = CHANGELOG.read_text(encoding="utf-8")
    body = find_section(text, version)
    if body is None:
        print(
            f"error: no '## [{version}]' section in {CHANGELOG} (INV-REL-10: "
            f"a release tag requires a matching changelog section)",
            file=sys.stderr,
        )
        return 1
    if not has_content(body):
        print(f"error: the '## [{version}]' section is empty (INV-REL-10)", file=sys.stderr)
        return 1

    if extract:
        print(body)
    else:
        print(f"changelog check passed: '## [{version}]' is present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
