#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Internal markdown link checker for the ProvableWorldModel docs corpus.

Scans every tracked Markdown file (outside ``third_party/``), extracts inline
links, and verifies that each *internal* link resolves:

- the target file exists, relative to the linking file, and
- when the link carries a ``#fragment``, that anchor exists in the target
  Markdown file.

Anchors are resolved the way GitHub renders them: from heading text (slugified),
from explicit ``<a id="...">`` / ``<a name="...">`` HTML anchors, and from
``{#explicit-id}`` heading attributes. External links (``http(s)://``,
``mailto:``, ``tel:``) and protocol-relative links are skipped — this gate is
about internal cross-references, not link liveness.

Exit code is non-zero if any internal link is broken, with every broken link
reported (file, line, target, reason).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Inline Markdown link/image target: the (...) part of [text](target) or
# ![alt](target). Captures the raw target (which may include a #fragment and a
# trailing "title").
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Explicit HTML anchors: <a id="x"> / <a name="x">.
HTML_ANCHOR_RE = re.compile(r"""<a\s+(?:id|name)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
# Heading line: leading #s then text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Trailing {#explicit-id} on a heading.
HEADING_ID_RE = re.compile(r"\{#([A-Za-z0-9_-]+)\}\s*$")

SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "//")


def slugify(text: str) -> str:
    """GitHub-style heading slug: lowercase, drop punctuation, spaces to hyphens."""
    text = text.strip().lower()
    # Drop inline code backticks and Markdown emphasis markers before slugging.
    text = text.replace("`", "")
    out = []
    for ch in text:
        if ch.isalnum() or ch in "_- ":
            out.append(ch)
        # all other punctuation is dropped
    slug = "".join(out).strip()
    slug = slug.replace(" ", "-")
    return slug


def collect_anchors(path: Path) -> set[str]:
    """All anchors a link may target within ``path``."""
    anchors: set[str] = set()
    slug_counts: Counter[str] = Counter()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return anchors

    in_code_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            continue
        # Explicit HTML anchors are valid even inside other constructs.
        for m in HTML_ANCHOR_RE.finditer(line):
            anchors.add(m.group(1))
        if in_code_fence:
            continue
        heading = HEADING_RE.match(line)
        if not heading:
            continue
        title = heading.group(2)
        explicit = HEADING_ID_RE.search(title)
        if explicit:
            anchors.add(explicit.group(1))
            title = HEADING_ID_RE.sub("", title).strip()
        # Strip link syntax in heading text: [text](url) -> text.
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
        slug = slugify(title)
        if not slug:
            continue
        # GitHub disambiguates repeated slugs with -1, -2, ...
        n = slug_counts[slug]
        anchors.add(slug if n == 0 else f"{slug}-{n}")
        slug_counts[slug] += 1
    return anchors


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [Path(p) for p in out if not p.startswith("third_party/")]


def strip_title(target: str) -> str:
    """Drop an optional ``"title"`` after the URL in a link target."""
    target = target.strip()
    # A space not inside <> separates URL from title.
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split()[0] if target else target


def main() -> int:
    files = tracked_markdown()
    anchor_cache: dict[Path, set[str]] = {}

    def anchors_for(path: Path) -> set[str]:
        if path not in anchor_cache:
            anchor_cache[path] = collect_anchors(path)
        return anchor_cache[path]

    broken: list[str] = []
    checked = 0

    for md in files:
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        in_code_fence = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue
            for m in LINK_RE.finditer(line):
                raw = strip_title(m.group(1))
                if not raw or raw.startswith(SKIP_PREFIXES):
                    continue
                checked += 1
                path_part, _, fragment = raw.partition("#")

                if path_part == "":
                    # Same-file anchor.
                    target = md
                else:
                    target = (md.parent / path_part).resolve()
                    if not target.exists():
                        broken.append(
                            f"{md}:{lineno}: missing file '{path_part}' (link target '{raw}')"
                        )
                        continue

                if fragment:
                    # Only Markdown files carry resolvable anchors.
                    if target.suffix.lower() != ".md":
                        continue
                    if fragment not in anchors_for(target):
                        broken.append(
                            f"{md}:{lineno}: missing anchor '#{fragment}' in "
                            f"'{path_part or md.name}' (link '{raw}')"
                        )

    if broken:
        print(f"Internal link check FAILED: {len(broken)} broken link(s):", file=sys.stderr)
        for b in sorted(broken):
            print(f"  {b}", file=sys.stderr)
        return 1

    print(f"Internal link check passed: {checked} links across {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
