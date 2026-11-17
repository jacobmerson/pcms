#!/usr/bin/env python3
"""Point callers of the Omega_h-typed helpers at the Omega_h backend.

Usage (from the repository root):
    tools/reorg/omega_h_boundary_calls.py

* MakeConstRank2View(<omega_h array>, <dim>) (the two-argument overload),
  written pcms::, ::pcms:: or unqualified (in namespace pcms)
  -> pcms::backend::omega_h::MakeConstRank2View(...)
* CreateUniformGridFromMesh, written pcms::, ::pcms:: or unqualified
  -> pcms::backend::omega_h::CreateUniformGridFromMesh
and adds #include "pcms/backends/omega_h/omega_h.hpp" to each changed file
that does not include it yet, after its last unconditional #include (or
after its include guard if it has no #include). Running it twice is a no-op.
"""
import re
import subprocess

files = subprocess.run(["git", "ls-files", "src", "test", "examples", "tools"],
                       check=True, capture_output=True, text=True).stdout.split()
OH = "pcms::backend::omega_h::"
rank2 = re.compile(r"(?<![\w:])(?:(?:::)?pcms::)?MakeConstRank2View\s*\(")
grid = re.compile(r"(?<![\w:])(?:(?:::)?pcms::)?CreateUniformGridFromMesh\b")
include = '#include "pcms/backends/omega_h/omega_h.hpp"'


def top_level_args(text, open_paren):
    """Number of top-level arguments of the call whose '(' is at open_paren,
    and the index of its ')'."""
    depth, n, nonempty = 0, 1, False
    for i in range(open_paren, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return (n if nonempty else 0), i
        elif c == "," and depth == 1:
            n += 1
        elif depth >= 1 and not c.isspace():
            nonempty = True
    raise ValueError("unbalanced call")


def rewrite_rank2(text):
    out, pos = [], 0
    for m in rank2.finditer(text):
        if m.start() < pos:
            continue
        nargs, _ = top_level_args(text, m.end() - 1)
        if nargs == 2:
            out.append(text[pos:m.start()])
            out.append(OH + "MakeConstRank2View(")
            pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def add_include(text):
    lines = text.split("\n")
    depth, guard_depth, last, guard_define = 0, 0, None, None
    for i, l in enumerate(lines):
        s = l.strip()
        if re.match(r"#\s*if", s):
            depth += 1
            if guard_define is None and re.match(r"#\s*ifndef\s+\w+", s) and \
                    i + 1 < len(lines) and re.match(r"\s*#\s*define\s+\w+\s*$", lines[i + 1]):
                guard_depth, guard_define = depth, i + 1
        elif re.match(r"#\s*endif", s):
            depth -= 1
        elif s.startswith("#include") and depth <= guard_depth:
            last = i
    if last is None:
        last = guard_define if guard_define is not None else -1
    lines.insert(last + 1, include)
    return "\n".join(lines)


for f in files:
    if not re.search(r"\.(cpp|hpp|h)$", f) or f.startswith("include/") \
            or f == "tools/reorg/omega_h_boundary_calls.py":
        continue
    text = open(f).read()
    new = rewrite_rank2(text)
    new = grid.sub(OH + "CreateUniformGridFromMesh", new)
    new = new.replace("::" + OH, OH).replace(OH + OH, OH)
    if new != text and include not in new:
        new = add_include(new)
    if new != text:
        open(f, "w").write(new)
        print("updated", f)
