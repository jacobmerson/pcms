#!/usr/bin/env python3
"""Merge the JSON-line output of pcms-refactor runs and apply it.

Usage: apply_edits.py --mode qualify|spell --root <repo> <edits.jsonl>...

The same header is parsed by many translation units, so identical edits are
merged. In spell mode, different spellings for the same qualifier (the
shortest unambiguous spelling can depend on what a translation unit included
before the header) resolve to the longest one, which is valid everywhere.
Any other conflict is an error.
"""
import argparse
import collections
import json
import os
import re
import sys

BLOCK = re.compile(rb"\nnamespace pcms\n\{\n((?:using [\w:]+;\n)+)\} // namespace pcms\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("qualify", "spell"))
    ap.add_argument("--root", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    edits = collections.defaultdict(lambda: collections.defaultdict(set))
    usings = collections.defaultdict(set)
    insert_at = {}
    notes, errors = set(), 0
    for path in args.inputs:
        for line in open(path):
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            if "note" in rec:
                notes.add(f"{rec['at']}: {rec['note']}")
            elif "error" in rec:
                errors += 1
            elif "usings" in rec:
                usings[rec["file"]].update(rec["usings"])
                insert_at[rec["file"]] = rec["insert"]
            else:
                edits[rec["file"]][(rec["offset"], rec["length"])].add(rec["text"])
    for n in sorted(notes):
        print("note:", n)
    if errors:
        sys.exit(f"{errors} translation unit(s) did not parse; nothing applied")

    changed = 0
    for f in sorted(set(edits) | set(usings)):
        path = os.path.join(args.root, f)
        text = open(path, "rb").read()  # offsets are byte offsets
        flat = []
        for (off, length), texts in edits[f].items():
            if len(texts) > 1:
                if args.mode != "spell":
                    sys.exit(f"{f}:{off}: conflicting edits {sorted(texts)}")
                texts = {max(texts, key=len)}
            flat.append((off, length, texts.pop().encode()))
        if f in usings:
            m = BLOCK.search(text)
            if m:
                have = set(re.findall(rb"using ([\w:]+);", m.group(1)).__iter__())
                have = {h.decode() for h in have}
                body = "".join(f"using {n};\n" for n in sorted(have | usings[f]))
                flat.append((m.start(1), m.end(1) - m.start(1), body.encode()))
            else:
                block = ("\nnamespace pcms\n{\n"
                         + "".join(f"using {n};\n" for n in sorted(usings[f]))
                         + "} // namespace pcms\n")
                flat.append((insert_at[f], 0, block.encode()))
        flat.sort()
        # insertions at the same offset keep their order; overlaps are errors
        end = -1
        for off, length, _ in flat:
            if off < end:
                sys.exit(f"{f}:{off}: overlapping edits")
            end = max(end, off + length)
        for off, length, new in reversed(flat):
            text = text[:off] + new + text[off + length:]
        open(path, "wb").write(text)
        changed += 1
    print(f"{args.mode}: updated {changed} files")


if __name__ == "__main__":
    main()
