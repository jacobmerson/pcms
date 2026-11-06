#!/bin/bash
# move_namespaces.sh <tree> <build> <spec>
# Applies every namespace move in <spec> (see namespaces.spec) to <tree> in
# one qualify pass and one spell pass of pcms-refactor (see README.md).
# <build> is a configured build of <tree> with every feature enabled and
# CMAKE_EXPORT_COMPILE_COMMANDS=ON; run this in the environment that build
# uses (the dependencies' headers must be found).
#   PCMS_REFACTOR      the pcms-refactor binary (default: pcms-refactor on PATH)
#   PCMS_REFACTOR_ARGS extra compiler arguments, e.g.
#                      --gcc-install-dir=<dir> when Clang does not find the
#                      GCC the build uses
#   JOBS               parallel tool processes (default 8)
set -euo pipefail
tree=$(realpath "$1"); build=$(realpath "$2"); spec=$(realpath "$3")
here=$(dirname "$(realpath "$0")")
tool=${PCMS_REFACTOR:-pcms-refactor}
jobs=${JOBS:-8}
out=$(mktemp -d)
extra=(--extra-arg=-resource-dir=$(clang -print-resource-dir)
       --extra-arg=-Wno-unknown-warning-option)
for a in ${PCMS_REFACTOR_ARGS:-}; do extra+=(--extra-arg=$a); done
# Spack's compiler wrappers add these include directories implicitly; they
# are not in compile_commands.json
for d in ${SPACK_STORE_INCLUDE_DIRS//:/ }; do extra+=(--extra-arg=-isystem$d); done

# the spec, expanded to `<file> <target>` lines
python3 - "$spec" "$tree" > $out/moves <<'PY'
import os, sys
spec, tree = sys.argv[1], sys.argv[2]
CODE = (".hpp", ".h", ".cpp", ".cc", ".cxx")
moves, excluded, target = {}, set(), None
for n, line in enumerate(open(spec), 1):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("namespace "):
        target = line.split()[1]
        continue
    if target is None:
        sys.exit(f"{spec}:{n}: path before any namespace line")
    if line.startswith("!"):
        excluded.add(line[1:].strip())
        continue
    full = os.path.join(tree, line)
    if line.endswith("/"):
        if not os.path.isdir(full):
            sys.exit(f"{spec}:{n}: no directory {line}")
        files = [os.path.relpath(os.path.join(d, f), tree)
                 for d, _, fs in os.walk(full) for f in fs if f.endswith(CODE)]
    else:
        if not os.path.isfile(full):
            sys.exit(f"{spec}:{n}: no file {line}")
        files = [line]
    for f in files:
        if moves.get(f, target) != target:
            sys.exit(f"{spec}:{n}: {f} is listed for {moves[f]} and {target}")
        moves[f] = target
for f in sorted(moves):
    if f not in excluded:
        print(f, moves[f])
PY
echo "$(wc -l < $out/moves) files move to $(cut -d' ' -f2 $out/moves | sort -u | wc -l) namespaces"

# translation units: the configured ones, plus one per header that nothing
# includes (it would otherwise never be parsed)
mkdir -p $out/db $out/orphans
python3 - "$build/compile_commands.json" "$tree" $out > $out/tus <<'PY'
import json, os, re, sys
db, tree, out = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3]
cmds = json.load(open(db))
tus, seen = [], set()
for c in cmds:
    f = c["file"]
    if f.startswith(tree) and f.endswith((".cpp", ".cc", ".cxx")) and f not in seen:
        seen.add(f); tus.append(c)
headers = []
for top in ("src", "include"):
    for d, _, files in os.walk(tree + top):
        headers += [os.path.join(d, n) for n in files if n.endswith((".hpp", ".h"))]
# a header counts as included if some #include names it: relative to the
# including file, or as a path under one of the include roots
by_path = set(headers)
roots = [tree + r + "/" for r in ("include", "src", "test", "")]
included = set()
inc = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
for d, _, files in os.walk(tree):
    if "/.git" in d:
        continue
    for n in files:
        if n.endswith((".hpp", ".h", ".cpp", ".cc", ".cxx", ".in")):
            try:
                text = open(os.path.join(d, n), errors="replace").read()
            except OSError:
                continue
            for m in inc.finditer(text):
                for base in [d + "/"] + roots:
                    p = os.path.normpath(base + m.group(1))
                    if p in by_path:
                        included.add(p)
richest = max(tus, key=lambda c: c.get("command", " ".join(c.get("arguments", []))).count(" -I"))
extra = []
for h in headers:
    if h in included or "/fortranapi/" in h or "/capi/" in h:
        continue
    src = os.path.join(out, "orphans", h[len(tree):].replace("/", "__") + ".cpp")
    open(src, "w").write(f'#include "{h}"\n')
    c = dict(richest)
    if "command" in c:
        c["command"] = c["command"].replace(richest["file"], src)
    else:
        c["arguments"] = [src if a == richest["file"] else a for a in c["arguments"]]
    c["file"] = src
    extra.append(c)
json.dump(cmds + extra, open(os.path.join(out, "db", "compile_commands.json"), "w"))
for c in tus + extra:
    print(c["file"])
print(f"{len(extra)} header(s) included by nothing get their own translation unit", file=sys.stderr)
PY
echo "$(wc -l < $out/tus) translation units"
run() {  # run <mode> [args]
  local mode=$1; shift
  split -n l/$jobs $out/tus $out/part.$mode.
  local pids=() p pid failed=0
  for p in $out/part.$mode.??; do
    "$tool" --mode=$mode "$@" --root="$tree" -p $out/db "${extra[@]}" $(cat $p) \
      > $p.jsonl 2> $p.err &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait $pid || failed=1
  done
  if [ $failed = 1 ]; then
    cat $out/part.$mode.??.err | grep -vE "^[0-9]+ warnings? generated" | head -40
    echo "pcms-refactor --mode=$mode failed; nothing was changed" >&2
    exit 1
  fi
  cat $out/part.$mode.??.err | grep -E "error:" | sort -u | head -20 || true
  python3 $here/apply_edits.py --mode $mode --root "$tree" $out/part.$mode.??.jsonl
}
run qualify --moves=$out/moves
run spell
dirs=$(cd "$tree" && ls -d src include test examples tools 2>/dev/null)
left=$(cd "$tree" && grep -rln --include='*.cpp' --include='*.hpp' --include='*.h' --exclude=pcms_refactor.cpp '::pcms::' $dirs || true)
if [ -n "$left" ]; then echo "marked qualifiers left in: $left"; exit 1; fi
rm -rf $out
