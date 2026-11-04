#!/usr/bin/env python3
"""Rewrite Python code to the factory-based PCMS Python API (0.4).

Usage (from the repository root):
    tools/reorg/python_api_names.py <file> [<file> ...]

Rewrites, preserving the arguments:

  pcms.LagrangeFunctionSpace.from_mesh(mesh, order, n[, csys][, global_id_name=g][, backend=B])
    -> pcms.create_lagrange_space(pcms.backend.omega_h.create_discretization(mesh[, global_id_name=g]),
                                  order, n[, csys][, implementation=B])
  pcms.LagrangeFunctionSpace.from_uniform_grid(grid, n[, csys][, order])
    -> pcms.create_lagrange_space(pcms.create_uniform_grid_discretization(grid), order(=1), n[, csys])
  pcms.LagrangeFunctionSpace.Backend.X        -> pcms.LagrangeImplementation.X
  pcms.PolynomialReconstructionFunctionSpace.create(...)
                                              -> pcms.create_polynomial_reconstruction_space(...)
  pcms.PolynomialReconstructionFunctionSpace.from_mesh(mesh, ...)
    -> pcms.create_polynomial_reconstruction_space(pcms.backend.omega_h.create_discretization(mesh), ...)
  pcms.OmegaHConservativeProjection(...)      -> pcms.create_conservative_projection(...)
  pcms.OmegaHControlVariateProjection(...)    -> pcms.create_control_variate_projection(...)
  pcms.create_uniform_grid_binary_field(...)  -> pcms.backend.omega_h.create_uniform_grid_binary_field(...)
  pcms.create_uniform_grid_from_mesh(...)     -> pcms.backend.omega_h.create_uniform_grid_from_mesh(...)
  pcms.map_entity_field_to_vertices_X(...)    -> pcms.backend.omega_h.map_entity_field_to_vertices_X(...)
  pcms.compute_entity_centroids[_few](...)    -> pcms.backend.omega_h.compute_entity_centroids(...)
  <expr>.mesh()                               -> pcms.backend.omega_h.get_mesh(<expr>.get_discretization())

The keyword mesh_coordinate_system= becomes coordinate_system=. A module
imported as `import pcms as <alias>` is rewritten under its alias. Running the
script twice is a no-op.

Review the result: `<expr>.mesh()` is rewritten whatever <expr> is (the
script does not know types), and names imported with `from pcms import ...`
that moved are reported, not rewritten.
"""

import re
import sys

KW_RENAME = {"mesh_coordinate_system": "coordinate_system",
             "backend": "implementation"}


def split_args(text):
    """Split a call's argument text at top-level commas."""
    args, depth, cur, quote = [], 0, "", None
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def find_call(text, start):
    """Given the index of an opening '(', return the index of its ')'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced call")


def kw(arg):
    m = re.match(r"(\w+)\s*=(?!=)\s*(.*)", arg, re.S)
    return (m.group(1), m.group(2)) if m else (None, arg)


def rewrite_calls(text, prefix, fn):
    out, pos = [], 0
    while True:
        i = text.find(prefix, pos)
        if i < 0:
            out.append(text[pos:])
            return "".join(out)
        open_paren = i + len(prefix) - 1
        if i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_."):
            # part of a longer name (mypcms.X, other.pcms.X)
            out.append(text[pos:open_paren + 1])
            pos = open_paren + 1
            continue
        close = find_call(text, open_paren)
        args = split_args(text[open_paren + 1:close])
        out.append(text[pos:i])
        try:
            name, new_args = fn(args)
        except (IndexError, ValueError):
            # not a call this script understands (e.g. text in a docstring)
            out.append(text[i:close + 1])
            pos = close + 1
            continue
        if "\n" in text[i:close]:
            # keep a multi-line call multi-line: one argument per line,
            # indented 4 more than the line the call starts on
            line_start = text.rfind("\n", 0, i) + 1
            indent = re.match(r"[ \t]*", text[line_start:]).group(0) + "    "
            sep = ",\n" + indent
            out.append(f"{name}(\n{indent}{sep.join(new_args)})")
        else:
            out.append(f"{name}({', '.join(new_args)})")
        pos = close + 1


# the old LagrangeFunctionSpace.from_mesh parameters after (mesh, order,
# num_components), in order
FROM_MESH_TRAILING = ("mesh_coordinate_system", "global_id_name", "backend")


def lagrange_from_mesh(mod, args):
    pos = [a for a in args if kw(a)[0] is None]
    kws = [kw(a) for a in args if kw(a)[0] is not None]
    if len(pos) > 3 + len(FROM_MESH_TRAILING):
        raise ValueError("too many arguments")
    # a positional coordinate system stays positional; the old positional
    # global_id_name and backend are passed by keyword
    kws = list(zip(FROM_MESH_TRAILING[1:], pos[4:])) + kws
    pos = pos[:4]
    disc_args = [pos[0]] + [f"{k}={v}" for k, v in kws if k == "global_id_name"]
    rest = pos[1:] + [f"{KW_RENAME.get(k, k)}={v}" for k, v in kws
                      if k != "global_id_name"]
    disc = f"{mod}.backend.omega_h.create_discretization({', '.join(disc_args)})"
    return f"{mod}.create_lagrange_space", [disc] + rest


def lagrange_from_uniform_grid(mod, args):
    pos = [a for a in args if kw(a)[0] is None]
    kws = dict(kw(a) for a in args if kw(a)[0] is not None)
    grid, ncomp = pos[0], pos[1]
    csys = pos[2] if len(pos) > 2 else kws.pop("mesh_coordinate_system", None)
    order = pos[3] if len(pos) > 3 else kws.pop("order", "1")
    out = [f"{mod}.create_uniform_grid_discretization({grid})", order, ncomp]
    if csys is not None:
        out.append(csys)
    out += [f"{KW_RENAME.get(k, k)}={v}" for k, v in kws.items()]
    return f"{mod}.create_lagrange_space", out


def pr_from_mesh(mod, args):
    disc = f"{mod}.backend.omega_h.create_discretization({args[0]})"
    rest = [f"{KW_RENAME.get(kw(a)[0], kw(a)[0])}={kw(a)[1]}"
            if kw(a)[0] else a for a in args[1:]]
    return f"{mod}.create_polynomial_reconstruction_space", [disc] + rest


def simple(name):
    return lambda mod, args: (f"{mod}.{name}", args)


# top-level names that moved to pcms.backend.omega_h
MOVED_TO_OMEGA_H = ("create_uniform_grid_binary_field",
                    "create_uniform_grid_from_mesh",
                    "map_entity_field_to_vertices_average",
                    "map_entity_field_to_vertices_mls",
                    "compute_entity_centroids")
# removed aliases: (old, new)
ALIASES = (("compute_entity_centroids_few", "compute_entity_centroids"),)


def convert(text, mod="pcms"):
    """Rewrite the calls made through the module name `mod`."""
    def calls(name, fn):
        nonlocal text
        text = rewrite_calls(text, f"{mod}.{name}(", lambda a: fn(mod, a))

    m = re.escape(mod)
    calls("LagrangeFunctionSpace.from_mesh", lagrange_from_mesh)
    calls("LagrangeFunctionSpace.from_uniform_grid", lagrange_from_uniform_grid)
    text = re.sub(rf"(?<![\w.]){m}\.LagrangeFunctionSpace\.Backend\.",
                  f"{mod}.LagrangeImplementation.", text)
    calls("PolynomialReconstructionFunctionSpace.from_mesh", pr_from_mesh)
    calls("PolynomialReconstructionFunctionSpace.create",
          simple("create_polynomial_reconstruction_space"))
    calls("OmegaHConservativeProjection", simple("create_conservative_projection"))
    calls("OmegaHControlVariateProjection",
          simple("create_control_variate_projection"))
    for old, new in ALIASES:
        text = re.sub(rf"(?<![\w.]){m}\.{old}\b", f"{mod}.{new}", text)
    for name in MOVED_TO_OMEGA_H:
        text = re.sub(rf"(?<![\w.]){m}\.{name}\b",
                      f"{mod}.backend.omega_h.{name}", text)
    text = re.sub(r"(?<![\w.])((?:\w+\.)*\w+)\.mesh\(\)",
                  rf"{mod}.backend.omega_h.get_mesh(\1.get_discretization())",
                  text)
    return text


def main(files):
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        new = text
        for mod in ["pcms"] + re.findall(r"^\s*import\s+pcms\s+as\s+(\w+)",
                                         text, re.M):
            new = convert(new, mod)
        for n, line in enumerate(new.split("\n"), 1):
            imp = re.match(r"\s*from\s+pcms\s+import\s+(.*)", line)
            if imp:
                moved = [x for x in re.findall(r"\w+", imp.group(1))
                         if x in MOVED_TO_OMEGA_H
                         or x in dict(ALIASES)]
                if moved:
                    print(f"{f}:{n}: {', '.join(moved)} moved to "
                          "pcms.backend.omega_h; update this import by hand",
                          file=sys.stderr)
        if new != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
            print(f"updated {f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
