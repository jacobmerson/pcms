#!/usr/bin/env python3
"""Rewrite Python code to the final PCMS Python API (0.4).

Usage (from the repository root):
    tools/reorg/python_api_names.py <file> [<file> ...]

Rewrites, preserving the arguments:

  pcms.LagrangeFunctionSpace.from_mesh(mesh, order, n[, csys][, global_id_name=g][, backend=B])
    -> pcms.create_lagrange_space(pcms.omega_h.create_geometry(mesh[, csys][, global_id_name=g]),
                                  order, n[, backend=B])
  pcms.LagrangeFunctionSpace.from_uniform_grid(grid, n[, csys][, order])
    -> pcms.create_lagrange_space(pcms.uniform_grid.create_geometry(grid[, csys]), order(=1), n)
  pcms.LagrangeFunctionSpace.Backend.X        -> pcms.LagrangeBackend.X
  pcms.PolynomialReconstructionFunctionSpace.create(coords[, options])
    -> pcms.create_polynomial_reconstruction_space(pcms.point_cloud.create_geometry(coords)[, options])
  pcms.PolynomialReconstructionFunctionSpace.from_mesh(mesh, entity_dim[, csys][, options])
    -> pcms.create_polynomial_reconstruction_space(
         pcms.omega_h.create_point_cloud_geometry(
           pcms.omega_h.create_geometry(mesh[, system=csys]), entity_dim)[, options])
  pcms.OmegaHConservativeProjection(...)      -> pcms.create_conservative_projection(...)
  pcms.OmegaHControlVariateProjection(...)    -> pcms.create_control_variate_projection(...)
  pcms.create_omega_h_geometry(...)           -> pcms.omega_h.create_geometry(...)
  pcms.create_point_cloud_geometry(geometry, entity_dim[, ...])
                                              -> pcms.omega_h.create_point_cloud_geometry(...)
  pcms.create_point_cloud_geometry(coords_or_layout[, ...])
                                              -> pcms.point_cloud.create_geometry(...)
  pcms.create_uniform_grid_geometry(...)      -> pcms.uniform_grid.create_geometry(...)
  pcms.create_uniform_grid_binary_field(...)  -> pcms.omega_h.create_uniform_grid_binary_field(...)
  pcms.create_uniform_grid_from_mesh(...)     -> pcms.omega_h.create_uniform_grid_from_mesh(...)
  pcms.Interpolator(...)                      -> pcms.create_interpolator(...)
  pcms.Copy(...)                              -> pcms.create_copy(...)
  pcms.csys.Cartesian.create(dim)             -> pcms.csys.create_cartesian(dim)
  pcms.csys.CylindricalRThetaZ.create()       -> pcms.csys.create_cylindrical_r_theta_z()
  pcms.csys.CylindricalRZ.create()            -> pcms.csys.create_cylindrical_rz()
  pcms.CartesianToCylindrical()               -> pcms.create_cartesian_to_cylindrical()
  pcms.CylindricalToCartesian()               -> pcms.create_cylindrical_to_cartesian()

The keyword mesh_coordinate_system= becomes system= (the coordinate system is
the geometry's). A module imported as `import pcms as <alias>` is rewritten
under its alias. Running the script twice is a no-op.

Reported, not rewritten (they have no direct replacement): `<expr>.mesh()`
(PCMS does not hand a mesh back: keep the Omega_h mesh the geometry was
created from), the removed helpers map_entity_field_to_vertices_average/_mls,
compute_entity_centroids[_few], patch_size_for_degree, same_coordinate_system,
same_value_basis and has_identity_metric (see REMOVED below for each
replacement). Review the result: `pcms.create_point_cloud_geometry(a, b)` is
treated as the Omega_h-geometry form when `b` is an integer literal, a name ending in
`dim`, or entity_dim= is given; the MLSOptions fields radius,
min_req_supports and adapt_radius are now NeighbourhoodOptions fields
(radius, min_supports, adapt_radius) of the point-cloud geometry, and are
reported, not rewritten; names imported with `from pcms import ...` that moved
are reported, not rewritten (so is `from pcms import create_omega_h_geometry`).
"""

import re
import sys

KW_RENAME = {"mesh_coordinate_system": "system"}


def split_args(text):
    """Split a call's argument text at top-level commas."""
    args, depth, cur, quote, skip = [], 0, "", None, False
    for ch in text:
        if skip:
            cur += ch
            skip = False
            continue
        if quote:
            cur += ch
            if ch == "\\":
                skip = True
            elif ch == quote:
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
    """Given the index of an opening '(', return the index of its ')'.

    Parentheses inside string literals and comments are not counted."""
    depth, i = 0, start
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            quote = ch * 3 if text.startswith(ch * 3, i) else ch
            i += len(quote)
            while i < len(text) and not text.startswith(quote, i):
                i += 2 if text[i] == "\\" else 1
            i += len(quote)
            continue
        if ch == "#":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
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
        try:
            close = find_call(text, open_paren)
        except ValueError:
            # not a complete call (e.g. text in a docstring): skip it
            out.append(text[pos:open_paren + 1])
            pos = open_paren + 1
            continue
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
    # the coordinate system moves to the geometry (positional stays
    # positional); the old positional global_id_name and backend are passed by
    # keyword
    csys = pos[3:4]
    kws = list(zip(FROM_MESH_TRAILING[1:], pos[4:])) + kws
    pos = pos[:3]
    geometry_args = [pos[0]] + csys
    geometry_args += [f"{KW_RENAME.get(k, k)}={v}" for k, v in kws
                      if k in ("mesh_coordinate_system", "global_id_name")]
    rest = [pos[1], pos[2]] + [f"{k}={v}" for k, v in kws if k == "backend"]
    geometry = f"{mod}.omega_h.create_geometry({', '.join(geometry_args)})"
    return f"{mod}.create_lagrange_space", [geometry] + rest


def lagrange_from_uniform_grid(mod, args):
    pos = [a for a in args if kw(a)[0] is None]
    kws = dict(kw(a) for a in args if kw(a)[0] is not None)
    grid, ncomp = pos[0], pos[1]
    csys = pos[2] if len(pos) > 2 else kws.pop("mesh_coordinate_system", None)
    order = pos[3] if len(pos) > 3 else kws.pop("order", "1")
    geometry_args = [grid] + ([csys] if csys is not None else [])
    geometry = f"{mod}.uniform_grid.create_geometry({', '.join(geometry_args)})"
    out = [geometry, order, ncomp]
    out += [f"{KW_RENAME.get(k, k)}={v}" for k, v in kws.items()]
    return f"{mod}.create_lagrange_space", out


def pr_create(mod, args):
    pos = [a for a in args if kw(a)[0] is None]
    kws = [a for a in args if kw(a)[0] is not None]
    geometry = f"{mod}.point_cloud.create_geometry({pos[0]})"
    return f"{mod}.create_polynomial_reconstruction_space", [geometry] + pos[1:] + kws


def pr_from_mesh(mod, args):
    pos = [a for a in args if kw(a)[0] is None]
    kws = dict(kw(a) for a in args if kw(a)[0] is not None)
    # (mesh, source_entity_dim, coordinate_system, options)
    names = ("mesh", "source_entity_dim", "coordinate_system", "options")
    given = dict(zip(names, pos))
    given.update({"coordinate_system" if k == "mesh_coordinate_system" else k: v
                  for k, v in kws.items()})
    mesh_args = [given["mesh"]]
    if "coordinate_system" in given:
        mesh_args.append(f"system={given['coordinate_system']}")
    mesh_geometry = f"{mod}.omega_h.create_geometry({', '.join(mesh_args)})"
    geometry = f"{mod}.omega_h.create_point_cloud_geometry({mesh_geometry}, {given['source_entity_dim']})"
    out = [geometry]
    if "options" in given:
        out.append(given["options"])
    return f"{mod}.create_polynomial_reconstruction_space", out


def point_cloud_geometry(mod, args):
    """create_point_cloud_geometry(geometry, entity_dim[, ...]): the form on an Omega_h geometry."""
    pos = [a for a in args if kw(a)[0] is None]
    kws = dict(kw(a) for a in args if kw(a)[0] is not None)
    second = pos[1] if len(pos) > 1 else None
    is_mesh_form = ("entity_dim" in kws or
                    (second is not None and
                     re.fullmatch(r"\d+|\w*dim", second) is not None))
    if not is_mesh_form:
        return f"{mod}.point_cloud.create_geometry", args
    return f"{mod}.omega_h.create_point_cloud_geometry", args


def simple(name):
    return lambda mod, args: (f"{mod}.{name}", args)


# top-level names that moved to pcms.omega_h
MOVED_TO_OMEGA_H = ("create_uniform_grid_binary_field",
                    "create_uniform_grid_from_mesh")
# removed aliases: (old, new)
ALIASES = ()
# renamed (and moved) names: (old, new)
RENAMED = (("create_omega_h_geometry", "omega_h.create_geometry"),
           ("create_uniform_grid_geometry", "uniform_grid.create_geometry"),
           ("csys.Cartesian.create", "csys.create_cartesian"),
           ("csys.CylindricalRThetaZ.create", "csys.create_cylindrical_r_theta_z"),
           ("csys.CylindricalRZ.create", "csys.create_cylindrical_rz"),
           ("CartesianToCylindrical", "create_cartesian_to_cylindrical"),
           ("CylindricalToCartesian", "create_cylindrical_to_cartesian"))
# removed names and their replacement, reported
REMOVED = {
    "map_entity_field_to_vertices_average":
        "read the entity values as a P0 field (create_lagrange_space(geometry, "
        "0, n)) and interpolate it with create_interpolator",
    "map_entity_field_to_vertices_mls":
        "create_polynomial_reconstruction_space(omega_h.create_point_cloud_"
        "geometry(geometry, dim), options) and create_interpolator",
    "compute_entity_centroids":
        "the DOF-holder coordinates of a P0 space or of "
        "omega_h.create_point_cloud_geometry(geometry, dim)",
    "compute_entity_centroids_few":
        "the DOF-holder coordinates of a P0 space or of "
        "omega_h.create_point_cloud_geometry(geometry, dim)",
    "patch_size_for_degree": "NeighbourhoodOptions.min_supports",
    "same_coordinate_system": "a == b",
    "same_value_basis": "a == b",
    "has_identity_metric":
        "s.has_orthogonal_basis() and s.has_unit_scale_factors()",
}


def convert(text, mod="pcms"):
    """Rewrite the calls made through the module name `mod`."""
    def calls(name, fn):
        nonlocal text
        text = rewrite_calls(text, f"{mod}.{name}(", lambda a: fn(mod, a))

    m = re.escape(mod)
    calls("LagrangeFunctionSpace.from_mesh", lagrange_from_mesh)
    calls("LagrangeFunctionSpace.from_uniform_grid", lagrange_from_uniform_grid)
    text = re.sub(rf"(?<![\w.]){m}\.LagrangeFunctionSpace\.Backend\.",
                  f"{mod}.LagrangeBackend.", text)
    calls("PolynomialReconstructionFunctionSpace.from_mesh", pr_from_mesh)
    calls("PolynomialReconstructionFunctionSpace.create", pr_create)
    calls("OmegaHConservativeProjection", simple("create_conservative_projection"))
    calls("OmegaHControlVariateProjection",
          simple("create_control_variate_projection"))
    calls("create_point_cloud_geometry", point_cloud_geometry)
    calls("Interpolator", simple("create_interpolator"))
    calls("Copy", simple("create_copy"))
    for old, new in ALIASES + RENAMED:
        text = re.sub(rf"(?<![\w.]){m}\.{old}\b", f"{mod}.{new}", text)
    for name in MOVED_TO_OMEGA_H:
        text = re.sub(rf"(?<![\w.]){m}\.{name}\b",
                      f"{mod}.omega_h.{name}", text)
    return text


def main(files):
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        new = text
        aliases = re.findall(r"^\s*import\s+pcms\s+as\s+(\w+)", text, re.M)
        mods = list(aliases)
        # `pcms` itself, unless the file imports the module only under an alias
        if not aliases or re.search(r"^\s*import\s+pcms(?!\s+as\b)", text, re.M):
            mods.insert(0, "pcms")
        for mod in mods:
            new = convert(new, mod)
        for n, line in enumerate(new.split("\n"), 1):
            if re.search(r"\bmin_req_supports\b", line) or \
                    re.search(r"\b\w*(opts|options)\w*\.(radius|adapt_radius)\b", line) or \
                    re.search(r"MLSOptions\(.*\b(radius|adapt_radius)\b", line):
                print(f"{f}:{n}: radius, min_req_supports and adapt_radius are "
                      "NeighbourhoodOptions fields (radius, min_supports, "
                      "adapt_radius) of the point-cloud geometry; update by "
                      "hand", file=sys.stderr)
            if re.search(r"(?<![\w.])(?:\w+\.)*\w+\.mesh\(\)", line):
                print(f"{f}:{n}: .mesh() call not rewritten: if the receiver "
                      "is a PCMS space, use the Omega_h mesh its geometry was "
                      "created from (PCMS does not hand a mesh back)",
                      file=sys.stderr)
            for name, advice in REMOVED.items():
                if re.search(rf"\b(?:\w+\.)*{name}\b", line):
                    print(f"{f}:{n}: {name} was removed; use {advice}",
                          file=sys.stderr)
            imp = re.match(r"\s*from\s+pcms\s+import\s+(.*)", line)
            if imp:
                moved = [x for x in re.findall(r"\w+", imp.group(1))
                         if x in MOVED_TO_OMEGA_H
                         or x in dict(ALIASES)]
                for old, new_name in RENAMED:
                    if old in re.findall(r"\w+", imp.group(1)):
                        module, _, short = new_name.rpartition(".")
                        print(f"{f}:{n}: {old} is now pcms.{new_name}; use "
                              f"`from pcms.{module} import {short}` instead",
                              file=sys.stderr)
                if moved:
                    print(f"{f}:{n}: {', '.join(moved)} moved to "
                          "pcms.omega_h; update this import by hand",
                          file=sys.stderr)
        if new != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
            print(f"updated {f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
