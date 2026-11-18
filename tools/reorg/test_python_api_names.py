"""Tests for python_api_names.py (run with pytest from tools/reorg)."""

import os
import subprocess
import sys

import python_api_names as t

TOOL = os.path.join(os.path.dirname(__file__), "python_api_names.py")


def run(tmp_path, source):
    f = tmp_path / "x.py"
    f.write_text(source)
    r = subprocess.run([sys.executable, TOOL, str(f)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    return f.read_text(), r.stderr


def test_quoted_parenthesis_in_arguments():
    src = 'g = pcms.create_point_cloud_geometry(c, 3, n=lookup("("))\n'
    assert t.convert(src) == \
        'g = pcms.backend.omega_h.create_point_cloud_geometry(c, 3, n=lookup("("))\n'


def test_unbalanced_text_is_skipped(tmp_path):
    out, _ = run(tmp_path, '"""pcms.OmegaHConservativeProjection( ("""\n'
                 'x = pcms.OmegaHConservativeProjection(a)\n')
    assert "x = pcms.create_conservative_projection(a)" in out


def test_mesh_receivers():
    src = "space = pcms.create_lagrange_space(g, 1, 1)\n" \
          "a = space.mesh()\nb = other.mesh()\n"
    out = t.convert(src)
    assert "a = pcms.backend.omega_h.get_mesh(space.get_geometry())" in out
    assert "b = other.mesh()" in out


def test_unknown_mesh_reported_and_alias_only_file(tmp_path):
    out, err = run(tmp_path, "import pcms as p\nb = other.mesh()\n"
                   "s = p.create_lagrange_space(g, 1, 1)\nm = s.mesh()\n")
    assert "b = other.mesh()" in out and ".mesh() call not rewritten" in err
    assert "m = p.backend.omega_h.get_mesh(s.get_geometry())" in out
    assert "pcms." not in out


def test_renamed_direct_import_reported(tmp_path):
    out, err = run(tmp_path, "from pcms import create_omega_h_geometry\n")
    assert "from pcms.backend.omega_h import create_geometry" in err
