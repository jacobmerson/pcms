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
        'g = pcms.omega_h.create_point_cloud_geometry(c, 3, n=lookup("("))\n'


def test_unbalanced_text_is_skipped(tmp_path):
    out, _ = run(tmp_path, '"""pcms.OmegaHConservativeProjection( ("""\n'
                 'x = pcms.OmegaHConservativeProjection(a)\n')
    assert "x = pcms.create_conservative_projection(a)" in out


def test_mesh_calls_and_removed_helpers_reported(tmp_path):
    out, err = run(tmp_path, "s = pcms.create_lagrange_space(g, 1, 1)\n"
                   "m = s.mesh()\n"
                   "c = pcms.compute_entity_centroids(mesh, 2)\n")
    assert "m = s.mesh()" in out and ".mesh() call not rewritten" in err
    assert "compute_entity_centroids was removed" in err


def test_transfers_and_coordinate_systems_are_factories():
    src = "op = pcms.Interpolator(a, b)\nc = pcms.Copy(a, b)\n" \
          "s = pcms.csys.Cartesian.create(3)\n" \
          "m = pcms.CartesianToCylindrical()\n"
    out = t.convert(src)
    assert "op = pcms.create_interpolator(a, b)" in out
    assert "c = pcms.create_copy(a, b)" in out
    assert "s = pcms.csys.create_cartesian(3)" in out
    assert "m = pcms.create_cartesian_to_cylindrical()" in out
    assert t.convert(out) == out


def test_renamed_direct_import_reported(tmp_path):
    out, err = run(tmp_path, "from pcms import create_omega_h_geometry\n")
    assert "from pcms.omega_h import create_geometry" in err


def test_uniform_grid_and_point_cloud_geometry_moved():
    src = "a = pcms.create_uniform_grid_geometry(grid)\n" \
          "b = pcms.create_point_cloud_geometry(coords)\n"
    out = t.convert(src)
    assert "a = pcms.uniform_grid.create_geometry(grid)" in out
    assert "b = pcms.point_cloud.create_geometry(coords)" in out
    assert t.convert(out) == out
