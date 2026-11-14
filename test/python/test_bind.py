"""
Tests for binding sources to query points from Python.
"""

import numpy as np
import pytest
import pcms
import PyOmega_h as omega_h


def _linear(x, y):
    return 1.0 + 2.0 * x + 3.0 * y


@pytest.fixture
def mesh(world):
    return omega_h.build_box(
        world, omega_h.Family.SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0, False
    )


@pytest.fixture
def space(mesh):
    return pcms.LagrangeFunctionSpace.from_mesh(
        mesh, 1, 1, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH)


@pytest.fixture
def field(space):
    f = space.create_function()
    xy = f.get_layout().get_dof_holder_coordinates().get_values()
    f.set_dof_holder_data(_linear(xy[:, 0], xy[:, 1]))
    return f


def test_evaluators_share_a_bound_source(space, field):
    # the second point is outside the unit square
    bound = space.bind(
        pcms.CoordinateView([[0.25, 0.5], [2.0, 2.0], [0.75, 0.1]]),
        pcms.OutOfBoundsPolicy(pcms.OutOfBoundsMode.FILL, -1.0))
    assert bound.num_points() == 3
    np.testing.assert_array_equal(bound.found_points(), [0, 2])
    np.testing.assert_array_equal(bound.missing_points(), [1])
    assert not bound.found_points().flags.writeable

    first = space.create_point_evaluator(bound)
    second = space.create_point_evaluator(bound)
    for evaluator in (first, second):
        out = np.zeros(3)
        evaluator.evaluate(field, out)
        np.testing.assert_allclose(
            out, [_linear(0.25, 0.5), -1.0, _linear(0.75, 0.1)], atol=1e-12)


def test_points_outside_raise_in_error_mode(space):
    with pytest.raises(RuntimeError, match="outside the source's domain"):
        space.bind(pcms.CoordinateView([[2.0, 2.0]]))


def test_bound_source_belongs_to_its_source(mesh, space):
    other = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, 1, 1, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH)
    bound = space.bind(pcms.CoordinateView([[0.5, 0.5]]))
    with pytest.raises(RuntimeError, match="bound by another factory"):
        other.create_point_evaluator(bound)


def test_derivatives_need_a_source_that_has_them(space):
    bound = space.bind(pcms.CoordinateView([[0.5, 0.5]]))
    with pytest.raises(RuntimeError, match="derivative order 1"):
        space.create_point_evaluator(bound, derivative_order=1)
    with pytest.raises(RuntimeError, match="derivative order 1"):
        pcms.gradient(space).create_point_evaluator(bound)


def test_interpolator_reuses_a_bound_source(mesh, space, field):
    target = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, 1, 1, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH)
    bound = space.bind(target)
    out = target.create_function()
    pcms.Interpolator(space, target, bound).apply(field, out)
    np.testing.assert_allclose(out.get_dof_holder_data().get_values(),
                               field.get_dof_holder_data().get_values(),
                               atol=1e-12)
