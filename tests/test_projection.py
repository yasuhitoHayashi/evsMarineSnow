import numpy as np
import pytest

from saveParticleSize import project_parallel_to_view_vector


def test_project_parallel_zero_vector():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    view = np.array([0.0, 0.0, 0.0])
    with pytest.warns(UserWarning):
        proj = project_parallel_to_view_vector(points, view)
    expected = points.copy()
    expected[:, 2] = 0
    np.testing.assert_allclose(proj, expected)


def test_project_parallel_zero_z_component():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    view = np.array([1.0, 0.0, 0.0])
    with pytest.warns(UserWarning):
        proj = project_parallel_to_view_vector(points, view)
    expected = points.copy()
    expected[:, 2] = 0
    np.testing.assert_allclose(proj, expected)
