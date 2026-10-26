"""Tests for the elastic split that starts the feasibility restoration."""

import numpy as np
import pytest

from amigo.algorithm.globalization.feasibility_restoration import elastic_split

RESIDUALS = np.array([-1e3, -2.0, -1e-6, 0.0, 3e-7, 0.5, 1e4])


@pytest.mark.parametrize("mu, rho", [(1e-1, 1e3), (1e-6, 1e3), (1.0, 10.0)])
def test_split_is_feasible_and_optimal(mu, rho):
    p, n = elastic_split(RESIDUALS, mu, rho)
    assert np.all(p > 0.0) and np.all(n > 0.0)
    assert np.allclose(p - n, RESIDUALS, rtol=1e-12, atol=1e-12 * rho)
    # Stationarity of rho (p + n) - mu log p - mu log n along p - n = r
    assert np.allclose(mu / p + mu / n, 2.0 * rho, rtol=1e-10)


def test_split_tends_to_the_l1_violation():
    r = np.array([-2.0, -0.3, 0.7, 5.0])
    p, n = elastic_split(r, 1e-10, 1e3)
    assert np.allclose(p + n, np.abs(r), rtol=1e-6)
