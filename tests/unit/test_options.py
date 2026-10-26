"""Tests for the legacy option aliases and the fraction to the boundary rule."""

import pytest

from amigo.algorithm.barrier_strategy import fraction_to_boundary
from amigo.algorithm.default_options import get_default_options


def test_tau_min_maps_to_fraction_to_boundary():
    options = get_default_options({"tau_min": 0.995})
    assert options["fraction_to_boundary"] == 0.995
    assert "tau_min" not in options


def test_explicit_fraction_to_boundary_wins_over_alias():
    options = get_default_options({"fraction_to_boundary": 0.98, "tau_min": 0.995})
    assert options["fraction_to_boundary"] == 0.98


@pytest.mark.parametrize(
    "mu, expected", [(1.0, 0.995), (0.1, 0.995), (1e-3, 0.999), (0.0, 1.0)]
)
def test_adaptive_tau_uses_the_option_as_floor(mu, expected):
    options = get_default_options({"fraction_to_boundary": 0.995})
    assert fraction_to_boundary(options, mu) == pytest.approx(expected)


def test_fixed_tau_ignores_mu():
    options = get_default_options({"fraction_to_boundary": 0.95, "adaptive_tau": False})
    assert fraction_to_boundary(options, 1e-6) == 0.95


@pytest.mark.parametrize(
    "name, value",
    [
        ("adaptive_mu_globalization", "kkt_error"),
        ("mu_init_strategy", "fix"),
        ("quality_function_centrality", "logarithmic"),
        ("barrier_strategy", "quality-function"),
    ],
)
def test_misspelled_choice_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        get_default_options({name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("convergence_tolerance", "1e-8"),
        ("max_iterations", True),
        ("second_order_correction", "yes"),
    ],
)
def test_wrong_type_is_rejected(name, value):
    with pytest.raises(TypeError, match=name):
        get_default_options({name: value})


def test_nan_tolerance_is_rejected():
    with pytest.raises(ValueError, match="convergence_tolerance"):
        get_default_options({"convergence_tolerance": float("nan")})


def test_integer_flags_and_numbers_are_accepted():
    options = get_default_options(
        {"second_order_correction": 0, "max_iterations": 50.0, "convergence_tolerance": 1}
    )
    assert options["second_order_correction"] == 0
    assert options["max_iterations"] == 50.0


def test_legacy_option_warns_that_it_has_no_effect():
    with pytest.warns(UserWarning, match="bound_push"):
        get_default_options({"bound_push": 1e-3})
