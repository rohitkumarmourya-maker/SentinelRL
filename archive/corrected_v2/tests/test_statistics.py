"""Mathematical utility checks with synthetic test values, not experiment results."""
import numpy as np
import pytest
from src.statistics import crossed_bootstrap


def test_constant_array_interval_is_exact():
    assert crossed_bootstrap(np.full((3, 4), 2.)) == (2., 2., 2.)


def test_paired_identical_policies_have_zero_difference():
    values = np.arange(12).reshape(3, 4)
    assert crossed_bootstrap(values - values) == (0., 0., 0.)


def test_bootstrap_reproducibility():
    values = np.arange(12).reshape(3, 4)
    assert crossed_bootstrap(values, repetitions=100) == crossed_bootstrap(values, repetitions=100)


def test_bootstrap_rejects_invalid_sampling_layout():
    with pytest.raises(ValueError):
        crossed_bootstrap([1, 2])
