"""Uncertainty utilities, independent of network/model code."""
import numpy as np


def crossed_bootstrap(values, seed=20261026, repetitions=2000):
    """Resample independent training runs and common scenario columns separately.

    values shape [training_seed, paired_scenario]. This retains pairing across
    policies when called on differences. Three seeds limit interval reliability.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("finite nonempty 2D [training_seed, scenario] array required")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    rows, cols = values.shape
    for i in range(repetitions):
        r = rng.integers(rows, size=rows)
        c = rng.integers(cols, size=cols)
        samples[i] = values[np.ix_(r, c)].mean()
    low, high = np.quantile(samples, [.025, .975])
    return float(values.mean()), float(low), float(high)
