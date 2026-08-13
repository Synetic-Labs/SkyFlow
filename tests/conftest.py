"""
Shared fixtures for all suites: a small deterministic fleet on the nominal Crazyflie.

Tests are CPU-only with fixed keys (DESIGN.md §11). Nothing here changes global JAX
config — suites that need adapter-grade tolerances toggle x64 with their own
module-scoped fixture and restore it on teardown.
"""

import jax
import pytest

from skyflow.params import AIRFRAMES, sample_params


@pytest.fixture
def fleet_size() -> int:
    """Small fleet — big enough to expose batching mistakes, small enough to loop over."""
    return 5


@pytest.fixture
def crazyflie():
    """Nominal Crazyflie airframe (spec reference vehicle + its `limits` entry)."""
    return AIRFRAMES["crazyflie"]


@pytest.fixture
def key():
    """Deterministic base PRNG key; split it, never reuse it raw."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def nominal_params(crazyflie, fleet_size):
    """[F,P] float32 nominal parameter rows (scale=0 sampling — bit-exact nominal)."""
    return sample_params(jax.random.PRNGKey(1), crazyflie, fleet_size, 0.0)
