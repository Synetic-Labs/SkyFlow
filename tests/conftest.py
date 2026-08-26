"""
Shared fixtures for all suites: a small deterministic fleet on the nominal Crazyflie.

Tests are CPU-only with fixed keys (DESIGN.md §11) — enforced below by pinning jax's
default device to the CPU for the whole session, so a CUDA box validates the same
numbers CI does instead of silently validating GPU fusion. @pytest.mark.gpu tests
place their arrays on the GPU explicitly and are unaffected. Suites that need
adapter-grade tolerances toggle x64 with their own module-scoped fixture and restore
it on teardown.
"""

import jax
import pytest

from skyflow.params import AIRFRAMES, sample_params


@pytest.fixture(scope="session", autouse=True)
def _cpu_default_device():
    """Pin unannotated computation to the CPU backend (DESIGN.md §11)."""
    prev = jax.config.jax_default_device
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    yield
    jax.config.update("jax_default_device", prev)


def _cpu_sitl_reason() -> str | None:
    """None when the CPU SITL can boot here, else the skip reason."""
    import importlib.util

    if importlib.util.find_spec("cudaflight") is None:
        return "cudaflight not installed"
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        return f"libcpuflight unavailable: {e}"
    return None


@pytest.fixture(params=["motors", "sticks"])
def control_mode(request) -> str:
    """Both control modes; sticks skips where the CPU SITL cannot boot. Production
    runs sticks — suites that step a real env should run through this fixture so
    the sticks axis is never untested again (TECH_DEBT R4)."""
    if request.param == "sticks":
        reason = _cpu_sitl_reason()
        if reason is not None:
            pytest.skip(reason)
    return request.param


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
