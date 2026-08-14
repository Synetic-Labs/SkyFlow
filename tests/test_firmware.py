"""
DESIGN.md §11, firmware suite — the control="sticks" seam against cudaflight.

The sensor helpers and protocol/signature conformance run everywhere. Everything that
touches the real CPU SITL sits behind `pytest.importorskip("cudaflight")` plus a further
skip when `load_cpu()` cannot produce a library (missing/blocked libcpuflight.so):
arm → spin-up → hover smoke at the 1 kHz tick, under jit, plus a masked-reset smoke.
Expected values come from the probed wheel v0.2.1 behavior: instances arm during create
and snapshot, so armed is truthy from the first tick and a masked reset replays the
fresh-snapshot trajectory deterministically.
"""

import importlib.util
import inspect
import math
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.firmware import CpuFirmwareFleet, GpuFirmwareFleet, baro_pa, flu_to_frd
from skyflow.types import FirmwareFleet


@runtime_checkable
class _RuntimeFirmwareFleet(FirmwareFleet, Protocol):
    """types.FirmwareFleet made runtime-checkable for structural isinstance checks."""


# -- sensor helpers: pure math, no cudaflight needed --------------------------------------


def test_flu_to_frd_flips_y_and_z():
    v = jnp.array([1.0, 2.0, 3.0], jnp.float32)
    np.testing.assert_array_equal(np.asarray(flu_to_frd(v)), [1.0, -2.0, -3.0])


def test_flu_to_frd_batched_involution_and_dtype():
    v = jax.random.normal(jax.random.PRNGKey(0), (5, 3), jnp.float32)
    out = flu_to_frd(v)
    assert out.shape == (5, 3) and out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(flu_to_frd(out)), np.asarray(v))  # own inverse
    # Level hover: generated-IMU specific force (0, 0, +g) in FLU becomes az = -g in FRD.
    frd = flu_to_frd(jnp.array([0.0, 0.0, 9.81], jnp.float32))
    np.testing.assert_array_equal(
        np.asarray(frd),
        np.array([0.0, -0.0, -9.81], np.float32),  # bit-exact f32 negation
    )


def test_baro_pa_isothermal_curve():
    assert float(baro_pa(0.0)) == pytest.approx(101325.0, rel=1e-6)  # sea level
    assert float(baro_pa(8434.0)) == pytest.approx(101325.0 / math.e, rel=1e-5)  # one scale height
    alt = jnp.linspace(0.0, 100.0, 33, dtype=jnp.float32)
    p = baro_pa(alt)
    assert p.dtype == jnp.float32
    assert np.all(np.diff(np.asarray(p)) < 0.0)  # strictly falls with altitude


# -- protocol / class shape: no cudaflight needed ------------------------------------------


def test_cpu_fleet_signatures_match_protocol():
    for name in ("fresh_firmware_state", "fw_step", "reset", "close"):
        proto = list(inspect.signature(getattr(FirmwareFleet, name)).parameters)
        impl = list(inspect.signature(getattr(CpuFirmwareFleet, name)).parameters)
        assert impl == proto, f"{name}: {impl} != {proto}"


def test_cpu_fleet_without_cudaflight_gives_install_guidance():
    if importlib.util.find_spec("cudaflight") is not None:
        pytest.skip("cudaflight installed; the guidance path cannot trigger")
    with pytest.raises(ImportError, match="cudaflight"):
        CpuFirmwareFleet(2)


def test_gpu_fleet_raises_pending_ffi_absorption():
    with pytest.raises(NotImplementedError, match="cudaflight"):
        GpuFirmwareFleet(8)


# -- real CPU SITL: importorskip + load_cpu gate --------------------------------------------

FLEET = 2


@pytest.fixture(scope="module")
def cpu_fleet():
    """A 2-instance CPU SITL fleet, or skip when the wheel/library is unavailable."""
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")
    fleet = CpuFirmwareFleet(FLEET)
    yield fleet
    fleet.close()


def _hover_inputs():
    """Level-hover sensor rows and mid-throttle AETR sticks for the whole fleet."""
    gyro_frd = flu_to_frd(jnp.zeros((FLEET, 3), jnp.float32))
    specforce_frd = flu_to_frd(jnp.tile(jnp.array([0.0, 0.0, 9.81], jnp.float32), (FLEET, 1)))
    baro = jnp.full((FLEET, 1), baro_pa(0.0), jnp.float32)
    sensors = jnp.concatenate([gyro_frd, specforce_frd, baro], axis=-1)
    sticks = jnp.zeros((FLEET, 4), jnp.float32)  # AETR: centered roll/pitch/yaw, mid throttle
    return sticks, sensors


def test_protocol_conformance(cpu_fleet):
    assert isinstance(cpu_fleet, _RuntimeFirmwareFleet)
    assert cpu_fleet.act_dim == 4


def test_hover_smoke_100_ticks_jitted(cpu_fleet):
    sticks, sensors = _hover_inputs()
    assert np.asarray(sensors[:, 5] == -9.81).all()  # level hover ⇒ az = -9.81 (FRD)

    step = jax.jit(cpu_fleet.fw_step)  # ordered io_callback must survive jit
    blob, fwstate = cpu_fleet.fresh_firmware_state()
    m = np.zeros((FLEET, 4), np.float32)
    for _ in range(100):
        blob, fwstate, motors, armed = step(blob, fwstate, sticks, sensors)
        m = np.asarray(motors)
        assert m.shape == (FLEET, 4) and m.dtype == np.float32
        assert np.all(np.isfinite(m)) and np.all(m >= 0.0) and np.all(m <= 1.0)
        assert armed.dtype == jnp.uint8 and np.all(np.asarray(armed) != 0)  # stays armed
    assert np.all(m > 0.0)  # armed at mid throttle: props actually spin

    assert blob.shape == (0,) and blob.dtype == jnp.uint8  # placeholder threading
    assert fwstate.shape == (0,) and fwstate.dtype == jnp.uint8


def test_masked_reset_smoke(cpu_fleet):
    sticks, sensors = _hover_inputs()
    blob, fwstate = cpu_fleet.fresh_firmware_state()
    blob, fwstate, m1, _ = cpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    blob, fwstate, m2, _ = cpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    m1, m2 = np.asarray(m1), np.asarray(m2)
    assert not np.array_equal(m2, m1)  # the firmware is actually evolving

    mask = jnp.array([1, 0], jnp.uint8)  # reset world 0 only
    blob, fwstate = cpu_fleet.reset(blob, fwstate, mask)
    _, _, m3, armed = cpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    m3 = np.asarray(m3)
    np.testing.assert_array_equal(m3[0], m1[0])  # world 0 replays the fresh first tick
    assert not np.array_equal(m3[1], m1[0])  # world 1 kept flying, untouched
    assert np.all(np.asarray(armed) != 0)
