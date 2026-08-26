"""
DESIGN.md §11, firmware suite — the control="sticks" seam against cudaflight.

The sensor helpers and protocol/signature conformance run everywhere. Everything that
touches the real CPU SITL sits behind `pytest.importorskip("cudaflight")` plus a further
skip when `load_cpu()` cannot produce a library (missing/blocked libcpuflight.so):
arm → spin-up → hover smoke at the 1 kHz tick, under jit, plus a masked-reset smoke.
Expected values follow the cudaflight CPU SITL behavior: instances arm during create
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


def test_gpu_fleet_signatures_match_protocol():
    for name in ("fresh_firmware_state", "fw_step", "reset", "close"):
        proto = list(inspect.signature(getattr(FirmwareFleet, name)).parameters)
        impl = list(inspect.signature(getattr(GpuFirmwareFleet, name)).parameters)
        assert impl == proto, f"{name}: {impl} != {proto}"


def test_gpu_fleet_rejects_small_fleet():
    # the fleet-size gate fires before any cudaflight import, so it runs everywhere
    with pytest.raises(ValueError, match="fleet >= 3"):
        GpuFirmwareFleet(2)


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


# -- real GPU fleet (DESIGN.md §10): cudaflight.xla in-jit custom calls ---------------------

GPU_FLEET = 4


@pytest.fixture(scope="module")
def gpu_fleet():
    """A 4-instance GPU fleet, or skip without CUDA / cudaflight >= 0.3.3."""
    pytest.importorskip("cudaflight")
    pytest.importorskip("cudaflight.xla", reason="cudaflight < 0.3.3: no xla module")
    try:
        jax.devices("gpu")
    except RuntimeError:
        pytest.skip("no CUDA device visible to jax")
    try:
        fleet = GpuFirmwareFleet(GPU_FLEET)
    except Exception as e:  # driver/VRAM/create failures are environment, not code
        pytest.skip(f"GPU fleet construction failed: {e}")
    yield fleet
    fleet.close()


def _hover_inputs_n(n: int):
    gyro_frd = flu_to_frd(jnp.zeros((n, 3), jnp.float32))
    specforce_frd = flu_to_frd(jnp.tile(jnp.array([0.0, 0.0, 9.81], jnp.float32), (n, 1)))
    baro = jnp.full((n, 1), baro_pa(0.0), jnp.float32)
    sensors = jnp.concatenate([gyro_frd, specforce_frd, baro], axis=-1)
    sticks = jnp.zeros((n, 4), jnp.float32)
    return sticks, sensors


@pytest.mark.gpu
def test_gpu_protocol_conformance(gpu_fleet):
    assert isinstance(gpu_fleet, _RuntimeFirmwareFleet)
    assert gpu_fleet.act_dim == 4


@pytest.mark.gpu
def test_gpu_hover_smoke_100_ticks_jitted(gpu_fleet):
    sticks, sensors = _hover_inputs_n(GPU_FLEET)
    step = jax.jit(gpu_fleet.fw_step)
    blob, fwstate = gpu_fleet.fresh_firmware_state()
    assert blob.size > 0 and fwstate.size > 0  # genuinely value-threaded, not placeholders
    m = np.zeros((GPU_FLEET, 4), np.float32)
    for _ in range(100):
        blob, fwstate, motors, armed = step(blob, fwstate, sticks, sensors)
        m = np.asarray(motors)
        assert m.shape == (GPU_FLEET, 4) and m.dtype == np.float32
        assert np.all(np.isfinite(m)) and np.all(m >= 0.0) and np.all(m <= 1.0)
        assert armed.dtype == jnp.uint8 and np.all(np.asarray(armed) != 0)
    assert np.all(m > 0.0)  # armed at mid throttle: props actually spin


@pytest.mark.gpu
def test_gpu_snapshot_restore_is_deterministic(gpu_fleet):
    """The value-threaded pair makes the GPU fleet replayable: a fresh snapshot copy
    replays the identical first tick, and a masked reset restores only flagged worlds."""
    sticks, sensors = _hover_inputs_n(GPU_FLEET)
    blob, fwstate = gpu_fleet.fresh_firmware_state()
    blob, fwstate, m1, _ = gpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    blob, fwstate, m2, _ = gpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    m1, m2 = np.asarray(m1), np.asarray(m2)
    assert not np.array_equal(m2, m1)  # the firmware is actually evolving

    # replay from a fresh snapshot copy: bit-identical first tick
    blob_b, fwstate_b = gpu_fleet.fresh_firmware_state()
    _, _, m1_replay, _ = gpu_fleet.fw_step(blob_b, fwstate_b, sticks, sensors)
    np.testing.assert_array_equal(np.asarray(m1_replay), m1)

    # masked reset: world 0 replays the fresh first tick, world 1 keeps flying
    mask = jnp.array([1, 0, 0, 0], jnp.uint8)
    blob, fwstate = gpu_fleet.reset(blob, fwstate, mask)
    _, _, m3, armed = gpu_fleet.fw_step(blob, fwstate, sticks, sensors)
    m3 = np.asarray(m3)
    np.testing.assert_array_equal(m3[0], m1[0])
    assert not np.array_equal(m3[1], m1[1])
    assert np.all(np.asarray(armed) != 0)


@pytest.mark.gpu
def test_gpu_snapshot_rides_as_jax_buffers(gpu_fleet):
    """cudaflight >= 0.3.4: the reset call receives the snapshot as JAX buffer
    arguments and the library-side snapshot copies are freed at construction."""
    from cudaflight import xla as cfx

    if not getattr(cfx, "SUPPORTS_SNAPSHOT_ARGS", False):
        pytest.skip("cudaflight < 0.3.4: snapshot rides as baked pointers")
    assert gpu_fleet._snapshot_args
    # the library copies are gone; the JAX-side snapshot is the only one left
    assert int(gpu_fleet._lib.cudaflight_snap_ptr(gpu_fleet._h)) == 0
    assert int(gpu_fleet._lib.cudaflight_snap_state_ptr(gpu_fleet._h)) == 0
    assert gpu_fleet._snap_blob.size > 0 and gpu_fleet._snap_state.size > 0


@pytest.mark.gpu
def test_env_firmware_auto_picks_gpu_fleet(gpu_fleet):
    """firmware="auto" on a CUDA box with fleet >= 3 builds the GPU backend, and the
    env steps through it (one jitted control step, finite outputs)."""
    from skyflow import SimConfig, SkyFlowEnv

    env = SkyFlowEnv(
        SimConfig(num_envs=GPU_FLEET, task="hover", control="sticks", firmware="auto")
    )
    try:
        assert isinstance(env._fw, GpuFirmwareFleet)
        obs, state = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        aetr = jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0], jnp.float32), (GPU_FLEET, 1))
        obs, state, reward, _done, _info = step(state, aetr)
        assert bool(jnp.isfinite(obs).all()) and reward.shape == (GPU_FLEET,)
    finally:
        env.close()
