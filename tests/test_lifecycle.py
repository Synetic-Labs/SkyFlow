"""
Lifecycle seam (TECH_DEBT R6): SkyFlowEnv.close(), the one-CPU-fleet-per-process
guard, use-after-close raises, and eeprom boot-image reaping.

Everything that boots the real CPU SITL sits behind the same importorskip +
load_cpu() gate as test_firmware.py. This module holds NO module-scoped fleet:
every test closes what it opens, so no fleet survives into any other module and
the in-module eeprom render (which needs "no live fleet") stays legal anywhere
in the file.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from skyflow import DomainRand, SimConfig, SkyFlowEnv

_STOCK_DUMP = Path(__file__).parents[1] / "examples" / "configs" / "stock_dump.txt"


def _skip_without_cpu_sitl() -> None:
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")


# -- motors mode: close() needs no firmware and no wheel -----------------------------------


def test_motors_close_is_idempotent_and_guards_step_and_reset():
    env = SkyFlowEnv(SimConfig(num_envs=2, task="hover", dr=DomainRand().off()))
    _obs, state = env.reset(jax.random.PRNGKey(0))
    env.close()
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env.step(state, jnp.zeros((2, env.act_dim), jnp.float32))
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(jax.random.PRNGKey(1))


def test_context_manager_closes():
    with SkyFlowEnv(SimConfig(num_envs=2, task="hover", dr=DomainRand().off())) as env:
        _obs, state = env.reset(jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="closed"):
        env.step(state, jnp.zeros((2, env.act_dim), jnp.float32))


# -- real CPU SITL: importorskip + load_cpu gate --------------------------------------------


def test_cpu_fleet_one_per_process_frees_on_close():
    _skip_without_cpu_sitl()
    from skyflow.firmware import CpuFirmwareFleet

    first = CpuFirmwareFleet(2)
    try:
        with pytest.raises(RuntimeError, match="one global SITL fleet"):
            CpuFirmwareFleet(2)
    finally:
        first.close()
    second = CpuFirmwareFleet(2)  # constructs only because close() freed the slot
    second.close()


def test_cpu_fleet_use_after_close_raises_not_segfaults():
    _skip_without_cpu_sitl()
    from skyflow.firmware import CpuFirmwareFleet

    fleet = CpuFirmwareFleet(2)
    fleet.close()
    # the public entry points raise BEFORE any io_callback is emitted — a raise
    # inside an ordered callback would poison jax's token for the whole process
    with pytest.raises(RuntimeError, match="closed"):
        fleet.fresh_firmware_state()
    with pytest.raises(RuntimeError, match="closed"):
        fleet.fw_step(
            jnp.zeros((0,), jnp.uint8), jnp.zeros((0,), jnp.uint8),
            jnp.zeros((2, 4), jnp.float32), jnp.zeros((2, 7), jnp.float32),
        )


def test_sticks_env_close_frees_the_fleet_slot():
    _skip_without_cpu_sitl()
    cfg = SimConfig(
        num_envs=1, task="hover", control="sticks", firmware="cpu",
        dr=DomainRand().off(),
    )
    for _ in range(2):  # the second construction works only because close() ran
        env = SkyFlowEnv(cfg)
        _obs, state = env.reset(jax.random.PRNGKey(0))
        aetr = jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0], jnp.float32), (1, 1))
        env.step(state, aetr)
        env.close()


def test_injected_fleet_survives_env_close():
    _skip_without_cpu_sitl()
    from skyflow.firmware import CpuFirmwareFleet

    fleet = CpuFirmwareFleet(1)
    try:
        cfg = SimConfig(
            num_envs=1, task="hover", control="sticks", dr=DomainRand().off(),
        )
        env = SkyFlowEnv(cfg, firmware_fleet=fleet)
        env.close()
        # the injected fleet belongs to the caller: still open, still usable
        jax.block_until_ready(fleet.fresh_firmware_state())
    finally:
        fleet.close()


def test_env_close_reaps_eeprom_image():
    _skip_without_cpu_sitl()
    env = SkyFlowEnv(SimConfig(
        num_envs=1, task="hover", control="sticks", firmware="cpu",
        eeprom=str(_STOCK_DUMP), dr=DomainRand().off(),
    ))
    img = env.eeprom_image
    assert img is not None and Path(img).is_file()
    env.close()
    assert not Path(img).exists()
    assert env.eeprom_image == img  # the provenance path outlives the file


# -- real GPU fleet ------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_fleet_use_after_close_raises():
    pytest.importorskip("cudaflight")
    pytest.importorskip("cudaflight.xla", reason="cudaflight < 0.3.3: no xla module")
    from skyflow.firmware import GpuFirmwareFleet

    try:
        jax.devices("gpu")
    except RuntimeError:
        pytest.skip("no CUDA device visible to jax")
    try:
        fleet = GpuFirmwareFleet(4)
    except Exception as e:  # driver/VRAM/create failures are environment, not code
        pytest.skip(f"GPU fleet construction failed: {e}")
    fleet.close()
    with pytest.raises(RuntimeError, match="closed"):
        fleet.fresh_firmware_state()
