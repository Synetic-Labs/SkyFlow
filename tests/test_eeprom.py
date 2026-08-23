"""
SimConfig.eeprom — the drone-config seam: a Betaflight CLI `dump all` rendered into
the boot eeprom at env construction (DESIGN.md §10; examples/configs/README.md).

Validation tests run everywhere (they raise before any cudaflight import). The two
render tests sit behind `pytest.importorskip("cudaflight")` + a `load_cpu()` skip,
like the CPU SITL tests in test_firmware.py. They live in their OWN module because
the CPU library refuses a render while a fleet lives in the process ("render before
create or after destroy") — test_firmware.py holds a module-scoped fleet open across
its whole run, and module scoping guarantees no fleet survives into this module.
"""

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import DomainRand, SimConfig, SkyFlowEnv
from skyflow.types import FirmwareFleet

_STOCK_DUMP = Path(__file__).parents[1] / "examples" / "configs" / "stock_dump.txt"


def _skip_without_cpu_sitl() -> None:
    pytest.importorskip("cudaflight")
    from cudaflight.lib import load_cpu

    try:
        load_cpu()
    except Exception as e:  # missing or unloadable libcpuflight.so
        pytest.skip(f"libcpuflight unavailable: {e}")


# -- validation: raises before any cudaflight import, runs everywhere ---------------------


def test_overrides_without_dump_rejected():
    with pytest.raises(ValueError, match="eeprom_overrides"):
        SkyFlowEnv(SimConfig(num_envs=1, control="sticks",
                             eeprom_overrides="anything.txt"))


def test_eeprom_in_motors_mode_rejected():
    with pytest.raises(ValueError, match="sticks"):
        SkyFlowEnv(SimConfig(num_envs=1, control="motors", eeprom=str(_STOCK_DUMP)))


def test_eeprom_with_injected_fleet_rejected():
    cfg = SimConfig(num_envs=1, control="sticks", firmware="cpu", eeprom=str(_STOCK_DUMP))
    with pytest.raises(ValueError, match="exclusive"):
        SkyFlowEnv(cfg, firmware_fleet=cast(FirmwareFleet, object()))


def test_missing_dump_file_rejected():
    with pytest.raises(FileNotFoundError, match="no such CLI dump"):
        SkyFlowEnv(SimConfig(num_envs=1, control="sticks", firmware="cpu",
                             eeprom="/nonexistent/dump.txt"))


# -- render: one throwaway CPU boot per pass — needs the wheel + library ------------------


def test_stock_dump_renders_boots_and_climbs():
    """The full seam on the CPU backend: render the stock dump (version-gated
    round-trip), boot a 1-instance fleet from the image, and climb open loop."""
    _skip_without_cpu_sitl()

    env = SkyFlowEnv(SimConfig(
        num_envs=1, task="hover", task_kwargs={"goal_hold_s": 60.0},
        control="sticks", firmware="cpu", eeprom=str(_STOCK_DUMP),
        dr=DomainRand().off(),
    ))
    try:
        assert env.eeprom_image is not None
        assert Path(env.eeprom_image).stat().st_size > 0
        _obs, state = env.reset(jax.random.PRNGKey(0))
        z0 = float(np.asarray(state.plant[0, 2]))
        step = jax.jit(env.step)
        climb = jnp.array([[0.0, 0.0, 0.6, 0.0]], jnp.float32)  # AETR, above hover
        z_max = z0
        for _ in range(100):  # 1 s at 100 Hz
            _obs, state, _reward, done, _info = step(state, climb)
            z_max = max(z_max, float(np.asarray(state.plant[0, 2])))
            if bool(done.any()):
                break
        assert z_max > z0 + 0.05, f"never climbed: z0={z0:.3f}, z_max={z_max:.3f}"
    finally:
        assert env._fw is not None
        env._fw.close()


def test_version_gate_rejects_foreign_release(tmp_path):
    """A dump whose header names another firmware release fails at construction —
    never a silent factory-reset to stock defaults."""
    _skip_without_cpu_sitl()

    text = _STOCK_DUMP.read_text()
    assert "2026.6.1" in text  # the pinned wheel's release — see examples/configs
    stale = tmp_path / "stale_dump.txt"
    stale.write_text(text.replace("2026.6.1", "2026.6.0"))
    with pytest.raises(RuntimeError):
        SkyFlowEnv(SimConfig(num_envs=1, control="sticks", firmware="cpu",
                             eeprom=str(stale)))


# -- base auto-selection: the dump header picks the firmware binaries ----------------------

_HEADER_TMPL = ("# Betaflight / SITL_LOCKSTEP (SLCK) 2026.6.0-alpha May 15 2026 / "
                "06:14:55 ({rev}) MSP API: 1.48\n")


def _stage_bundle(root, rev):
    import hashlib
    import json

    d = root / rev
    d.mkdir(parents=True)
    sha = {}
    for name, payload in (("libcpuflight.so", b"lib"), ("fw.fatbin", b"fatbin")):
        (d / name).write_bytes(payload)
        sha[name] = hashlib.sha256(payload).hexdigest()
    (d / "manifest.json").write_text(json.dumps(
        {"tag": f"cudaflight-vX+bf.{rev}", "wheel_url": "staged", "sha256": sha}))


def test_base_resolution(tmp_path, monkeypatch):
    pytest.importorskip("cudaflight.bases")
    from skyflow.env import _installed_base_rev, _resolve_firmware_base

    monkeypatch.setenv("CUDAFLIGHT_BASE_CACHE", str(tmp_path / "cache"))

    # a norevision dump cannot select a base — the installed binaries serve
    norev = tmp_path / "norev.txt"
    norev.write_text(_HEADER_TMPL.format(rev="norevision"))
    assert _resolve_firmware_base(norev) is None

    # a dump naming the installed wheel's own base — no bundle lookup
    inst = _installed_base_rev()
    if inst is not None:
        own = tmp_path / "own.txt"
        own.write_text(_HEADER_TMPL.format(rev=inst + "f"))
        assert _resolve_firmware_base(own) is None

    # a foreign base with no cached bundle — fail loudly, naming the fetch command
    foreign = tmp_path / "foreign.txt"
    foreign.write_text(_HEADER_TMPL.format(rev="abcdef012"))
    with pytest.raises(FileNotFoundError, match=r"cudaflight\.bases abcdef012"):
        _resolve_firmware_base(foreign)

    # the same foreign base with a staged bundle — resolves to its pair
    _stage_bundle(tmp_path / "cache", "abcdef01")
    b = _resolve_firmware_base(foreign)
    assert b is not None and b.rev == "abcdef01"
    assert b.lib.name == "libcpuflight.so" and b.fatbin.name == "fw.fatbin"
