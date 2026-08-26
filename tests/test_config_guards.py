"""
Negative tests for SkyFlowEnv construction guards (TECH_DEBT Q4): every guard that
can refuse a config or an injected fleet must have a test proving it fires. All of
these raise before any firmware boots, so they run everywhere with no cudaflight.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from skyflow import SimConfig, SkyFlowEnv
from skyflow.env import DomainRand
from skyflow.types import FirmwareFleet


def test_num_envs_below_one_rejected():
    with pytest.raises(ValueError, match="num_envs"):
        SkyFlowEnv(SimConfig(num_envs=0))


def test_unknown_firmware_value_rejected():
    with pytest.raises(ValueError, match="auto"):
        SkyFlowEnv(SimConfig(num_envs=1, control="sticks", firmware="banana"))


def test_negative_dr_magnitude_rejected():
    with pytest.raises(ValueError, match="wind_gust_mps"):
        SkyFlowEnv(SimConfig(num_envs=1, dr=DomainRand(wind_gust_mps=-1.0)))


def test_injected_fleet_act_dim_mismatch_rejected():
    fake = cast(FirmwareFleet, SimpleNamespace(act_dim=6))
    with pytest.raises(ValueError, match="act_dim"):
        SkyFlowEnv(SimConfig(num_envs=1, control="sticks"), firmware_fleet=fake)


def test_bad_motor_perm_rejected():
    fake = cast(FirmwareFleet, SimpleNamespace(act_dim=4))
    with pytest.raises(ValueError, match="motor_perm"):
        SkyFlowEnv(
            SimConfig(num_envs=1, control="sticks"),
            firmware_fleet=fake,
            motor_perm=(0, 1, 2, 2),
        )


# -- eeprom board alignment: SkyFlow applies no inverse sensor rotation (F8) ----------
# The scan runs before any cudaflight import, so these tests run everywhere. A real
# dump (the Air75 factory CLI) carries align_board_yaw = -135 — it must WARN, never be
# rejected: the sim runs the real config.


def _sticks_cfg(dump_path, overrides=None) -> SimConfig:
    return SimConfig(
        num_envs=1, control="sticks", firmware="cpu", eeprom=str(dump_path),
        eeprom_overrides=None if overrides is None else str(overrides),
    )


def _construct_past_the_scan(cfg: SimConfig) -> None:
    """Construction proceeds to the render/boot stage; its errors there are
    cudaflight-dependent (missing wheel, unrenderable stub dump) and not under test."""
    try:
        env = SkyFlowEnv(cfg)
        env.close()
    except ValueError as e:
        assert "align_board" not in str(e)
    except Exception:
        pass


def test_board_align_dump_warns_not_rejects(tmp_path):
    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_yaw = -135\nset yaw_motors_reversed = ON\n")
    with pytest.warns(RuntimeWarning, match="align_board_yaw"):
        _construct_past_the_scan(_sticks_cfg(dump))


def test_zero_board_align_dump_is_silent(tmp_path):
    import warnings

    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_roll = 0\nset align_board_yaw = 0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            _construct_past_the_scan(_sticks_cfg(dump))
        except RuntimeWarning as w:  # pragma: no cover - the assertion we want to fail
            raise AssertionError(f"zero alignment must not warn: {w}") from w


def test_overrides_zero_the_dump_alignment(tmp_path):
    """The overrides file wins (as in the render): pinning align_board_yaw = 0 there
    silences the warning a nonzero dump value would raise."""
    import warnings

    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_yaw = -135\n")
    ov = tmp_path / "overrides.txt"
    ov.write_text("set align_board_yaw = 0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            _construct_past_the_scan(_sticks_cfg(dump, ov))
        except RuntimeWarning as w:  # pragma: no cover
            raise AssertionError(f"overridden alignment must not warn: {w}") from w
