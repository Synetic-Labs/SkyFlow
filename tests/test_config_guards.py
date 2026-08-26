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


# -- eeprom dump settings the sensor packaging cannot honor (F8, C13/F9) ---------------
# The scans run before any cudaflight import, so these tests run everywhere.


def _sticks_cfg(dump_path) -> SimConfig:
    return SimConfig(num_envs=1, control="sticks", firmware="cpu", eeprom=str(dump_path))


def test_board_align_dump_rejected(tmp_path):
    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_yaw = 90\nset motor_pwm_protocol = PWM\n")
    with pytest.raises(ValueError, match="align_board_yaw"):
        SkyFlowEnv(_sticks_cfg(dump))


def test_zero_board_align_dump_passes_the_scan(tmp_path):
    """align_board_* = 0 (every stock dump) must NOT trip the guard — construction
    proceeds to the render/boot stage (whose errors are cudaflight-dependent)."""
    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_roll = 0\nset align_board_yaw = 0\n")
    try:
        env = SkyFlowEnv(_sticks_cfg(dump))
        env.close()  # only reached with cudaflight present AND the dump renderable
    except ValueError as e:
        assert "align_board" not in str(e)
    except Exception:
        pass  # ImportError / render failures are fine — the scan let it through


def test_board_align_in_overrides_rejected(tmp_path):
    dump = tmp_path / "dump.txt"
    dump.write_text("set align_board_yaw = 0\n")
    ov = tmp_path / "overrides.txt"
    ov.write_text("set align_board_pitch = -45\n")
    cfg = SimConfig(num_envs=1, control="sticks", firmware="cpu",
                    eeprom=str(dump), eeprom_overrides=str(ov))
    with pytest.raises(ValueError, match="align_board_pitch"):
        SkyFlowEnv(cfg)
