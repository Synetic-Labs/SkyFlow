"""
DESIGN.md §6 — the correlated factor stage of sample_params: legacy bit-exactness,
ratio preservation inside a factor group, asymmetric limits, the two physical guards
(thrust-to-weight floor, shape-relative planar inertia cap), residual independence,
and loud validation. CPU-only: everything asserts on the sampled parameter rows.
"""

import copy

import jax
import numpy as np
import pytest

from skyflow.dynamics import N_ROTORS, pack_params, param_slices
from skyflow.env import DomainRand
from skyflow.params import (
    AIRFRAMES,
    FACTOR_LIMITS,
    TW_FLOOR,
    Airframe,
    factor_floor,
    register_airframe,
    sample_params,
)

CF = AIRFRAMES["crazyflie"]
SL = param_slices(N_ROTORS)
KEY = jax.random.PRNGKey(7)
FLEET = 4096

#: brackets that silence every per-entry residual — factor-stage effects in isolation.
NO_RESIDUALS = {
    k: 0.0
    for k in ("mass", "inertia", "ct0", "ct1", "ct2", "cq0", "cq1", "cq2", "tau_m",
              "ka1", "ka2", "kd1", "kd2", "I_rot", "c_D", "c_L", "k_d", "k_z",
              "k_flap", "k_h", "k_angle", "k_hor", "k_v2")
}


def _rows(scale=1.0, brackets=None, factors=None, airframe=CF, fleet=FLEET):
    return np.asarray(sample_params(KEY, airframe, fleet, scale, brackets, factors))


def test_factors_none_is_the_legacy_sampler_bit_exact():
    nominal = np.asarray(pack_params(CF.values), np.float32)
    legacy = _rows(scale=0.0, factors=None)
    assert (legacy == nominal).all()
    # And with jitter on, factors=None consumes the key exactly as before: the draw
    # is reproducible from the same key (regression anchor for resume paths).
    a = _rows(scale=1.0)
    b = _rows(scale=1.0)
    assert (a == b).all()


def test_scale_zero_is_nominal_for_both_stages():
    nominal = np.asarray(pack_params(CF.values), np.float32)
    rows = _rows(scale=0.0, factors={})
    # Includes the guards: neither may bind on the unjittered nominal (the crazyflie
    # nominal already sits ABOVE the textbook planar limit — the cap is shape-relative).
    assert (rows == nominal).all()


def test_group_preserves_torque_thrust_ratio_per_rotor():
    rows = _rows(brackets=NO_RESIDUALS, factors={})
    ct2, cq2 = rows[:, SL["ct2"]], rows[:, SL["cq2"]]
    ratio = cq2 / ct2
    nominal_ratio = CF.values["cq2"][0] / CF.values["ct2"][0]
    np.testing.assert_allclose(ratio, nominal_ratio, rtol=1e-5)
    # while the shared factor really moves the coefficients across worlds:
    assert ct2.std() / ct2.mean() > 0.05


def test_asymmetric_limits_are_respected_and_reached():
    rows = _rows(brackets=NO_RESIDUALS, factors={})
    lo, hi = FACTOR_LIMITS["air_prop"]
    rel = rows[:, SL["ct2"][0]] / CF.values["ct2"][0] - 1.0
    assert rel.min() >= lo - 1e-5 and rel.max() <= hi + 1e-5
    assert rel.min() < lo + 0.05 and rel.max() > hi - 0.05  # both tails reached


def test_never_jitter_keys_stay_exact():
    nominal = np.asarray(pack_params(CF.values), np.float32)
    rows = _rows(factors={})
    for key in ("spin", "axis", "grav"):
        assert (rows[:, SL[key]] == nominal[SL[key]]).all()


def test_thrust_to_weight_floor_holds():
    # Heavy payload draws against a thin-air/worn-prop day: the corner the guard owns.
    rows = _rows(brackets=NO_RESIDUALS,
                 factors={"mass": (0.0, 0.9), "air_prop": (-0.3, -0.3)})
    w = CF.rotor_speed_max
    thrust = (rows[:, SL["ct0"]].sum(-1) + rows[:, SL["ct1"]].sum(-1) * w
              + rows[:, SL["ct2"]].sum(-1) * w * w)
    tw = thrust / (rows[:, SL["mass"]][:, 0] * rows[:, SL["grav"]][:, 0])
    assert tw.min() >= TW_FLOOR - 1e-3
    # and the guard clamps, not zeroes: mass never falls below nominal.
    assert rows[:, SL["mass"]].min() >= CF.values["mass"] - 1e-9


def test_planar_inertia_cap_is_shape_relative():
    rows = _rows(brackets={**NO_RESIDUALS, "inertia": 0.4}, factors={})
    xx, yy, zz = (rows[:, SL["inertia"][i]] for i in range(3))
    I = CF.values["inertia"]
    nominal_ratio = I[2][2] / (I[0][0] + I[1][1])
    assert nominal_ratio > 1.0  # the crazyflie sits above the textbook limit
    assert (zz <= (xx + yy) * nominal_ratio * (1 + 1e-5)).all()
    assert (zz / (xx + yy)).max() > nominal_ratio * 0.999  # and the cap is reachable


def test_residuals_still_give_rotor_asymmetry():
    rows = _rows(brackets={**NO_RESIDUALS, "cq2": 0.02}, factors={})
    cq2 = rows[:, SL["cq2"]]
    spread = cq2.max(axis=1) / cq2.min(axis=1) - 1.0
    assert spread.max() > 0.02  # per-rotor mismatch present
    assert spread.max() < 0.09  # but bounded by the residual width (< ~2*0.02 + slack)


def test_unknown_group_and_inverted_limits_are_loud():
    with pytest.raises(ValueError, match="unknown group"):
        _rows(factors={"bogus": (0.0, 0.1)})
    with pytest.raises(ValueError, match=r"lo .* > hi"):
        _rows(factors={"mass": (0.3, -0.3)})
    assert factor_floor(None) == 0.0
    assert factor_floor({}) == pytest.approx(0.30)  # air_prop lo is the deepest default


def test_domainrand_carries_factors():
    dr = DomainRand(factors={"air_prop": (-0.1, 0.1)})
    assert dr.effective().factors == {"air_prop": (-0.1, 0.1)}  # limits are not scaled
    assert dr.off().factors == {"air_prop": (-0.1, 0.1)}  # off() zeroes scale, not shape


def test_low_twr_airframe_is_not_repaired_by_the_guard():
    # An airframe whose NOMINAL thrust-to-weight is below the floor stays what it is:
    # the guard bounds the draw, never the vehicle.
    if "test_low_twr" not in AIRFRAMES:
        values = copy.deepcopy(CF.values)
        values["mass"] = CF.values["mass"] * 2.0  # nominal T/W ~0.98
        register_airframe(
            "test_low_twr",
            Airframe(name="test_low_twr", values=values,
                     rotor_speed_min=CF.rotor_speed_min,
                     rotor_speed_max=CF.rotor_speed_max,
                     throttle_k=CF.throttle_k),
        )
    af = AIRFRAMES["test_low_twr"]
    rows = _rows(scale=0.0, factors={}, airframe=af, fleet=8)
    assert rows[:, SL["mass"]][0, 0] == np.float32(af.values["mass"])
