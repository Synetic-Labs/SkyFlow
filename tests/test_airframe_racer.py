"""
The default airframe, racer5in (params.RACER_5IN): a 5-inch racer whose values trace to
one real platform. Checks the properties its provenance promises — it is the default,
hovers at the published motor speed, has the published thrust ceiling, and damps body
rates (saturated motors level off instead of spinning up without bound).
"""

import jax
import jax.numpy as jnp
import numpy as np

from skyflow.dynamics import substep
from skyflow.env import SimConfig
from skyflow.params import AIRFRAMES, RACER_5IN, pack_params

AF = AIRFRAMES["racer5in"]
CT = RACER_5IN["ct2"][0]


def test_racer_is_the_default_airframe():
    assert SimConfig().airframe == "racer5in"


def test_hover_speed_and_thrust_ceiling_match_the_platform():
    w_hover = np.sqrt(RACER_5IN["mass"] * RACER_5IN["grav"] / (4 * CT))
    assert 1080 < w_hover < 1110  # the flight data hovers at ~1120 rad/s
    assert np.isclose(CT * AF.rotor_speed_max**2, 8.5)  # the per-motor thrust cap
    thrust_to_weight = 4 * 8.5 / (RACER_5IN["mass"] * RACER_5IN["grav"])
    assert 4.4 < thrust_to_weight < 4.6


def _rate_after(seconds: float, command, w0: float) -> float:
    params = jnp.asarray(pack_params(AF.values))[None]
    plant = jnp.zeros((1, 17), jnp.float32).at[:, 6].set(1.0).at[:, 2].set(10.0)
    plant = plant.at[:, 13:17].set(w0)
    cmd = jnp.asarray([command], jnp.float32)
    zero = jnp.zeros((1, 3), jnp.float32)
    dt = 1.0 / 1000.0
    step = jax.jit(lambda p: substep(p, cmd, zero, zero, zero, params, dt,
                                     AF.rotor_speed_min, AF.rotor_speed_max))
    for _ in range(int(seconds / dt)):
        plant = step(plant)
    return float(jnp.linalg.norm(plant[0, 10:13]))


def test_saturated_motors_level_off_instead_of_spinning_up():
    lo, hi = AF.rotor_speed_min, AF.rotor_speed_max
    w_hover = float(np.sqrt(RACER_5IN["mass"] * RACER_5IN["grav"] / (4 * CT)))
    # undamped, this torque spins the vehicle up by ~2000 rad/s every second without end;
    # rotor damping balances it near 340 rad/s (time constant ~0.8 s)
    r4, r6 = (_rate_after(t, [lo, hi, hi, hi], w_hover) for t in (4.0, 6.0))
    assert np.isfinite(r6) and r6 < 500.0
    assert abs(r6 - r4) < 0.03 * r4  # levelled off


def test_values_come_from_skyflow_dynamics():
    """One source for the racer's values: SkyFlow-Dynamics' traced reference vehicle."""
    from skyflow_dynamics.spec.parameters import RACER_5IN as SFD_RACER_5IN

    assert RACER_5IN is SFD_RACER_5IN
    assert AIRFRAMES["racer5in"].values == SFD_RACER_5IN
