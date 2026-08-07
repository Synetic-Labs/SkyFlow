"""Estimation-facing specific-force adapter over the SkyFlow plant (code of record).

An EqF's model channel (MonoRace arXiv 2601.15222 Eqs. 1–2 substitution; a drag-based
velocity pseudo-measurement) needs body-frame specific force from motor commands and
body velocity — the ``SpecificForcePredictor`` protocol a downstream estimator expects.
This adapter computes it with the SkyFlow plant's OWN equations (``plant._deriv`` via
``step``/``specific_force_body``) and the code-of-record parameters
(``params.airframe_params``): no duplicated physics, no copied constants —
a sysid improvement to the twin transfers to the filter with zero copying. It replaces
an earlier crazyflow-parameterised predictor whose hand-carried constants had
drifted from the identified twin (lateral drag −33% at 15 m/s, reversed motor-lag
asymmetry, convex thrust map).

Internal state is ONLY the plant's motor rows + wake state. Their dynamics depend on
(command, rotor speed) alone — never on pose or velocity — so advancing them on a state
with zeroed pose/velocity/rate rows is EXACT, and ``predict`` rebuilds the state with
the filter's velocity under an identity attitude (the plant's Rᵀ·v_world then IS the
body velocity). Frames: the plant is FLU/Z-up, the filter FRD/NED — the (x, −y, −z)
flip on the way in (velocity) and out (specific force), the same seam as
``plant.synth_sensors``. Motor ordering: collective thrust, drag and the wake state are
all permutation-invariant in the rotor index (ΣW², Σ(W·Ẇ)) and the per-rotor lag is
identical across motors, so no command-slot→rotor permutation is needed — only the
moment model (unused here) is order-sensitive.

Timing: the filter advances per IMU sample (~11 ms). RK4 substeps are capped at
``substep_s`` so straggler gaps stay accurate (first-order lags τ ≥ 20 ms; at
dt/τ ≈ 0.55 a single RK4 step is already ~4e-4 relative). jit dispatch is ~50 µs per
call — nothing at 90 Hz.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import plant
from .params import _PARAM_KEYS, PlantParams, airframe_params

_KEY_IDX = {k: i for i, k in enumerate(_PARAM_KEYS)}
_FRD_FLU = np.array([1.0, -1.0, -1.0])  # involution: FRD↔FLU body / NED↔Z-up world


class SkyflowSpecificForce:
    """``SpecificForcePredictor`` over the SkyFlow plant.

    ``advance(dt_s, motors_01)`` steps the rotor-lag + wake states toward the commanded
    speeds (the plant's own RK4); ``predict(motors_01, v_body)`` returns FRD specific
    force from the CURRENT rotor state. A caller that never advances gets the static
    steady-state curve — the same opt-in semantics the crazyflow-era model had, so
    stateless offline uses stay reproducible. The first ``advance`` seeds the rotors at
    the commanded steady state (there is no earlier command to lag from).

    ``airframe`` names the coefficient set to pull from ``params.AIRFRAME_PARAMS``
    (default the Air75 II Racer); pass ``params`` to supply a :class:`PlantParams`
    directly instead, which takes precedence.
    """

    def __init__(self, airframe: str = "air75_ii_racer",
                 params: PlantParams | None = None,
                 substep_s: float = 0.004, warmup: bool = True) -> None:
        self.plant_params = params if params is not None else airframe_params(airframe)
        self._p = np.asarray(self.plant_params.to_array())[None, :]   # [1, 46]
        self._substep_s = float(substep_s)
        self._w: jax.Array | None = None                              # normalised [1, 4]
        self._wake = jnp.zeros((1, 1), jnp.float32)
        self._step_j = jax.jit(plant.step)
        self._spec_j = jax.jit(plant.specific_force_body)
        if warmup:                                    # compile off the hot path
            state = self._state(jnp.zeros((1, 3), jnp.float32), self._target_w(np.zeros(4)))
            self._step_j(state, jnp.zeros((1, 4), jnp.float32), self._p, 1e-3)
            self._spec_j(state, self._p)

    def _target_w(self, motors_01: np.ndarray) -> jax.Array:
        """Commanded steady-state rotor speeds as the plant's normalised motor state."""
        cols = tuple(self._p[:, _KEY_IDX[n]] for n in
                     ("k", "k_w", "w_min", "w_max", "sc_tmax", "sc_u50", "sc_p"))
        wc = plant.commanded_rotor_speed(jnp.asarray(motors_01, jnp.float32)[None, :], *cols)
        return ((wc - plant._W_MIN_N) / (plant._W_MAX_N - plant._W_MIN_N)) * 2.0 - 1.0

    def _state(self, v_flu: jax.Array, w: jax.Array) -> jax.Array:
        """Plant state carrying only what specific force needs: identity attitude and
        zero rates/position, so the velocity row is the body velocity verbatim."""
        zero3 = jnp.zeros((1, 3), jnp.float32)
        quat = jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
        return plant.make_state(zero3, v_flu, quat, zero3, w, self._wake)

    def advance(self, dt_s: float, motors_01: np.ndarray) -> None:
        """Advance the rotor-lag + wake states ``dt_s`` toward the commanded speeds."""
        motors = np.asarray(motors_01, dtype=np.float64)
        if motors.shape != (4,):
            raise ValueError(f"motors_01 must be [4], got {motors.shape}")
        if self._w is None:
            self._w = self._target_w(motors)
            return
        if dt_s <= 0.0:
            return
        n = max(1, int(np.ceil(float(dt_s) / self._substep_s)))
        sub = float(dt_s) / n
        u = jnp.asarray(motors, jnp.float32)[None, :]
        state = self._state(jnp.zeros((1, 3), jnp.float32), self._w)
        for _ in range(n):
            state = self._step_j(state, u, self._p, sub)
        self._w = state[:, 13:17]     # motor + wake rows are velocity-independent —
        self._wake = state[:, 17:18]  # the zero-velocity advance is exact for them

    def get_state(self) -> tuple[jax.Array | None, jax.Array]:
        """Opaque internal checkpoint (rotor speeds + wake) for filter OOSM rollback.

        jax arrays are immutable, so sharing references is safe — no copies needed.
        """
        return (self._w, self._wake)

    def set_state(self, state: tuple[jax.Array | None, jax.Array]) -> None:
        """Restore a checkpoint taken by :meth:`get_state`."""
        self._w, self._wake = state

    def predict(self, motors_01: np.ndarray, v_body: np.ndarray) -> np.ndarray:
        """Body-frame specific force (m/s², FRD) at the current rotor state."""
        motors = np.asarray(motors_01, dtype=np.float64)
        vel_frd = np.asarray(v_body, dtype=np.float64)
        if motors.shape != (4,):
            raise ValueError(f"motors_01 must be [4], got {motors.shape}")
        if vel_frd.shape != (3,):
            raise ValueError(f"v_body must be [3], got {vel_frd.shape}")
        w = self._w if self._w is not None else self._target_w(motors)
        v_flu = jnp.asarray(vel_frd * _FRD_FLU, jnp.float32)[None, :]
        spec_flu = np.asarray(self._spec_j(self._state(v_flu, w), self._p)[0],
                              dtype=np.float64)
        return spec_flu * _FRD_FLU
