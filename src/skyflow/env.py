"""
The platform — SimConfig and SkyFlowEnv (DESIGN.md §7, §8).

SkyFlowEnv is the fleet-batched harness around the generated dynamics: it schedules
substeps, applies command transport delay, drives the OU wind and poke disturbances,
applies the ground-contact heuristic, detects crashes, and auto-resets finished worlds
in-jit. It contains no physics and no reward code: forces, torques and sensors come from
skyflow-dynamics through dynamics.py/sensors.py, and rewards live in the task
(DESIGN.md §1, §9).

`reset`, `step` and `metrics` are pure functions of their inputs — the caller jits.
Every array the env creates is float32 with the fleet axis [F, ...] leading; frames are
world z-up, body FLU, quaternions wxyz Hamilton body→world (DESIGN.md §3).

Step pipeline (order is normative, DESIGN.md §7):

 1. split the key; push the action into the delay ring, read the delayed action per world
 2. command map (motors mode: u = (a+1)/2 → verified throttle curve → Ω_c)
 3. OU gust advance (exact discretization: decay exp(-dt/τ) + matched kick); the wind
    every consumer sees is the per-episode steady mean (DomainRand trait) + the gust
 4. poke sampling (world-frame F_ext, body τ_ext through the backend's exogenous inputs —
    velocity state is never written directly)
 5. `decimation` x [generated substep + ground contact §8], all inputs zero-order-held
    (sticks mode re-derives Ω_c from the firmware every 1 kHz substep, feeding it
    DomainRand-corrupted sensor rows, §10)
 6. airborne latch; env crash set; task `evaluate` on the transition
 7. terminated / truncated / done
 8. IMU measurement (DomainRand bias + noise) + task `observe` + DomainRand obs noise
 9. in-jit auto-reset of done worlds via `tree_where` blending (fresh params, traits,
    delay draw); the pre-reset observation goes to info["final_obs"], the pre-reset
    flags to info["terminated"/"truncated"]
10. episode bookkeeping EMAs for `metrics`: the SimState EMA leaves (outcome fractions,
    completed-episode return/length) update from the pre-reset done rows, decayed
    `_METRICS_EMA_DECAY` per completed episode; no-op when nothing finished
"""

import inspect
import math
import re
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp

from skyflow import dynamics, sensors
from skyflow.params import AIRFRAMES, factor_floor, max_bracket, sample_params
from skyflow.types import Array, DRState, FirmwareFleet, SimState, StepInfo, Task

__all__ = ["DomainRand", "SimConfig", "SkyFlowEnv", "tree_where"]

#: Near-ground band, metres: below it a descending/tilted airborne vehicle is a ground
#: crash; above it the airborne latch sets (DESIGN.md §7 step 6 uses 0.05 for both).
_GROUND_BAND_M = 0.05

#: Descent speed at which touching down inside the band counts as a crash, m/s (§7).
_CRASH_DESCENT_MPS = 1.0

#: Episode-bookkeeping EMA decay per COMPLETED episode (§7 step 10): a step finishing
#: n episodes keeps `decay**n` of the old EMA and blends in the done-row mean.
_METRICS_EMA_DECAY = 0.99


@dataclass(frozen=True)
class DomainRand:
    """
    All training-robustness randomization in one object (DESIGN.md §7). One instance is
    one robustness setting: a curriculum or an outer controller produces DomainRand
    values, the env consumes them. Every magnitude is in physical units. Randomness
    enters only at loop boundaries — the params row, exogenous forces, the delay ring,
    the sensor rows, the observation vector; the ODE and the firmware stay exact.

    Draw classes: traits (once per episode, constant within it — body params, steady
    wind, IMU bias, transport delay), processes (every sample — gusts, sensor noise,
    obs noise) and events (pokes). Spawn spread is task variety, not model error.

    `scale` multiplies every continuous magnitude; it never touches time constants
    (`wind_tau_s`), event rates (`poke_prob`), integer delays (`delay_steps`) or the
    spawn spread (`spawn_scale`). scale=0 with delay_steps=(0,0) is bit-exact nominal —
    `off()` returns that setting. Defaults reproduce a plain SimConfig(): body DR on,
    everything else off.
    """

    scale: float = 1.0  # master dial over every continuous magnitude

    # -- body: the vehicle (trait, §6 sample_params) --------------------------------
    body_scale: float = 1.0  # multiplies the bracket half-widths and factor limits
    brackets: dict | None = None  # per-key half-width overrides (§6); base table is
    # DR_BRACKETS, or RESIDUAL_BRACKETS once `factors` is on
    factors: dict | None = None  # correlated factor stage (§6): None = off (legacy
    # independent draws, bit-exact); {} = on with FACTOR_LIMITS defaults;
    # {group: (lo, hi)} overrides one group's limits. Measured 2026-08-22: at equal
    # ±20% width on ct2+cq2 the factor structure flies 0.88, independent draws 0.12.

    # -- world: wind and shocks ------------------------------------------------------
    wind_mean_mps: float = 0.0  # trait: steady horizontal wind, magnitude ceiling, m/s
    wind_gust_mps: float = 0.0  # process: OU gust stationary std per axis, m/s
    wind_tau_s: float = 0.5  # OU gust correlation time, s (a clock — never scaled)
    poke_prob: float = 0.0  # event rate per control step per world (never scaled)
    poke_force_n: float = 0.0  # world-frame poke force magnitude ceiling, N
    poke_torque_nm: float = 0.0  # body-frame poke torque magnitude ceiling, N·m

    # -- actuation: command transport (trait) ----------------------------------------
    delay_steps: tuple[int, int] = (0, 0)  # (min, max) control steps (never scaled)
    # trait: battery voltage sag — per-episode rotor-speed-ceiling factor
    # 1 - U(0, battery_sag) on the airframe's rotor_speed_max (one-sided: a pack only
    # sags). Measured on the Air75 II Racer battery_hover sysid: -9.6% RPM per motor
    # command over 3.71->3.30 V; ~13% across a full pack. The firmware sees nothing —
    # full stick simply buys less rotor speed, exactly like a tired pack.
    battery_sag: float = 0.0

    # -- sensing: the IMU/baro rows the firmware and IMU-observing tasks consume ------
    gyro_noise_rps: float = 0.0  # process: white per-sample std, rad/s
    accel_noise_mps2: float = 0.0  # process: white per-sample std, m/s²
    gyro_bias_rps: float = 0.0  # trait: constant per-axis bias half-width, rad/s
    accel_bias_mps2: float = 0.0  # trait: constant per-axis bias half-width, m/s²
    baro_noise_pa: float = 0.0  # process: white per-sample std on the baro row, Pa

    # -- observation: the policy vector (applied by the env after task.observe) -------
    obs_noise: float = 0.0  # process: additive uniform half-width

    # -- initial state: task variety, forwarded to builders as spawn_dr_scale ---------
    spawn_scale: float = 1.0

    def off(self) -> "DomainRand":
        """This setting with all model error and corruption off: bit-exact nominal."""
        return replace(self, scale=0.0, delay_steps=(0, 0))

    def effective(self) -> "DomainRand":
        """This setting with `scale` folded into every continuous magnitude (scale=1)."""
        s = self.scale
        return replace(
            self,
            scale=1.0,
            body_scale=s * self.body_scale,
            wind_mean_mps=s * self.wind_mean_mps,
            wind_gust_mps=s * self.wind_gust_mps,
            poke_force_n=s * self.poke_force_n,
            poke_torque_nm=s * self.poke_torque_nm,
            gyro_noise_rps=s * self.gyro_noise_rps,
            accel_noise_mps2=s * self.accel_noise_mps2,
            gyro_bias_rps=s * self.gyro_bias_rps,
            accel_bias_mps2=s * self.accel_bias_mps2,
            baro_noise_pa=s * self.baro_noise_pa,
            obs_noise=s * self.obs_noise,
            battery_sag=s * self.battery_sag,
        )


@dataclass(frozen=True)
class SimConfig:
    """
    Platform configuration (DESIGN.md §7). Frozen and plain — no omegaconf/hydra; build
    variants with `dataclasses.replace`. Physics advances at `physics_hz`; the policy
    acts at `control_hz`; `decimation = round(physics_hz / control_hz)` substeps run per
    control step with all inputs zero-order-held. All randomization lives in `dr`.
    """

    num_envs: int = 1024
    task: str = "hover"
    task_kwargs: dict = field(default_factory=dict)
    airframe: str = "crazyflie"
    control: str = "motors"  # "motors" | "sticks" (DESIGN.md §10)
    firmware: str = "auto"  # sticks backend: "auto" | "cpu" | "gpu" (DESIGN.md §10)
    # sticks firmware config: path to a drone's Betaflight CLI `dump all` file. The env
    # renders it into the boot eeprom at construction (cudaflight.render_eeprom — a
    # version-gated strict round-trip, so a dump from another firmware release or a
    # line that does not hold raises here, at construction). None boots the wheel's
    # stock defaults. CLI text is the config source of truth; the rendered image is a
    # derived artifact (examples/configs/README.md). `eeprom_overrides` is an optional
    # file of sim-only CLI lines appended after the dump (e.g. blackbox_device = NONE).
    # The dump header also SELECTS the firmware base: a dump built from another
    # Betaflight base than the installed wheel picks the matching per-base binaries
    # from the cudaflight bundle cache (fetch once: `python -m cudaflight.bases
    # <rev>`; needs cudaflight >= 0.6.0) — no reinstall, one wheel flies every drone.
    eeprom: str | None = None
    eeprom_overrides: str | None = None
    control_hz: float = 100.0
    physics_hz: float = 1000.0  # sticks mode requires exactly 1000 (§10: 1 kHz fw tick)
    differentiable: bool = False  # raises NotImplementedError("planned") if True
    # randomization / disturbance — the DomainRand block, one object = one setting
    dr: DomainRand = field(default_factory=DomainRand)
    # episode / safety
    max_episode_steps: int = 1000
    stuck_steps: int = 200  # never-airborne worlds truncate after this many steps
    bounds_xy_m: float = 20.0
    bounds_z_m: float = 8.0
    max_speed_mps: float = 30.0
    max_rate_rps: float = 50.0
    ground_tilt_limit_rad: float = math.pi / 3


def _installed_base_rev() -> "str | None":
    """The Betaflight base of the INSTALLED cudaflight wheel, from its version
    metadata (`0.6.0+bf.6dbc4218` → `6dbc4218`). None for source checkouts. Metadata
    is trustworthy here because it describes the wheel's own embedded binaries; the
    render gate separately verifies whichever lib actually boots."""
    try:
        from importlib.metadata import version

        m = re.search(r"\+bf\.([0-9a-f]+)", version("cudaflight"))
        return m.group(1) if m else None
    except Exception:
        return None


def _resolve_firmware_base(dump: Path) -> Any:
    """The per-base firmware bundle a dump selects, or None for the installed base.

    The dump header names the Betaflight base the drone's firmware was built from.
    When it matches the installed wheel (or carries no revision), the wheel's own
    binaries serve and this returns None. Otherwise the cudaflight bundle cache
    provides the matching (libcpuflight.so, fw.fatbin) pair — fetched once with
    `python -m cudaflight.bases <rev>`; a cache miss raises with that command. The
    render's version gate verifies the selection against the booted lib either way.
    """
    from cudaflight import bases  # cudaflight >= 0.6.0
    from cudaflight.config import parse_header

    header = parse_header(dump.read_text(errors="replace"))
    rev = header["rev"] if header else None
    if rev is None or rev == "norevision":
        return None
    installed = _installed_base_rev()
    if installed is not None and bases.rev_match(installed, rev):
        return None
    return bases.paths(rev)


def tree_where(done: Array, fresh: Any, current: Any) -> Any:
    """
    Row-wise pytree blend: leaf rows from `fresh` where `done`, else from `current`.

    `done` is [F] bool; every leaf of both trees leads with the fleet axis (DESIGN.md §4).
    This is the auto-reset blend of the step pipeline (§7 step 9): done worlds take the
    freshly spawned leaves, live worlds keep theirs, with no host round-trip.
    """

    def sel(a: Array, b: Array) -> Array:
        return jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)

    return jax.tree.map(sel, fresh, current)


class _FirmwareCarry(NamedTuple):
    """
    Sticks-mode `SimState.task_state` wrapper: the task's own pytree plus the
    value-threaded firmware pair of `types.FirmwareFleet`. The env wraps/unwraps it
    around every task call, so tasks never see it; motors mode stores the task pytree
    bare. (SimState §4 has no firmware slot, so the pair rides in the one opaque slot
    the env owns end to end.)
    """

    task: Any
    blob: Any
    fwstate: Any


def _uniform_ball(key: Array, n: int) -> Array:
    """[n,3] f32 points uniform in the unit ball (direction · radius^(1/3) law)."""
    k_dir, k_r = jax.random.split(key)
    v = jax.random.normal(k_dir, (n, 3), jnp.float32)
    v = v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    r = jax.random.uniform(k_r, (n, 1), jnp.float32) ** (1.0 / 3.0)
    return v * r


def _ground_contact(plant: Array) -> Array:
    """
    Ground contact per DESIGN.md §8 — harness bookkeeping, not physics. Where z ≤ 0:
    clamp z = 0, clamp v_z = max(v_z, 0), zero in-plane velocity and body rates, hold the
    quaternion and rotor speeds. This is the registry's `ground_contact_heuristic`
    (candidate, harness) — replaceable by a real contact model through the
    SkyFlow-Dynamics spec later.
    """
    on = plant[:, 2] <= 0.0
    z = jnp.where(on, 0.0, plant[:, 2])
    v_xy = jnp.where(on[:, None], 0.0, plant[:, 3:5])
    v_z = jnp.where(on, jnp.maximum(plant[:, 5], 0.0), plant[:, 5])
    w = jnp.where(on[:, None], 0.0, plant[:, 10:13])
    return jnp.concatenate(
        [plant[:, 0:2], z[:, None], v_xy, v_z[:, None], plant[:, 6:10], w, plant[:, 13:17]],
        axis=-1,
    )


def _ground_impact(plant: Array, cos_tilt_min: float) -> Array:
    """
    Ground-crash predicate rows [F] on a PRE-clamp substep state: inside the near-ground
    band and descending faster than the crash limit or tilted past the limit. Sampled
    every substep because the contact clamp (§8) zeroes the descent evidence — at control
    rate alone, a fall faster than ~5 m/s crosses the whole band between two checks.
    """
    q = plant[:, 6:10]
    cos_tilt = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # ẑ_B · ẑ_W for a unit quaternion
    hard = -plant[:, 5] > _CRASH_DESCENT_MPS
    return (plant[:, 2] < _GROUND_BAND_M) & (hard | (cos_tilt < cos_tilt_min))


class SkyFlowEnv:
    """
    Fleet-batched quadrotor simulation platform (DESIGN.md §7).

    Natively batched: `reset(key)` and `step(state, action)` operate on all
    `cfg.num_envs` worlds at once and are pure — jit them (`jax.jit(env.step)`), scan
    them, checkpoint the SimState pytree. Done worlds respawn in-jit with fresh task
    spawn, fresh domain-randomized params and a fresh delay draw; the pre-reset
    observation and flags are returned through `info` (final_obs / terminated /
    truncated), so training code never sees a dead state.

    Exposes `fleet`, `obs_spec`, `obs_dim`, `act_dim` (=4), `image_shape`, `decimation`,
    `dt_control` and `dt_physics`. Actions are [F,4] in [-1,1] (clipped defensively):
    motor throttles in motors mode, AETR sticks in sticks mode (§10).

    info keys beyond the pre-reset flags and observation, all [F]: `poke_active` (this
    step's poke draw) and `ep_return`/`ep_len` (pre-reset episode accumulators, valid on
    done rows — SimState has no post-hoc episode storage, so done-step stats surface
    here), plus whatever the task's `evaluate` info carries.
    """

    def __init__(
        self,
        cfg: SimConfig,
        *,
        task: Task | None = None,
        firmware_fleet: FirmwareFleet | None = None,
        motor_perm: Sequence[int] = (3, 1, 0, 2),
    ) -> None:
        """
        Args:
          cfg: frozen platform configuration.
          task: pre-built Task instance; None builds `cfg.task` through the
            `skyflow.tasks` registry with `cfg.task_kwargs` (forwarding the env-owned
            `spawn_dr_scale` and `control_hz` to builders that name them).
          firmware_fleet: injected `types.FirmwareFleet` for control="sticks",
            overriding `cfg.firmware`. None builds the backend `cfg.firmware` selects
            (§10): "cpu"/"gpu" force one, "auto" picks the GPU fleet when fleet >= 3
            and a CUDA device is visible, else the CPU SITL fleet. Construction raises
            ImportError with install guidance when the cudaflight wheel is absent.
            Motors mode ignores it.
          motor_perm: sticks mode only — sim rotor i takes firmware motor
            `motor_perm[i]`. The default maps Betaflight QUADX output order
            (RR, FR, RL, FL) onto the built-in airframes' rotor order (FL, FR, RR, RL).
        """
        if cfg.differentiable:
            raise NotImplementedError("planned")
        if cfg.control not in ("motors", "sticks"):
            raise ValueError(f'cfg.control must be "motors" or "sticks", got {cfg.control!r}')
        if cfg.eeprom is None and cfg.eeprom_overrides is not None:
            raise ValueError(
                "cfg.eeprom_overrides is set without cfg.eeprom — overrides are "
                "sim-only CLI lines appended to a dump, not a config by themselves"
            )
        if cfg.eeprom is not None and cfg.control != "sticks":
            raise ValueError('cfg.eeprom requires control="sticks": motors mode boots no firmware')
        if cfg.airframe not in AIRFRAMES:
            raise ValueError(
                f"unknown airframe {cfg.airframe!r}; registered: {sorted(AIRFRAMES)}"
            )
        if cfg.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {cfg.num_envs}")
        dr = cfg.dr.effective()  # fold the master scale once; the env reads only this
        d_min, d_max = (int(d) for d in dr.delay_steps)
        if not 0 <= d_min <= d_max:
            raise ValueError(f"need 0 <= delay min <= max, got dr.delay_steps={dr.delay_steps}")
        if dr.wind_tau_s <= 0.0:
            raise ValueError(f"dr.wind_tau_s must be > 0, got {dr.wind_tau_s}")
        if not 0.0 <= dr.poke_prob <= 1.0:
            raise ValueError(f"dr.poke_prob must be in [0,1], got {dr.poke_prob}")
        for name in (
            "scale", "body_scale", "wind_mean_mps", "wind_gust_mps", "poke_force_n",
            "poke_torque_nm", "gyro_noise_rps", "accel_noise_mps2", "gyro_bias_rps",
            "accel_bias_mps2", "baro_noise_pa", "obs_noise", "spawn_scale",
        ):
            if getattr(cfg.dr, name) < 0.0:
                raise ValueError(f"dr.{name} must be >= 0, got {getattr(cfg.dr, name)}")
        # Loud key/group validation happens inside max_bracket and factor_floor.
        b_max = max_bracket(dr.brackets, residual=dr.factors is not None)
        if dr.body_scale * b_max >= 1.0:
            raise ValueError(
                f"dr.scale·dr.body_scale·max bracket = {dr.body_scale * b_max:.3f} >= 1: "
                "a multiplicative factor could reach zero or flip a physical parameter"
            )
        f_low = factor_floor(dr.factors)
        if dr.body_scale * f_low >= 1.0:
            raise ValueError(
                f"dr.scale·dr.body_scale·max factor low = {dr.body_scale * f_low:.3f} >= 1: "
                "a shared factor could reach zero or flip a physical parameter"
            )
        if not 0.0 <= dr.battery_sag < 1.0:
            raise ValueError(
                f"dr.scale·dr.battery_sag must be in [0, 1), got {dr.battery_sag}: "
                "a sagged ceiling of zero or below stops every rotor"
            )
        decimation = round(cfg.physics_hz / cfg.control_hz)
        if decimation < 1:
            raise ValueError(
                f"physics_hz={cfg.physics_hz} must be >= control_hz={cfg.control_hz}"
            )

        self.cfg = cfg
        self.dr = dr  # effective DomainRand: master scale already folded in
        self._delay_min, self._delay_max = d_min, d_max
        self._imu_noise_on = dr.gyro_noise_rps > 0.0 or dr.accel_noise_mps2 > 0.0
        self._imu_bias_on = dr.gyro_bias_rps > 0.0 or dr.accel_bias_mps2 > 0.0
        self.fleet = int(cfg.num_envs)
        self.airframe = AIRFRAMES[cfg.airframe]
        self.decimation = int(decimation)
        self.dt_physics = 1.0 / cfg.physics_hz
        self.dt_control = self.decimation / cfg.physics_hz
        self.act_dim = 4

        self.task: Task = task if task is not None else self._build_task(cfg)
        self.obs_spec = self.task.obs_spec
        self.obs_dim = self.obs_spec.dim
        self.image_shape = self.task.image_shape

        self._fw: FirmwareFleet | None = None
        # boot-image temp file path when cfg.eeprom rendered (logs/provenance), else None
        self.eeprom_image: str | None = None
        # per-base bundle revision when the dump selected one (logs/provenance)
        self.firmware_base: str | None = None
        if cfg.control == "sticks":
            if cfg.physics_hz != 1000.0:
                raise ValueError(
                    f"sticks mode requires physics_hz=1000: the firmware tick is fixed "
                    f"at 1 kHz and each substep pairs one plant step with exactly one "
                    f"tick (DESIGN.md §10); got physics_hz={cfg.physics_hz}"
                )
            # Imported here, not at module top: motors mode never touches the firmware
            # seam (DESIGN.md §10), and this module must import without the wheel.
            from skyflow import firmware as _firmware

            self._flu_to_frd = _firmware.flu_to_frd
            self._baro_pa = _firmware.baro_pa
            if firmware_fleet is not None and cfg.eeprom is not None:
                raise ValueError(
                    "cfg.eeprom and firmware_fleet= are exclusive — an injected "
                    "fleet already booted its own config"
                )
            fw = firmware_fleet if firmware_fleet is not None else self._build_fleet(
                cfg, _firmware
            )
            if fw.act_dim != self.act_dim:
                raise ValueError(f"firmware fleet act_dim {fw.act_dim} != {self.act_dim}")
            perm = tuple(int(i) for i in motor_perm)
            if sorted(perm) != [0, 1, 2, 3]:
                raise ValueError(f"motor_perm must permute (0,1,2,3), got {motor_perm!r}")
            self._fw = fw
            self._motor_perm = jnp.asarray(perm, jnp.int32)

    def _build_fleet(self, cfg: SimConfig, _firmware: Any) -> FirmwareFleet:
        """cfg.firmware → a FirmwareFleet (DESIGN.md §10).

        "cpu"/"gpu" force a backend and fail loudly. "auto" picks the GPU fleet when
        `fleet >= 3` and a CUDA device is visible, and falls back to the CPU SITL
        fleet with a warning when GPU construction fails (wheel too old, no VRAM, …).
        """
        if cfg.firmware not in ("auto", "cpu", "gpu"):
            raise ValueError(
                f'cfg.firmware must be "auto", "cpu" or "gpu", got {cfg.firmware!r}'
            )
        # Render before any fleet exists: the render boots its own throwaway CPU
        # instance, and the CPU library allows one live fleet per process.
        eeprom, lib, fatbin = self._render_eeprom(cfg)
        if cfg.firmware == "cpu":
            return _firmware.CpuFirmwareFleet(self.fleet, eeprom=eeprom, lib=lib)
        if cfg.firmware == "gpu":
            return _firmware.GpuFirmwareFleet(self.fleet, eeprom=eeprom, cubin=fatbin)
        try:
            gpu_visible = bool(jax.devices("gpu"))
        except RuntimeError:
            gpu_visible = False
        if self.fleet >= 3 and gpu_visible:
            try:
                return _firmware.GpuFirmwareFleet(self.fleet, eeprom=eeprom, cubin=fatbin)
            except Exception as e:  # ImportError (wheel < 0.3.3), RuntimeError (create)
                warnings.warn(
                    f'firmware="auto": GPU fleet unavailable ({e}); '
                    "using the CPU SITL fleet",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return _firmware.CpuFirmwareFleet(self.fleet, eeprom=eeprom, lib=lib)

    def _render_eeprom(self, cfg: SimConfig) -> tuple[str | None, str | None, str | None]:
        """cfg.eeprom (CLI `dump all` path) → (boot image, CPU lib, GPU fatbin) paths.

        All three are None when cfg.eeprom is None; lib/fatbin are None when the dump
        belongs to the installed wheel's own base (or carries no revision) — the
        wheel's embedded binaries serve. A dump from ANOTHER base selects the matching
        per-base pair from the cudaflight bundle cache (`_resolve_firmware_base`).

        cudaflight renders through a version-gated strict round-trip on one throwaway
        CPU boot of the SELECTED lib, so a stale or foreign dump fails HERE — an image
        one parameter-group version behind would factory-reset silently and fly stock
        defaults. The image is a derived artifact: rendered fresh per construction,
        never committed. Provenance lands on `self.eeprom_image` / `self.firmware_base`.
        """
        if cfg.eeprom is None:
            return None, None, None
        dump = Path(cfg.eeprom)
        if not dump.is_file():
            raise FileNotFoundError(f"cfg.eeprom: no such CLI dump file: {dump}")
        overrides: Path | None = None
        if cfg.eeprom_overrides is not None:
            overrides = Path(cfg.eeprom_overrides)
            if not overrides.is_file():
                raise FileNotFoundError(
                    f"cfg.eeprom_overrides: no such CLI file: {overrides}"
                )
        import cudaflight  # deferred like the fleet import: only this seam needs it

        bundle = _resolve_firmware_base(dump)
        lib = str(bundle.lib) if bundle is not None else None
        fatbin = str(bundle.fatbin) if bundle is not None else None
        self.firmware_base = bundle.rev if bundle is not None else None

        image = cudaflight.render_eeprom(dump, overrides, lib_path=lib)
        f = tempfile.NamedTemporaryFile(
            prefix="skyflow-eeprom-", suffix=".bin", delete=False
        )
        with f:
            f.write(image)
        self.eeprom_image = f.name
        return f.name, lib, fatbin

    @staticmethod
    def _build_task(cfg: SimConfig) -> Task:
        """cfg.task through the skyflow.tasks registry (imported lazily: the registry
        pulls in every shipped task, and injected-task callers never need it).

        Two quantities the env owns are forwarded to builders that NAME them —
        `spawn_dr_scale` (= cfg.dr.spawn_scale, the §7 spawn-jitter scale) and `control_hz` (the Task
        protocol carries no clock, and a task counting seconds must count them at the
        platform's rate). Explicit `task_kwargs` entries always win, and builders that
        do not name a quantity are built untouched (GateCourseTask names neither).
        """
        from skyflow import tasks as task_registry

        builder = task_registry.get_builder(cfg.task)
        try:
            accepted = inspect.signature(builder).parameters
        except (TypeError, ValueError):  # opaque callable: nothing is forwarded
            accepted = {}
        kwargs = dict(cfg.task_kwargs)
        for name, value in (
            ("spawn_dr_scale", cfg.dr.spawn_scale),
            ("control_hz", cfg.control_hz),
        ):
            if name in accepted and name not in kwargs:
                kwargs[name] = value
        return task_registry.build_task(cfg.task, **kwargs)

    # -- DomainRand draws (all read self.dr — the effective, master-scaled setting) ------

    def _draw_traits(self, key: Array, f: int) -> DRState:
        """Fresh per-episode trait rows (types.DRState) — used at reset and respawn.

        Steady wind: horizontal direction uniform on the circle, magnitude
        U(0, wind_mean_mps). IMU bias: per-axis U(-half_width, +half_width). Zero
        ceilings give exactly-zero rows, so the leaves always exist and stay inert."""
        dr = self.dr
        k_dir, k_mag, k_bias = jax.random.split(key, 3)
        theta = jax.random.uniform(k_dir, (f,), jnp.float32, 0.0, 2.0 * math.pi)
        mag = jax.random.uniform(k_mag, (f,), jnp.float32, 0.0, dr.wind_mean_mps)
        wind_mean = jnp.stack(
            [mag * jnp.cos(theta), mag * jnp.sin(theta), jnp.zeros((f,), jnp.float32)],
            axis=-1,
        )
        half = jnp.asarray(
            [dr.accel_bias_mps2] * 3 + [dr.gyro_bias_rps] * 3, jnp.float32
        )
        imu_bias = half * jax.random.uniform(k_bias, (f, 6), jnp.float32, -1.0, 1.0)
        # Battery sag: one-sided ceiling factor per world (a pack only sags). sag=0
        # gives exactly the airframe ceiling, so the leaf always exists and stays inert.
        k_sag = jax.random.fold_in(k_bias, 1)
        sag = dr.battery_sag * jax.random.uniform(k_sag, (f,), jnp.float32, 0.0, 1.0)
        w_max = self.airframe.rotor_speed_max * (1.0 - sag)
        return DRState(wind_mean=wind_mean, imu_bias=imu_bias, w_max=w_max)

    def _measure(
        self, plant: Array, omega: Array, wind: Array, params: Array,
        dr_state: DRState, key: Array,
    ) -> tuple[Array, Array]:
        """sensors.measure with this env's DomainRand corruption. The static gates keep
        the nominal path key-free and bit-exact (sensors.py charter)."""
        dr = self.dr
        return sensors.measure(
            plant, omega, wind, params,
            key=key if self._imu_noise_on else None,
            accel_noise_std=dr.accel_noise_mps2,
            gyro_noise_std=dr.gyro_noise_rps,
            imu_bias=dr_state.imu_bias if self._imu_bias_on else None,
        )

    def _corrupt_obs(self, obs: Array, key: Array) -> Array:
        """DomainRand.obs_noise on the finalized task observation (uniform half-width);
        applied by the env so tasks keep semantics and the env keeps corruption."""
        if self.dr.obs_noise <= 0.0:
            return obs
        return obs + jax.random.uniform(
            key, obs.shape, obs.dtype, -self.dr.obs_noise, self.dr.obs_noise
        )

    # -- public API --------------------------------------------------------------------

    def reset(self, key: Array) -> tuple[Array, SimState]:
        """
        Fresh fleet → (obs [F,obs_dim] f32, SimState). Per-world DomainRand draws
        (params, traits, delay), task spawn; gust state, delay ring and episode
        accumulators cleared. Pure — same key, same fleet, bit for bit.
        """
        dr = self.dr
        f = self.fleet
        k_params, k_spawn, k_traits, k_delay, k_imu, k_obs, k_carry = jax.random.split(
            key, 7
        )

        params = sample_params(
            k_params, self.airframe, f, dr.body_scale, dr.brackets, dr.factors
        )
        plant, task_state = self.task.spawn(k_spawn, f, params)
        plant = plant.astype(jnp.float32)
        dr_state = self._draw_traits(k_traits, f)
        delay_idx = jax.random.randint(
            k_delay, (f,), self._delay_min, self._delay_max + 1, dtype=jnp.int32
        )
        wind_vel = jnp.zeros((f, 3), jnp.float32)  # gust deviation; total adds the mean
        act_buf = jnp.zeros((f, self._delay_max + 1, 4), jnp.float32)
        last_action = jnp.zeros((f, 4), jnp.float32)

        # First observation: IMU with the rotors held at their spawn speeds (no command
        # has been issued yet, so hold is the only self-consistent input).
        k_obs_task, k_obs_dr = jax.random.split(k_obs)
        imu = self._measure(plant, plant[:, 13:17], dr_state.wind_mean, params, dr_state, k_imu)
        obs, task_state = self.task.observe(
            plant, task_state, imu, last_action, k_obs_task, fresh_spawn=True
        )
        obs = self._corrupt_obs(obs, k_obs_dr)

        if self._fw is not None:
            blob, fwstate = self._fw.fresh_firmware_state()
            task_state = _FirmwareCarry(task=task_state, blob=blob, fwstate=fwstate)

        state = SimState(
            plant=plant,
            params=params,
            key=k_carry,
            wind_vel=wind_vel,
            dr_state=dr_state,
            act_buf=act_buf,
            delay_idx=delay_idx,
            last_action=last_action,
            steps=jnp.zeros(f, jnp.int32),
            airborne=jnp.zeros(f, bool),
            ep_return=jnp.zeros(f, jnp.float32),
            ep_len=jnp.zeros(f, jnp.int32),
            crash_frac=jnp.zeros((), jnp.float32),
            success_frac=jnp.zeros((), jnp.float32),
            trunc_frac=jnp.zeros((), jnp.float32),
            ep_return_ema=jnp.zeros((), jnp.float32),
            ep_len_ema=jnp.zeros((), jnp.float32),
            task_state=task_state,
        )
        return obs, state

    def step(
        self, state: SimState, action: Array
    ) -> tuple[Array, SimState, Array, Array, StepInfo]:
        """
        One control step for the fleet → (obs, state', reward [F] f32, done [F] bool,
        info). The full normative pipeline (module docstring); done worlds come back
        already respawned, with their pre-reset observation in info["final_obs"] and the
        pre-reset flags in info["terminated"] / info["truncated"].
        """
        cfg = self.cfg
        dr = self.dr
        f = self.fleet
        af = self.airframe
        w_min, k_thr = af.rotor_speed_min, af.throttle_k
        # Per-world ceiling, materialized [F,4]: the battery-sag trait. Exact per-rotor
        # shape — the generated throttle map is elementwise and must see no implicit
        # rank growth. sag=0 draws the airframe constant, so numerics do not change.
        w_max = jnp.broadcast_to(state.dr_state.w_max[:, None], (f, 4))
        cos_tilt_min = math.cos(cfg.ground_tilt_limit_rad)

        if self._fw is not None:
            ts_in, blob, fwstate = state.task_state
        else:
            ts_in = state.task_state
            blob = fwstate = None  # never read in motors mode

        # 1. Keys; delay ring (newest first) and the per-world delayed command.
        (
            k_carry, k_wind, k_gate, k_poke_f, k_poke_tau, k_sub, k_imu, k_obs, k_reset,
        ) = jax.random.split(state.key, 9)
        action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
        act_buf = jnp.concatenate([action[:, None, :], state.act_buf[:, :-1, :]], axis=1)
        delayed = act_buf[jnp.arange(f), state.delay_idx]

        # 3. OU gust deviation, exact discretization over dt_control: stationary std is
        # exactly wind_gust_mps per axis for any step size (decay + variance-matched
        # kick). Every consumer sees the total wind: per-episode steady mean + gust.
        alpha = math.exp(-self.dt_control / dr.wind_tau_s)
        kick = dr.wind_gust_mps * math.sqrt(1.0 - alpha * alpha)
        wind_vel = alpha * state.wind_vel + kick * jax.random.normal(
            k_wind, (f, 3), jnp.float32
        )
        wind_total = state.dr_state.wind_mean + wind_vel

        # 4. Pokes: exogenous inputs only — the backend integrates them, velocity state
        # is never written. Uniform-ball direction · configured magnitude ceiling.
        poke = jax.random.bernoulli(k_gate, dr.poke_prob, (f,))
        f_ext = jnp.where(poke[:, None], dr.poke_force_n * _uniform_ball(k_poke_f, f), 0.0)
        tau_ext = jnp.where(
            poke[:, None], dr.poke_torque_nm * _uniform_ball(k_poke_tau, f), 0.0
        )

        # 2 + 5. Command map, then `decimation` substeps with every input zero-order-held
        # and the §8 contact clamp after each; the pre-clamp ground-crash predicate is
        # latched per substep (see _ground_impact).
        if self._fw is None:
            u = 0.5 * (delayed + 1.0)
            omega_cmd = dynamics.throttle_to_omega(u, w_min, w_max, k_thr).astype(
                jnp.float32
            )

            def substep_motors(carry, _):
                plant, impact = carry
                raw = dynamics.substep(
                    plant, omega_cmd, wind_total, f_ext, tau_ext, state.params,
                    self.dt_physics, w_min, w_max,
                )
                impact = impact | _ground_impact(raw, cos_tilt_min)
                return (_ground_contact(raw), impact), None

            (plant, impact), _ = jax.lax.scan(
                substep_motors, (state.plant, jnp.zeros(f, bool)), None, length=self.decimation
            )
            omega_last = omega_cmd
        else:
            # Sticks (§10, normative substep order): synth FRD sensors from the generated
            # IMU + isothermal baro, corrupted per DomainRand (bias + per-sample noise —
            # the firmware filters what the real one filters) → firmware tick → QUADX
            # motors reordered by motor_perm → duties feed the throttle map as u, ZOH
            # for this 1 ms substep.
            fw = self._fw
            baro_on = dr.baro_noise_pa > 0.0
            sub_keys = jax.random.split(k_sub, self.decimation)

            def substep_sticks(carry, k_t):
                plant, blob, fwstate, omega_prev, impact = carry
                k_imu_t, k_baro_t = jax.random.split(k_t)
                accel, gyro = self._measure(
                    plant, omega_prev, wind_total, state.params, state.dr_state, k_imu_t
                )
                baro = self._baro_pa(plant[:, 2:3]).astype(jnp.float32)
                if baro_on:
                    baro = baro + dr.baro_noise_pa * jax.random.normal(
                        k_baro_t, baro.shape, jnp.float32
                    )
                rows = jnp.concatenate(
                    [self._flu_to_frd(gyro), self._flu_to_frd(accel), baro], axis=-1
                )
                blob, fwstate, motors, _armed = fw.fw_step(blob, fwstate, delayed, rows)
                u = motors[:, self._motor_perm]
                omega_cmd = dynamics.throttle_to_omega(u, w_min, w_max, k_thr).astype(
                    jnp.float32
                )
                raw = dynamics.substep(
                    plant, omega_cmd, wind_total, f_ext, tau_ext, state.params,
                    self.dt_physics, w_min, w_max,
                )
                impact = impact | _ground_impact(raw, cos_tilt_min)
                return (_ground_contact(raw), blob, fwstate, omega_cmd, impact), None

            init = (state.plant, blob, fwstate, state.plant[:, 13:17], jnp.zeros(f, bool))
            (plant, blob, fwstate, omega_last, impact), _ = jax.lax.scan(
                substep_sticks, init, sub_keys
            )

        # 6. Airborne latch; env crash set; task verdict on the transition.
        z = plant[:, 2]
        airborne = state.airborne | (z > _GROUND_BAND_M)
        flyaway = (
            (jnp.abs(plant[:, 0]) > cfg.bounds_xy_m)
            | (jnp.abs(plant[:, 1]) > cfg.bounds_xy_m)
            | (z > cfg.bounds_z_m)
            | (jnp.linalg.norm(plant[:, 3:6], axis=-1) > cfg.max_speed_mps)
            | (jnp.linalg.norm(plant[:, 10:13], axis=-1) > cfg.max_rate_rps)
        )
        crash = flyaway | (impact & airborne)
        ev = self.task.evaluate(state.plant, plant, ts_in)
        reward = ev.reward.astype(jnp.float32)

        # 7. Done split. `stuck`: spawned-on-ground worlds whose policy never got them
        # airborne within stuck_steps have nothing left to learn from the episode.
        terminated = crash | ev.crash
        if self.task.success_terminates:
            terminated = terminated | ev.success
        steps = state.steps + 1
        stuck = ~airborne & (steps >= cfg.stuck_steps)
        truncated = (steps >= cfg.max_episode_steps) | stuck
        done = terminated | truncated

        # 8. Observe the post-transition state (pre-reset: this is final_obs on done rows).
        k_obs_task, k_obs_dr = jax.random.split(k_obs)
        imu = self._measure(plant, omega_last, wind_total, state.params, state.dr_state, k_imu)
        obs, ts_out = self.task.observe(
            plant, ev.task_state, imu, action, k_obs_task, fresh_spawn=False
        )
        obs = self._corrupt_obs(obs, k_obs_dr)

        # 10 (accumulated before the blend so done rows report full-episode stats in
        # info and feed the EMAs below).
        ep_return = state.ep_return + reward
        ep_len = state.ep_len + 1

        # 10. Episode bookkeeping EMAs (§7 step 10): fleet-global 0-d leaves, updated
        # with the done-row means and decayed per completed episode; when nothing
        # finished alpha = 1 and the EMAs pass through unchanged. Kept out of the §7
        # step 9 tree_where blend below — they are cross-episode, not per-world.
        n_done = jnp.sum(done.astype(jnp.float32))
        alpha = jnp.asarray(_METRICS_EMA_DECAY, jnp.float32) ** n_done
        denom = jnp.maximum(n_done, 1.0)

        def ema(old: Array, rows: Array) -> Array:
            done_mean = jnp.sum(jnp.where(done, rows.astype(jnp.float32), 0.0)) / denom
            return (alpha * old + (1.0 - alpha) * done_mean).astype(jnp.float32)

        crash_frac = ema(state.crash_frac, crash | ev.crash)
        success_frac = ema(state.success_frac, ev.success)
        trunc_frac = ema(state.trunc_frac, truncated & ~terminated)
        ep_return_ema = ema(state.ep_return_ema, ep_return)
        ep_len_ema = ema(state.ep_len_ema, ep_len)

        # 9. Auto-reset: fresh spawn + fresh DomainRand draws (params, traits, delay) +
        # cleared buffers/gust for done worlds, blended leaf-wise; live worlds pass
        # through untouched (bit-identical).
        k_rp, k_rs, k_rt, k_rd, k_ri, k_ro = jax.random.split(k_reset, 6)
        params_new = sample_params(k_rp, af, f, dr.body_scale, dr.brackets, dr.factors)
        plant_new, ts_fresh = self.task.spawn(k_rs, f, params_new)
        plant_new = plant_new.astype(jnp.float32)
        dr_new = self._draw_traits(k_rt, f)
        la_new = jnp.zeros((f, 4), jnp.float32)
        wind_new = jnp.zeros((f, 3), jnp.float32)
        k_ro_task, k_ro_dr = jax.random.split(k_ro)
        imu_new = self._measure(
            plant_new, plant_new[:, 13:17], dr_new.wind_mean, params_new, dr_new, k_ri
        )
        obs_new, ts_fresh = self.task.observe(
            plant_new, ts_fresh, imu_new, la_new, k_ro_task, fresh_spawn=True
        )
        obs_new = self._corrupt_obs(obs_new, k_ro_dr)
        fresh = dict(
            plant=plant_new,
            params=params_new,
            wind_vel=wind_new,
            dr_state=dr_new,
            act_buf=jnp.zeros((f, self._delay_max + 1, 4), jnp.float32),
            delay_idx=jax.random.randint(
                k_rd, (f,), self._delay_min, self._delay_max + 1, dtype=jnp.int32
            ),
            last_action=la_new,
            steps=jnp.zeros(f, jnp.int32),
            airborne=jnp.zeros(f, bool),
            ep_return=jnp.zeros(f, jnp.float32),
            ep_len=jnp.zeros(f, jnp.int32),
            task_state=ts_fresh,
        )
        current = dict(
            plant=plant,
            params=state.params,
            wind_vel=wind_vel,
            dr_state=state.dr_state,
            act_buf=act_buf,
            delay_idx=state.delay_idx,
            last_action=action,
            steps=steps,
            airborne=airborne,
            ep_return=ep_return,
            ep_len=ep_len,
            task_state=ts_out,
        )
        merged = tree_where(done, fresh, current)
        obs_out = jnp.where(done[:, None], obs_new, obs)

        ts_state = merged.pop("task_state")
        if self._fw is not None:
            # The fleet masks internally (types.FirmwareFleet.reset), so the pair is
            # taken whole rather than tree_where-blended (CPU placeholders are [0]).
            blob, fwstate = self._fw.reset(blob, fwstate, done.astype(jnp.uint8))
            ts_state = _FirmwareCarry(task=ts_state, blob=blob, fwstate=fwstate)

        state_out = SimState(
            key=k_carry,
            task_state=ts_state,
            crash_frac=crash_frac,
            success_frac=success_frac,
            trunc_frac=trunc_frac,
            ep_return_ema=ep_return_ema,
            ep_len_ema=ep_len_ema,
            **merged,
        )
        # cast: the task's ev.info keys ride along beyond the typed StepInfo contract.
        info = cast(
            "StepInfo",
            {
                **ev.info,
                "terminated": terminated,
                "truncated": truncated,
                "final_obs": obs,
                "poke_active": poke,
                "ep_return": ep_return,
                "ep_len": ep_len,
            },
        )
        return obs_out, state_out, reward, done, info

    def task_state(self, state: SimState) -> Any:
        """
        The task's OWN pytree for this state. In sticks mode `SimState.task_state` is
        the firmware carry (§10) with the task's pytree inside it; every consumer that
        wants task fields (metrics, viewers, recorders) must read through this accessor
        instead of unwrapping the carry itself.
        """
        return state.task_state.task if self._fw is not None else state.task_state

    def metrics(self, state: SimState) -> dict[str, Array]:
        """
        Scalar means for logging (DESIGN.md §7): outcome fractions and episode stats as
        EMAs over COMPLETED episodes (step 10 bookkeeping — crash_frac / success_frac /
        trunc_frac, ep_return_ema / ep_len_ema; 0 until the first episode finishes),
        plus instantaneous live-fleet means (ep_return_mean / ep_len_mean average the
        IN-PROGRESS accumulators, airborne_frac and wind_speed_mean the current state)
        and the task's own diagnostics. All values are 0-d f32 — cheap to device-get at
        any cadence.
        """
        ts = self.task_state(state)
        out = {
            "crash_frac": state.crash_frac,
            "success_frac": state.success_frac,
            "trunc_frac": state.trunc_frac,
            "ep_return_ema": state.ep_return_ema,
            "ep_len_ema": state.ep_len_ema,
            "ep_return_mean": jnp.mean(state.ep_return),
            "ep_len_mean": jnp.mean(state.ep_len.astype(jnp.float32)),
            "airborne_frac": jnp.mean(state.airborne.astype(jnp.float32)),
            "wind_speed_mean": jnp.mean(
                jnp.linalg.norm(state.dr_state.wind_mean + state.wind_vel, axis=-1)
            ),
        }
        for name, value in self.task.metrics(ts).items():
            out[name] = jnp.mean(value.astype(jnp.float32))
        return out
