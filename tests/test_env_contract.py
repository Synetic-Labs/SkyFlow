"""
DESIGN.md §11 — the SkyFlowEnv platform contract: shapes/dtypes/finiteness,
determinism, done semantics, final_obs, auto-reset isolation, delay buffer, ZOH. The
jitted 50-step rollout + no-retrace smoke (the env half of the jit checks) lives here
too, since env.py is the unit under test.

All tests run the hover task injected directly (`SkyFlowEnv(cfg, task=...)`) so the
suite does not depend on the tasks/ registry, which test_registry.py covers. Fleets are
small and keys fixed; body DR defaults to off here so thresholds are exact.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow import dynamics
from skyflow.env import _METRICS_EMA_DECAY, DomainRand, SimConfig, SkyFlowEnv
from skyflow.tasks.hover import HoverTask

FLEET = 5


def dr0(**dr_kwargs) -> DomainRand:
    """DomainRand with body jitter off (thresholds stay exact) plus the given knobs."""
    return DomainRand(body_scale=0.0, **dr_kwargs)


def make_env(
    fleet: int = FLEET, task_kwargs: dict | None = None,
    dr: DomainRand | None = None, **cfg_kwargs,
) -> SkyFlowEnv:
    cfg = SimConfig(num_envs=fleet, dr=dr if dr is not None else dr0(), **cfg_kwargs)
    return SkyFlowEnv(cfg, task=HoverTask(**(task_kwargs or {})))


def _actions(key, n_steps: int, fleet: int) -> jax.Array:
    """[T,F,4] deterministic in-range action sequence."""
    return jnp.tanh(jax.random.normal(key, (n_steps, fleet, 4), jnp.float32))


# -- shapes / dtypes / finiteness ---------------------------------------------------------


def test_env_attributes():
    env = make_env()
    assert env.fleet == FLEET
    assert env.act_dim == 4
    assert env.obs_dim == 19 and env.obs_spec.dim == 19
    assert env.image_shape is None
    assert env.decimation == 10
    assert env.dt_control == pytest.approx(0.01)
    assert env.dt_physics == pytest.approx(1e-3)


def test_obs_terms_declare_units_and_stay_two_field_compatible():
    """Every shipped task declares units on every ObsTerm; the field defaults to ""
    so 2-field construction (downstream tasks written before it) keeps working."""
    from skyflow.types import ObsTerm

    assert ObsTerm("legacy", 3) == ObsTerm("legacy", 3, "")  # back-compat default
    env = make_env()
    assert all(t.units for t in env.obs_spec), [t.name for t in env.obs_spec if not t.units]


def test_reset_shapes_dtypes_finiteness(key):
    env = make_env()
    obs, state = env.reset(key)
    assert obs.shape == (FLEET, 19) and obs.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert state.plant.shape == (FLEET, dynamics.STATE_DIM)
    for leaf, dtype in [
        (state.plant, jnp.float32), (state.params, jnp.float32),
        (state.wind_vel, jnp.float32), (state.act_buf, jnp.float32),
        (state.last_action, jnp.float32), (state.ep_return, jnp.float32),
        (state.delay_idx, jnp.int32), (state.steps, jnp.int32), (state.ep_len, jnp.int32),
    ]:
        assert leaf.dtype == dtype and leaf.shape[0] == FLEET
    assert state.airborne.dtype == jnp.bool_ and not bool(state.airborne.any())
    assert state.act_buf.shape == (FLEET, 1, 4)  # delay_steps=(0,0) ⇒ ring depth 1
    assert bool(jnp.all(state.steps == 0)) and bool(jnp.all(state.ep_len == 0))


def test_step_shapes_dtypes_and_info(key):
    env = make_env()
    _, state = env.reset(key)
    obs, state, reward, done, info = env.step(state, _actions(key, 1, FLEET)[0])
    assert obs.shape == (FLEET, 19) and obs.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert reward.shape == (FLEET,) and reward.dtype == jnp.float32
    assert done.shape == (FLEET,) and done.dtype == jnp.bool_
    assert info["terminated"].shape == (FLEET,) and info["terminated"].dtype == jnp.bool_
    assert info["truncated"].shape == (FLEET,) and info["truncated"].dtype == jnp.bool_
    assert info["final_obs"].shape == (FLEET, 19) and info["final_obs"].dtype == jnp.float32
    assert "hover/dist" in info  # task evaluate info merges into StepInfo
    assert bool(jnp.all(state.steps == 1)) and bool(jnp.all(state.ep_len == 1))


def test_metrics_are_scalars_with_the_exact_documented_keys(key):
    """Exact key set — DESIGN §7 env keys (outcome-fraction/ep-stat EMAs + live-fleet
    means) plus the hover task's diagnostics; a subset check could not catch a dropped
    contract key."""
    env = make_env()
    _, state = env.reset(key)
    m = env.metrics(state)
    assert set(m) == {
        "crash_frac", "success_frac", "trunc_frac", "ep_return_ema", "ep_len_ema",
        "ep_return_mean", "ep_len_mean", "airborne_frac", "wind_speed_mean",
        "hover/dist", "hover/goal_hold",
    }
    for v in m.values():
        assert v.shape == ()


# -- determinism ---------------------------------------------------------------------------


def test_same_key_same_rollout_bit_identical(key):
    env = make_env(dr=dr0(wind_gust_mps=1.0, poke_prob=0.2, poke_force_n=0.05, delay_steps=(0, 1)))
    acts = _actions(jax.random.PRNGKey(9), 15, FLEET)

    def rollout():
        obs, state = env.reset(key)
        trace = [obs]
        for a in acts:
            obs, state, reward, done, _ = env.step(state, a)
            trace += [obs, reward, done]
        return trace, state

    t1, s1 = rollout()
    t2, s2 = rollout()
    for a, b in zip(t1, t2, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(s1.plant), np.asarray(s2.plant))
    np.testing.assert_array_equal(np.asarray(s1.wind_vel), np.asarray(s2.wind_vel))


def test_out_of_range_actions_are_clipped(key):
    env = make_env()
    _, state = env.reset(key)
    _, s_hot, r_hot, _, _ = env.step(state, 5.0 * jnp.ones((FLEET, 4), jnp.float32))
    _, s_one, r_one, _, _ = env.step(state, jnp.ones((FLEET, 4), jnp.float32))
    np.testing.assert_array_equal(np.asarray(s_hot.plant), np.asarray(s_one.plant))
    np.testing.assert_array_equal(np.asarray(r_hot), np.asarray(r_one))


# -- ZOH + command map: one env step ≡ decimation manual substeps --------------------------


def test_zoh_step_equals_manual_substep_composition(key):
    """Constant action ⇒ one Ω_c, zero-order-held across all substeps: the env step must
    reproduce the manual composition of `decimation` backend substeps bit-for-bit up to
    scan-vs-loop fusion tolerance. Airborne placement keeps the contact clamp inactive."""
    env = make_env(fleet=3)
    af = env.airframe
    _, state = env.reset(key)
    state = state.replace(plant=state.plant.at[:, 2].set(1.5))
    action = _actions(jax.random.PRNGKey(3), 1, 3)[0]

    _, s_env, _, _, _ = env.step(state, action)

    u = 0.5 * (action + 1.0)
    omega_cmd = dynamics.throttle_to_omega(
        u, af.rotor_speed_min, af.rotor_speed_max, af.throttle_k
    ).astype(jnp.float32)
    zeros = jnp.zeros((3, 3), jnp.float32)
    plant = state.plant
    for _ in range(env.decimation):
        plant = dynamics.substep(
            plant, omega_cmd, zeros, zeros, zeros, state.params,
            env.dt_physics, af.rotor_speed_min, af.rotor_speed_max,
        )
    np.testing.assert_allclose(
        np.asarray(s_env.plant), np.asarray(plant), rtol=2e-6, atol=2e-6
    )


# -- delay buffer ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 2])
def test_action_takes_effect_exactly_k_steps_late(key, k):
    """Idle-filled ring, then full throttle from call 0: with Ω_c = 0 the rotor speeds
    stay exactly zero (first-order motor, zero state, zero command), so the first step
    whose state shows spinning rotors is exactly the k-delayed one."""
    env = make_env(fleet=3, dr=dr0(delay_steps=(k, k)))
    _, state = env.reset(key)
    assert bool(jnp.all(state.delay_idx == k))
    state = state.replace(act_buf=jnp.full_like(state.act_buf, -1.0))  # idle-stick fill

    full = jnp.ones((3, 4), jnp.float32)
    speeds = []
    for _ in range(k + 2):
        _, state, _, _, _ = env.step(state, full)
        speeds.append(float(jnp.max(state.plant[:, 13:17])))
    for t in range(k):
        assert speeds[t] == 0.0, f"rotors moved {k - t} steps early"
    assert speeds[k] > 1.0, "the delayed command never arrived"


# -- done semantics --------------------------------------------------------------------------


def test_truncation_at_max_episode_steps(key):
    env = make_env(max_episode_steps=4, stuck_steps=10**6)
    _, state = env.reset(key)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    for t in range(1, 4):
        _, state, _, done, info = env.step(state, a)
        assert not bool(done.any()), f"early done at step {t}"
    _, state, _, done, info = env.step(state, a)
    assert bool(done.all())
    assert bool(info["truncated"].all()) and not bool(info["terminated"].any())
    assert bool(jnp.all(info["ep_len"] == 4))  # pre-reset accumulator on done rows
    # auto-reset: bookkeeping cleared in the returned state
    assert bool(jnp.all(state.steps == 0)) and bool(jnp.all(state.ep_len == 0))
    assert bool(jnp.all(state.ep_return == 0.0)) and not bool(state.airborne.any())


def test_stuck_truncation_for_never_airborne_worlds(key):
    env = make_env(stuck_steps=3, max_episode_steps=1000)
    _, state = env.reset(key)
    idle = -jnp.ones((FLEET, 4), jnp.float32)
    for _ in range(2):
        _, state, _, done, _ = env.step(state, idle)
        assert not bool(done.any())
    _, state, _, done, info = env.step(state, idle)
    assert bool(done.all()) and bool(info["truncated"].all())
    assert not bool(info["terminated"].any())


def test_termination_on_flyaway(key):
    env = make_env(task_kwargs={"safe_xy_m": 1000.0, "safe_z_m": 1000.0})
    _, state = env.reset(key)
    state = state.replace(plant=state.plant.at[:, 0].set(100.0))  # beyond bounds_xy=20
    _, state, _, done, info = env.step(state, jnp.zeros((FLEET, 4), jnp.float32))
    assert bool(done.all())
    assert bool(info["terminated"].all()) and not bool(info["truncated"].any())
    # respawned inside the pad box, episode bookkeeping cleared
    assert bool(jnp.all(jnp.abs(state.plant[:, 0]) < 2.0))
    assert bool(jnp.all(state.steps == 0))
    # §7 step 10: a fleet-wide crash registers in the crash EMA, not the truncation one
    m = env.metrics(state)
    alpha = _METRICS_EMA_DECAY**FLEET
    assert float(m["crash_frac"]) == pytest.approx(1.0 - alpha, rel=1e-5)
    assert float(m["trunc_frac"]) == 0.0


# -- §7 step 10 episode bookkeeping EMAs -------------------------------------------------------


def test_outcome_ema_metrics_update_on_completed_episodes(key):
    """DESIGN §7 step 10 / §4 EMA leaves: zero after reset, untouched while no episode
    finishes, then blended with the done-row means at decay**n_done on the step the
    fleet truncates. Regression for metrics() lacking outcome fractions entirely."""
    env = make_env(max_episode_steps=4, stuck_steps=10**6)
    _, state = env.reset(key)
    ema_keys = ("crash_frac", "success_frac", "trunc_frac", "ep_return_ema", "ep_len_ema")
    m0 = env.metrics(state)
    for k in ema_keys:
        assert m0[k].shape == () and m0[k].dtype == jnp.float32
        assert float(m0[k]) == 0.0

    a = jnp.zeros((FLEET, 4), jnp.float32)
    returns = np.zeros(FLEET, np.float64)
    for t in range(1, 4):
        _, state, reward, done, _ = env.step(state, a)
        returns += np.asarray(reward, np.float64)
        assert not bool(done.any())
        m = env.metrics(state)
        for k in ema_keys:
            assert float(m[k]) == 0.0, f"{k} moved before any episode completed (t={t})"

    _, state, reward, done, info = env.step(state, a)
    returns += np.asarray(reward, np.float64)
    assert bool(done.all()) and bool(info["truncated"].all())

    m = env.metrics(state)
    alpha = _METRICS_EMA_DECAY**FLEET  # FLEET episodes completed on this step
    assert float(m["trunc_frac"]) == pytest.approx(1.0 - alpha, rel=1e-5)
    assert float(m["crash_frac"]) == 0.0
    assert float(m["ep_len_ema"]) == pytest.approx((1.0 - alpha) * 4.0, rel=1e-5)
    assert float(m["ep_return_ema"]) == pytest.approx(
        (1.0 - alpha) * returns.mean(), rel=1e-4
    )
    # live-fleet means reset with the respawn — they track in-progress accumulators
    assert float(m["ep_len_mean"]) == 0.0


# -- final_obs -------------------------------------------------------------------------------


def test_final_obs_is_the_pre_reset_observation(key):
    """Twin envs differing only in max_episode_steps consume identical RNG streams, so
    the long env's observation at the truncating step IS the short env's final_obs."""
    task = HoverTask()
    env_a = SkyFlowEnv(SimConfig(num_envs=FLEET, dr=dr0(), max_episode_steps=4), task=task)
    env_b = SkyFlowEnv(SimConfig(num_envs=FLEET, dr=dr0(), max_episode_steps=1000), task=task)
    acts = _actions(jax.random.PRNGKey(11), 4, FLEET)

    obs_a, state_a = env_a.reset(key)
    obs_b, state_b = env_b.reset(key)
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
    for a in acts[:-1]:
        obs_a, state_a, _, done_a, info_a = env_a.step(state_a, a)
        obs_b, state_b, _, done_b, _ = env_b.step(state_b, a)
        assert not bool(done_a.any())
        # no reset ⇒ the returned obs IS the pre-reset obs
        np.testing.assert_array_equal(
            np.asarray(info_a["final_obs"]), np.asarray(obs_a)
        )
    obs_a, state_a, _, done_a, info_a = env_a.step(state_a, acts[-1])
    obs_b, state_b, _, done_b, _ = env_b.step(state_b, acts[-1])
    assert bool(done_a.all()) and not bool(done_b.any())
    np.testing.assert_array_equal(np.asarray(info_a["final_obs"]), np.asarray(obs_b))
    # while the returned obs is the fresh spawn's, not the dead state's
    assert not np.array_equal(np.asarray(obs_a), np.asarray(info_a["final_obs"]))


# -- auto-reset isolation ---------------------------------------------------------------------


def test_auto_reset_leaves_live_worlds_bit_identical(key):
    """World 0 forced out of bounds; every other world's next state must be bit-identical
    to the run where world 0 stayed healthy (auto-reset is fully world-local)."""
    env = make_env(dr=dr0(wind_gust_mps=1.0, poke_prob=0.3, poke_force_n=0.02))
    _, state = env.reset(key)
    a = jnp.zeros((FLEET, 4), jnp.float32)
    for _ in range(2):
        _, state, _, _, _ = env.step(state, a)

    state_bad = state.replace(plant=state.plant.at[0, 0].set(100.0))
    obs_x, s_x, r_x, done_x, _ = env.step(state_bad, a)
    obs_y, s_y, r_y, done_y, _ = env.step(state, a)

    assert bool(done_x[0]) and not bool(done_x[1:].any()) and not bool(done_y.any())
    np.testing.assert_array_equal(np.asarray(obs_x[1:]), np.asarray(obs_y[1:]))
    np.testing.assert_array_equal(np.asarray(r_x[1:]), np.asarray(r_y[1:]))
    for lx, ly in zip(jax.tree.leaves(s_x), jax.tree.leaves(s_y), strict=True):
        if lx.ndim == 0 or lx.shape[0] != FLEET:
            continue  # the env-owned PRNG key is fleet-global by design
        np.testing.assert_array_equal(np.asarray(lx[1:]), np.asarray(ly[1:]))


# -- jitted rollout smoke + retrace check ------------------------------------------------------


def test_jitted_50_step_rollout_no_nan_no_retrace(key):
    env = make_env(
        fleet=8, dr=DomainRand(
            wind_gust_mps=0.5, poke_prob=0.05, poke_force_n=0.02, delay_steps=(0, 1)
        ),
    )
    traces = []

    def stepper(state, action):
        traces.append(1)  # python body runs only while jax traces
        return env.step(state, action)

    jstep = jax.jit(stepper)
    acts = _actions(jax.random.PRNGKey(21), 50, 8)
    obs, state = env.reset(key)
    for a in acts:
        obs, state, reward, _done, _info = jstep(state, a)
        assert bool(jnp.all(jnp.isfinite(obs))) and bool(jnp.all(jnp.isfinite(reward)))
    assert len(traces) == 1, f"env.step retraced: {len(traces)} traces for 50 calls"
    assert bool(jnp.all(jnp.isfinite(state.plant)))


# -- construction guards ------------------------------------------------------------------------


def test_differentiable_raises_planned():
    with pytest.raises(NotImplementedError, match="planned"):
        SkyFlowEnv(SimConfig(num_envs=2, differentiable=True), task=HoverTask())


def test_bad_config_raises():
    with pytest.raises(ValueError, match="control"):
        SkyFlowEnv(SimConfig(num_envs=2, control="wands"), task=HoverTask())
    with pytest.raises(ValueError, match="airframe"):
        SkyFlowEnv(SimConfig(num_envs=2, airframe="voliro"), task=HoverTask())
    with pytest.raises(ValueError, match="delay"):
        SkyFlowEnv(
            SimConfig(num_envs=2, dr=DomainRand(delay_steps=(3, 1))), task=HoverTask()
        )
    with pytest.raises(ValueError, match="physics_hz"):
        SkyFlowEnv(
            SimConfig(num_envs=2, physics_hz=50.0, control_hz=100.0), task=HoverTask()
        )


def test_sticks_without_cudaflight_raises_import_error():
    if importlib.util.find_spec("cudaflight") is not None:
        pytest.skip("cudaflight installed; the guidance path cannot trigger")
    with pytest.raises(ImportError, match="cudaflight"):
        SkyFlowEnv(SimConfig(num_envs=2, control="sticks"), task=HoverTask())


# -- sticks branch against the FirmwareFleet protocol (no cudaflight involved) ------------------


class _FakeFleet:
    """Pure-JAX types.FirmwareFleet stand-in: motors follow the throttle stick, fwstate
    counts 1 kHz ticks, reset zeroes the masked worlds' counters."""

    act_dim = 4

    def __init__(self, fleet: int):
        self.fleet = fleet

    def fresh_firmware_state(self):
        return jnp.zeros((0,), jnp.uint8), jnp.zeros((self.fleet,), jnp.int32)

    def fw_step(self, blob, fwstate, sticks, sensors):
        assert sticks.shape == (self.fleet, 4), sticks.shape
        assert sensors.shape == (self.fleet, 7), sensors.shape  # gyro(3)+sf(3)+baro(1) FRD
        throttle = 0.5 * (sticks[:, 2:3] + 1.0)
        motors = jnp.tile(throttle, (1, 4)).astype(jnp.float32)
        return blob, fwstate + 1, motors, jnp.ones((self.fleet,), jnp.uint8)

    def reset(self, blob, fwstate, mask):
        return blob, jnp.where(mask.astype(bool), 0, fwstate)

    def close(self):
        pass


@pytest.mark.parametrize("hz", [500.0, 2000.0, 999.0])
def test_sticks_requires_1khz_physics(hz):
    """The firmware tick is hard-fixed at 1 ms (types.FirmwareFleet, DESIGN §10): any
    non-1000 physics_hz in sticks mode would silently skew Betaflight's virtual clock
    against the plant, so construction must refuse it — even with an injected fleet."""
    fleet = 3
    with pytest.raises(ValueError, match="physics_hz=1000"):
        SkyFlowEnv(
            SimConfig(num_envs=fleet, control="sticks", physics_hz=hz, control_hz=100.0),
            task=HoverTask(),
            firmware_fleet=_FakeFleet(fleet),
        )


def test_sticks_pipeline_with_injected_fleet(key):
    fleet = 3
    env = SkyFlowEnv(
        SimConfig(num_envs=fleet, control="sticks", dr=dr0()),
        task=HoverTask(),
        firmware_fleet=_FakeFleet(fleet),
    )
    obs, state = env.reset(key)
    assert obs.shape == (fleet, 19)
    assert state.task_carry.task.goal.shape == (fleet, 3)  # firmware carry wraps the task
    assert env.task_state(state).goal.shape == (fleet, 3)  # the consumer-facing read

    # Throttle stick high → fake fleet outputs duty 1.0 → rotors spin up immediately.
    sticks = jnp.tile(jnp.asarray([0.0, 0.0, 1.0, 0.0], jnp.float32), (fleet, 1))
    obs, state, reward, _done, _info = env.step(state, sticks)
    assert obs.shape == (fleet, 19) and reward.shape == (fleet,)
    assert bool(jnp.all(state.plant[:, 13:17] > 0.0))
    # the firmware ticked once per 1 kHz substep
    np.testing.assert_array_equal(np.asarray(state.task_carry.fwstate), env.decimation)
    assert env.metrics(state)["hover/dist"].shape == ()
