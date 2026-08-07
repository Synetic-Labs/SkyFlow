"""SkyFlow — a differentiable, fleet-batched quadrotor simulator in pure JAX.

The whole rollout (physics, sensors, camera, reward) is one ``lax.scan`` on the
accelerator: thousands of drones step together, nothing leaves the device, and the
step is differentiable end-to-end.

Layout:

* :mod:`plant` / :mod:`params` — the analytic quadrotor model and its per-airframe
  coefficients. Pure functions over arrays; no env, no state, trivially testable.
* :mod:`env` — the *platform*: the plant, domain randomization, disturbances,
  transport latency, the fused rollout, the generic crash set and the in-jit
  auto-reset.
* :mod:`tasks` — the *objective*: spawn, observation, reward, task terminals.
  ``hover`` ships; register your own with :func:`skyflow.tasks.register_task`.
* :mod:`render` — analytic gate-mask rendering and mask-noise randomization.

``env`` is not re-exported here. Importing it constructs nothing, but keeping it an
explicit ``from skyflow.env import SkyFlowEnv`` keeps :mod:`plant`/:mod:`params`
importable in isolation for unit tests and offline analysis.

    from skyflow.env import SkyFlowEnv
    env = SkyFlowEnv(num_envs=4096, task="hover", control="motors")

MIT licensed. See README.md for credits and scope.
"""

__version__ = "0.1.0"
