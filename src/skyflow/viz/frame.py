"""
ViewFrame — the host-side snapshot the panes draw from (DESIGN.md §13).

The one device→host crossing: `snapshot` gathers the watched worlds' rows on the device
and pulls them in a single `jax.device_get` — kilobytes per displayed frame, regardless of
fleet size. Everything downstream (scene pane, FPV, HUD, FlightLog, binds) speaks this
numpy dataclass, so live viewing, replay and export share one vocabulary.
"""

import dataclasses
from typing import Any

import numpy as np

__all__ = ["ViewFrame", "quat_to_rot", "snapshot"]


@dataclasses.dataclass
class ViewFrame:
    """Watched-world rows as numpy; `[W, ...]` leading unless noted."""

    plant: np.ndarray  # [W,17] spec layout: x(3) v(3) q_wxyz(4) ω(3) Ω(4)
    step: int = 0  # focused world's control-step count
    t: float = 0.0  # step * dt, seconds
    focus: int = 0  # index into the watch list, not the fleet
    action: np.ndarray | None = None  # [W,4] in [-1,1]
    # named scalar traces, each [W] — reward is a channel like any other (§13); the
    # HUD plots one graph per name and FlightLog stores them under their names
    channels: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    done: np.ndarray | None = None  # [W] bool
    obs: np.ndarray | None = None  # [W,obs_dim] — the policy FPV pane slices this
    task_state: Any = None  # watch-row task pytree (binds walk it)
    info: dict[str, np.ndarray] | None = None  # [W] rows of the step info
    positions: np.ndarray | None = None  # [F,3] whole-fleet scatter, only when asked

    @property
    def reward(self) -> np.ndarray | None:
        """Convenience read of the "reward" channel (None when the caller omitted it)."""
        return self.channels.get("reward")

    @property
    def pos(self) -> np.ndarray:
        return self.plant[:, 0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.plant[:, 3:6]

    @property
    def quat(self) -> np.ndarray:
        return self.plant[:, 6:10]

    @property
    def body_rates(self) -> np.ndarray:
        return self.plant[:, 10:13]

    @property
    def rotor_speeds(self) -> np.ndarray:
        return self.plant[:, 13:17]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[3,3] body→world rotation from one wxyz Hamilton quaternion (numpy, host-side)."""
    w, x, y, z = (float(v) for v in np.asarray(q, np.float64).reshape(4))
    n = max(np.sqrt(w * w + x * x + y * y + z * z), 1e-9)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def snapshot(
    state: Any,
    watch: np.ndarray | tuple[int, ...],
    *,
    dt: float,
    obs: Any = None,
    action: Any = None,
    reward: Any = None,
    channels: dict[str, Any] | None = None,
    done: Any = None,
    info: dict[str, Any] | None = None,
    task_state: Any = None,
    focus: int = 0,
    fleet_positions: bool = False,
    fleet_stride: int = 1,
) -> ViewFrame:
    """
    One ViewFrame from a SimState (+ optional step outputs), single host transfer.

    `watch` indexes the fleet axis; `focus` indexes the watch list. `channels` are named
    [F] scalars to trace; `reward` is shorthand for `channels={"reward": ...}`.
    `task_state` overrides `state.task_state` — pass `env.task_state(state)` so binds see
    the task's own pytree in sticks mode too. On a sticks-mode SimState the fallback
    read RAISES (SimState.task_state refuses to hand out the firmware carry), so a
    snapshot without the accessor fails loudly instead of hiding every bound
    primitive. Task-state leaves keep their non-fleet shape when they don't lead
    with [F]. `fleet_positions` adds the fleet [F/stride,3] scatter — the only
    fleet-sized pull the viewer ever makes; `fleet_stride` thins it ON DEVICE, so a
    100k-world fleet never crosses the host bus whole (TECH_DEBT V5).
    """
    import jax  # deferred: keeps this module importable for pure-numpy consumers

    idx = np.asarray(watch, np.int32)
    fleet = state.plant.shape[0]
    chans = dict(channels or {})
    if reward is not None:
        chans.setdefault("reward", reward)

    def rows(a: Any) -> Any:
        return None if a is None else a[idx]

    task_rows = jax.tree.map(
        lambda leaf: leaf[idx]
        if getattr(leaf, "ndim", 0) >= 1 and leaf.shape[0] == fleet
        else leaf,
        task_state if task_state is not None else state.task_state,
    )
    info_rows = (
        {
            k: v[idx]
            for k, v in info.items()
            if getattr(v, "ndim", 0) >= 1 and v.shape[0] == fleet
        }
        if info
        else None
    )
    bundle = (
        rows(state.plant),
        rows(action),
        {k: rows(v) for k, v in chans.items()},
        rows(done),
        rows(obs),
        task_rows,
        info_rows,
        state.steps[idx[focus]],
        state.plant[:: max(1, int(fleet_stride)), 0:3] if fleet_positions else None,
    )
    plant, action_w, chans_w, done_w, obs_w, ts, info_w, steps, positions = jax.device_get(
        bundle
    )
    step = int(steps)
    return ViewFrame(
        plant=np.asarray(plant),
        step=step,
        t=step * float(dt),
        focus=int(focus),
        action=None if action_w is None else np.asarray(action_w),
        channels={k: np.asarray(v) for k, v in chans_w.items()},
        done=None if done_w is None else np.asarray(done_w),
        obs=None if obs_w is None else np.asarray(obs_w),
        task_state=ts,
        info=info_w,
        positions=None if positions is None else np.asarray(positions),
    )
