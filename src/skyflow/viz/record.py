"""
FlightLog — pose logs, never pixels (DESIGN.md §13).

The camera is a pure function of pose, so poses are a sufficient statistic for every view
at every resolution, forever: a watch row costs ~96 bytes per control step, two orders of
magnitude below even the 64x64 mask stream it can reproduce. Two capture paths feed one
format: `capture(state, ...)` slices watch rows per control step (live loops), and
`extend(...)` takes whole [T, W|F, ...] chunks — e.g. a pose buffer carried through a
training scan and pulled once per chunk, so the fused scan is never stalled.

`flight.npz` is self-describing: a JSON header (serialized scene, camera, gate geometry,
airframe, config fields) plus arrays — plant, action, done, named channel traces
("ch:<name>") and the values behind the scene's string binds ("bind:<path>"), so replay
resolves the same binds and plots the same channels with no task code. Pure numpy + json;
jax and vision imports stay inside functions.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FlightLog", "ReplayLog", "gateset_from_dict", "gateset_to_dict"]


def gateset_to_dict(gates: Any) -> dict:
    """
    GateSet → JSON-safe dict, via its internal NED field arrays. Deliberately the raw
    fields, not the z-up builders: an exact round-trip (DESIGN.md §3 allows NED at
    contained boundaries; this dict exists only to rebuild the same GateSet on replay).
    """
    return {
        "centers": np.asarray(gates.centers).tolist(),
        "yaws": np.asarray(gates.yaws).tolist(),
        "inner_half": np.asarray(gates.inner_half).tolist(),
        "outer_half": np.asarray(gates.outer_half).tolist(),
        "pitches": np.asarray(gates.pitches).tolist(),
        "depths": np.asarray(gates.depths).tolist(),
    }


def gateset_from_dict(d: dict) -> Any:
    """Rebuild the GateSet recorded by :func:`gateset_to_dict`."""
    import jax.numpy as jnp

    from skyflow.vision.gates import GateSet

    return GateSet(
        centers=jnp.asarray(d["centers"], jnp.float32),
        yaws=jnp.asarray(d["yaws"], jnp.float32),
        inner_half=jnp.asarray(d["inner_half"], jnp.float32),
        outer_half=jnp.asarray(d["outer_half"], jnp.float32),
        pitches=jnp.asarray(d["pitches"], jnp.float32),
        depths=jnp.asarray(d["depths"], jnp.float32),
    )


@dataclasses.dataclass
class ReplayLog:
    """A loaded flight.npz: header dict + [T, W, ...] arrays (empty dicts when absent)."""

    header: dict
    plant: np.ndarray  # [T,W,17]
    action: np.ndarray | None
    done: np.ndarray | None
    channels: dict[str, np.ndarray]  # named scalar traces, each [T,W]
    binds: dict[str, np.ndarray]  # values of the scene's string binds, keyed by path

    def __len__(self) -> int:
        return int(self.plant.shape[0])

    @property
    def dt(self) -> float:
        """Wall seconds per logged row (control period x capture stride)."""
        return float(self.header.get("dt", 0.01)) * int(self.header.get("every", 1))


class FlightLog:
    """Append-only recorder for the watched worlds; `save` writes flight.npz."""

    def __init__(
        self,
        *,
        watch: tuple[int, ...] = (0,),
        every: int = 1,
        header: dict | None = None,
        task_state_of: Any = None,
    ):
        """
        Args:
          watch: fleet rows to record.
          every: capture stride in control steps (capture() counts calls; extend()
            callers are expected to pass already-strided chunks).
          header: replay context — `for_env` fills it; hand-built logs may pass their own.
          task_state_of: callable state → the task's own pytree (for_env wires
            `env.task_state`, which unwraps the sticks-mode firmware carry).
        """
        self.watch = tuple(int(w) for w in watch)
        self.every = max(1, int(every))
        self.header = dict(header or {})
        self._task_state_of = task_state_of
        self.header.setdefault("watch", list(self.watch))
        self.header.setdefault("every", self.every)
        self._idx = np.asarray(self.watch, np.int32)
        self._n_seen = 0
        self._plant: list[np.ndarray] = []
        self._action: list[np.ndarray | None] = []
        self._done: list[np.ndarray | None] = []
        self._channels: dict[str, list[np.ndarray]] = {}
        self._binds: dict[str, list[np.ndarray]] = {}
        # scene primitives declare their live inputs as string binds; capture records
        # those values so replay resolves the same binds with no task code (§13)
        self._bind_paths = sorted(
            {
                d["bind"]
                for d in self.header.get("scene", [])
                if isinstance(d.get("bind"), str)
            }
        )

    @classmethod
    def for_env(
        cls,
        env: Any,
        watch: tuple[int, ...] = (0,),
        every: int = 1,
        scene: Any = None,
    ) -> "FlightLog":
        """Log wired to a SkyFlowEnv: the header carries everything replay needs."""
        task = env.task
        if scene is None:
            hook = getattr(task, "viz_scene", None)
            scene_dicts = hook() if callable(hook) else []
        else:
            scene_dicts = scene.to_dicts()
        camera = getattr(task, "camera", None)
        gates = getattr(task, "gates", None)
        from skyflow import __version__

        header = {
            "skyflow": __version__,
            "dt": float(env.dt_control),
            "control": env.cfg.control,
            "airframe": env.cfg.airframe,
            "task": env.cfg.task,
            "omega_max": float(env.airframe.rotor_speed_max),
            "image_shape": list(env.image_shape) if env.image_shape else None,
            "scene": scene_dicts,
        }
        if camera is not None:
            header["camera"] = dataclasses.asdict(camera)
        if gates is not None:
            header["gateset"] = gateset_to_dict(gates)
        return cls(
            watch=watch,
            every=every,
            header=header,
            task_state_of=getattr(env, "task_state", None),
        )

    def __len__(self) -> int:
        return len(self._plant)

    def _resolve_path(self, state: Any, path: str) -> Any:
        parts = path.split(".")
        # the first hop honors the env's task-state accessor (sticks-mode carry, §10)
        if parts[0] == "task_state" and self._task_state_of is not None:
            obj: Any = self._task_state_of(state)
            parts = parts[1:]
        else:
            obj = state
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def capture(
        self,
        state: Any,
        *,
        action: Any = None,
        reward: Any = None,
        channels: dict[str, Any] | None = None,
        done: Any = None,
    ) -> None:
        """
        Record this control step's watch rows (respecting `every`). Live-loop path.
        `channels` are named [F] scalars; `reward` is shorthand for the "reward" channel.
        The values behind the scene's string binds are recorded too, so replay resolves
        the same binds with no task code.
        """
        self._n_seen += 1
        if (self._n_seen - 1) % self.every:
            return
        idx = self._idx

        def rows(a: Any) -> np.ndarray | None:
            return None if a is None else np.asarray(a[idx])

        self._plant.append(np.asarray(state.plant[idx]))
        self._action.append(rows(action))
        self._done.append(rows(done))
        chans = dict(channels or {})
        if reward is not None:
            chans.setdefault("reward", reward)
        for name, values in chans.items():
            self._channels.setdefault(name, []).append(np.asarray(values)[idx])
        for p in self._bind_paths:
            val = self._resolve_path(state, p)
            if val is not None:
                self._binds.setdefault(p, []).append(np.asarray(val)[idx])

    def extend(
        self,
        plant: Any,
        *,
        action: Any = None,
        reward: Any = None,
        channels: dict[str, Any] | None = None,
        done: Any = None,
        binds: dict[str, Any] | None = None,
    ) -> None:
        """
        Record whole [T, W|F, ...] chunks — the training path: accumulate a pose buffer
        in the scan carry, `device_get` once per chunk, hand it here. Fleet-sized chunks
        are sliced to the watch list; watch-sized chunks pass through. `binds` carries
        the scene's bind values by path (e.g. {"task_state.active_gate": [T,F]}).
        """
        w = len(self.watch)

        def norm(a: Any) -> np.ndarray | None:
            if a is None:
                return None
            arr = np.asarray(a)
            return arr if arr.shape[1] == w else arr[:, self._idx]

        plant_a = norm(plant)
        assert plant_a is not None
        chans = {k: norm(v) for k, v in (channels or {}).items()}
        if reward is not None:
            chans.setdefault("reward", norm(reward))
        bind_a = {k: norm(v) for k, v in (binds or {}).items()}
        action_a, done_a = norm(action), norm(done)
        for t in range(plant_a.shape[0]):
            self._plant.append(plant_a[t])
            self._action.append(None if action_a is None else action_a[t])
            self._done.append(None if done_a is None else done_a[t])
            for name, arr in chans.items():
                self._channels.setdefault(name, []).append(arr[t])  # type: ignore[index]
            for p, arr in bind_a.items():
                self._binds.setdefault(p, []).append(arr[t])  # type: ignore[index]

    def save(self, path: str | Path) -> Path:
        """Write flight.npz (compressed); returns the path."""
        if not self._plant:
            raise ValueError("nothing recorded — capture() or extend() first")
        n = len(self._plant)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"plant": np.stack(self._plant)}
        for name, buf in (("action", self._action), ("done", self._done)):
            if all(b is not None for b in buf):
                arrays[name] = np.stack(buf)  # type: ignore[arg-type]
        for prefix, table in (("ch:", self._channels), ("bind:", self._binds)):
            for name, buf in table.items():
                if len(buf) != n:
                    raise ValueError(
                        f"{prefix}{name} has {len(buf)} rows but plant has {n} — "
                        "pass the same channels/binds on every capture/extend"
                    )
                arrays[prefix + name] = np.stack(buf)
        np.savez_compressed(path, header=json.dumps(self.header), **arrays)
        return path

    @staticmethod
    def load(path: str | Path) -> ReplayLog:
        """Read a flight.npz back into a ReplayLog."""
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            return ReplayLog(
                header=header,
                plant=z["plant"],
                action=z["action"] if "action" in z else None,
                done=z["done"] if "done" in z else None,
                channels={k[3:]: z[k] for k in z.files if k.startswith("ch:")},
                binds={k[5:]: z[k] for k in z.files if k.startswith("bind:")},
            )
