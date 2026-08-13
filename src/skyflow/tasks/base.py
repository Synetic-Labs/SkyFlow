"""
Observation helpers shared by the example tasks (DESIGN.md §2).

Packaging utilities, not physics: frame conversion for observation blocks and the final
sanitation every task applies before handing obs to the env. The Task protocol itself
lives in types.py — this module only helps implement it. Frames follow DESIGN.md §3:
world z-up, body FLU, quaternions wxyz scalar-first Hamilton body→world.
"""

import jax.numpy as jnp

from skyflow.types import Array

#: Body rates enter observations as ω / OBS_RATE_SCALE — rad/s scaled so aggressive
#: flight (|ω| ≈ 10 rad/s) lands near ±1, keeping rate blocks comparable across tasks.
OBS_RATE_SCALE = 10.0

#: finalize_obs clip half-width. Large enough that any sane observation passes
#: untouched; small enough that a diverged world cannot poison a training batch.
_OBS_CLIP = 100.0


def quat_to_rot(q_wxyz: Array) -> Array:
    """
    Body→world rotation matrix [..., 3, 3] from a wxyz Hamilton quaternion [..., 4].

    Observation-side frame conversion only — dynamics and sensor math stay generated
    (DESIGN.md §1). Assumes unit norm: plant quaternions are renormalized by the backend
    post-step every substep, and a defensive second normalization would hide bugs.
    """
    w, x, y, z = (q_wxyz[..., i] for i in range(4))
    r = jnp.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(*q_wxyz.shape[:-1], 3, 3)


def world_to_body(q_wxyz: Array, v_world: Array) -> Array:
    """
    World-frame vectors [..., 3] into body FLU: Rᵀ·v for the body→world rotation of
    q_wxyz [..., 4]. Batch shapes broadcast; the trailing axis is the vector.
    """
    return jnp.einsum("...ij,...i->...j", quat_to_rot(q_wxyz), v_world)


def finalize_obs(obs: Array) -> Array:
    """
    The last thing every task does to an observation: nan_to_num → float32 → clip ±100.

    nan_to_num runs first so NaN becomes 0 and ±inf becomes the float maximum, which the
    clip then folds to ±100 — every returned entry is finite float32 no matter how badly
    a diverged world's state blew up.
    """
    return jnp.clip(jnp.nan_to_num(obs).astype(jnp.float32), -_OBS_CLIP, _OBS_CLIP)
