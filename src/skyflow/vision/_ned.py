"""
Frame conversion at the vision boundary — z-up FLU outside, NED/FRD inside (DESIGN.md §3a).

The renderer math in this package is ported from a NED/FRD implementation verified against
MuJoCo segmentation renders, and stays in that frame. These helpers are the single
conversion site its public entry points use: world z-up ↔ NED and body FLU ↔ FRD are the
SAME (x, −y, −z) flip, so one vector negation converts positions and one quaternion
conjugation converts attitudes for both frame pairs at once. All maps here are
self-inverse. Frame bookkeeping only — no physics (DESIGN.md §1 boundary).

Ported from the verified nav-train pose helpers (plant.pose_ned and friends).
"""

import jax
import jax.numpy as jnp


def flip_xyz(v: jax.Array) -> jax.Array:
    """FLU↔FRD body / z-up↔NED world: negate y and z. Batched ``[..., 3]``, self-inverse."""
    return v * jnp.array([1.0, -1.0, -1.0], v.dtype)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Hamilton product of wxyz quaternions, batched ``[..., 4]``."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


# 180° rotation about body-x (wxyz): conjugating by it maps FLU↔FRD body and z-up↔NED
# world in one shot — the quaternion twin of flip_xyz. Self-inverse.
_QF_FLIP = jnp.array([0.0, 1.0, 0.0, 0.0], jnp.float32)


def quat_flip(quat: jax.Array) -> jax.Array:
    """
    Re-express a body→world attitude between the (FLU, z-up) and (FRD, NED) frame pairs:
    wxyz ``[..., 4]``, self-inverse (the flip quaternion is its own inverse up to sign).
    """
    qf = jnp.broadcast_to(_QF_FLIP, quat.shape)
    return quat_mul(qf, quat_mul(quat, qf))


def pose_ned(pos: jax.Array, quat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Public z-up FLU pose → internal NED/FRD pose: ``pos [..., 3]`` world metres,
    ``quat [..., 4]`` wxyz body→world. The inverse is the same call (self-inverse maps).
    """
    return flip_xyz(pos), quat_flip(quat)
