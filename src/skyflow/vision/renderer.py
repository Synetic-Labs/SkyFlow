"""
Analytic gate segmentation-mask renderer (pure JAX, fleet-batched).

A vision-based gate policy cannot carry a real renderer inside an all-JAX training loop —
the rollout is one jitted scan over the whole fleet, and a host-side render per step would
be the throughput wall. But the only thing the policy needs to *see* is a coverage mask of
the gate frames, exactly what a deployed HSV/segmentation front-end produces from a real
camera frame. Flat gates are trivial geometry, so that mask renders analytically.

Method — per-pixel ray-cast against each gate's SOLID frame (not forward polygon
projection): back-project each pixel into a world ray and intersect it with the gate solid
— the outer box (outer rectangle x ``depths[g]`` thick along the normal) minus the
inner-opening prism — via slab intervals in the gate's own axis frame. A pixel is marked
when the ray's inside-the-outer-box interval has any part outside the opening interval and
in front of the camera. This renders the true silhouette: at oblique views the bars' side
faces widen the band exactly as a physical gate does; ``depths[g] = 0`` degenerates to the
flat-plane band. Ray-casting handles partial visibility and gate-behind-camera for free.
Frames are the only geometry and they all belong to the same mask class, so the union over
gates needs no depth sorting: a frame seen through another gate's opening, or occluded by
a nearer frame, marks the pixel either way. The mask is exact up to rasterization.

Frame contract (DESIGN.md §3): public entry points take the world z-up FLU pose —
``pos`` [F, 3] z-up metres, ``quat`` [F, 4] wxyz Hamilton body FLU → world — and per-world
gate geometry in z-up world coordinates; the conversion to the NED/FRD internals
happens exactly once per call (vision._ned, §3a). The camera looks along body +x pitched
by its mount; its image frame is the standard pinhole convention (x right, y down,
z forward / optical axis).

The ray-cast math is validated in isolation against MuJoCo segmentation renders.
"""

import jax
import jax.numpy as jnp

from skyflow.vision._ned import flip_xyz, pose_ned
from skyflow.vision.camera import CameraModel
from skyflow.vision.gates import _BIG, GateSet, _frame_solid_interval, _hits_solid


def _rotate(quat_wxyz: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate body vector ``v`` into the world frame (R(q) v), batched over the leading axis."""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return jnp.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], axis=-1)


def _camera_rays(
    cam: CameraModel,
    pos_ned: jax.Array,
    quat_ned: jax.Array,
    R_body_from_cam: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """
    Shared ray setup, internal NED/FRD: camera origin [F, 3] and per-pixel world ray
    directions [F, P, 3] (P = H·ss · W·ss, un-normalised).

    ``R_body_from_cam`` is the PUBLIC per-agent mount override — camera frame → body FLU,
    [F, 3, 3] — or None for the camera's nominal mount; the FLU→FRD row flip happens here
    so override and nominal go through identical internal math.
    """
    fleet = pos_ned.shape[0]
    offset_frd = jnp.asarray(cam._offset_frd, jnp.float32)

    # camera origin in the world: body origin + R(q) · offset
    cam_origin = pos_ned + _rotate(quat_ned, jnp.broadcast_to(offset_frd, (fleet, 3)))

    # per-pixel ray directions: cam frame → body frame → world frame. Flatten the pixel
    # grid for the matmuls.
    rays_cam = cam.ray_dirs_cam.reshape(-1, 3)                       # [P, 3]
    if R_body_from_cam is None:
        rays_body = rays_cam @ cam._R_frd_from_cam.T                 # [P, 3]
        rays_body = jnp.broadcast_to(rays_body, (fleet, *rays_body.shape))
    else:
        # public FLU rotation → internal FRD (negate body y/z rows), per agent
        r_frd = R_body_from_cam * jnp.array([1.0, -1.0, -1.0], jnp.float32)[:, None]
        rays_body = jnp.einsum("fij,pj->fpi", r_frd, rays_cam)       # [F, P, 3]
    rays_world = _rotate(quat_ned[:, None, :], rays_body)            # [F, P, 3]
    return cam_origin, rays_world


def render_masks(
    cam: CameraModel,
    gates: GateSet,
    pos: jax.Array,
    quat: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    outer_grow: jax.Array | None = None,
) -> jax.Array:
    """
    Render the gate coverage mask (union over all gates) for a fleet.

    Args:
      cam: camera model (intrinsics + nominal FLU mount).
      gates: gate geometry (GateSet), any number of gates, shared by the whole fleet.
      pos: drone positions [F, 3], world z-up metres.
      quat: drone attitudes [F, 4], wxyz Hamilton, body FLU → world.
      R_body_from_cam: optional per-agent override of the body(FLU)←camera rotation,
        [F, 3, 3] — lets the task inject per-episode extrinsic jitter (domain
        randomization). Defaults to the camera's nominal mount for all agents.
      outer_grow: optional per-agent OUTWARD widening of every gate's outer edge, [F]
        metres — models the glow bleed of a real HSV mask, whose outer boundary sits
        wider than the physical frame while the inner boundary stays true (see
        mask_noise.grow_from_keys for the persistent draw). Only the render widens:
        contact geometry is untouched.

    Returns:
      mask [F, H, W] float32 in [0, 1] — the fraction of a pixel's supersample² rays that
      hit any gate's frame band in front of the camera (binary when cam.supersample == 1).
    """
    H, W, ss = cam.height, cam.width, cam.supersample
    pos_ned, quat_ned = pose_ned(pos, quat)
    cam_origin, rays_world = _camera_rays(cam, pos_ned, quat_ned, R_body_from_cam)
    fleet = pos_ned.shape[0]

    # Per-gate ray-solid intersection, unioned at subray resolution. A Python loop over
    # the (static, small) gate count keeps peak memory at F·P per gate instead of
    # materializing an F·P·G block; the frame bars share one mask class, so a boolean OR
    # is the exact composite (no depth sorting).
    grow = None if outer_grow is None else outer_grow[:, None]       # [F, 1]
    band_any = jnp.zeros(rays_world.shape[:2], bool)                 # [F, P]
    for g in range(len(gates)):
        oc = cam_origin - gates.centers[g]                           # [F, 3]
        # gate-frame coordinates: origin [F, 1] broadcast against dirs [F, P]
        oo_n = jnp.sum(oc * gates.normals[g], axis=-1)[:, None]
        oo_l = jnp.sum(oc * gates.laterals[g], axis=-1)[:, None]
        oo_v = jnp.sum(oc * gates.verticals[g], axis=-1)[:, None]
        dd_n = jnp.sum(rays_world * gates.normals[g], axis=-1)       # [F, P]
        dd_l = jnp.sum(rays_world * gates.laterals[g], axis=-1)
        dd_v = jnp.sum(rays_world * gates.verticals[g], axis=-1)
        ow, oh = gates.outer_half[g, 0], gates.outer_half[g, 1]
        if grow is not None:
            ow, oh = ow + grow, oh + grow                            # [F, 1]
        a0, a1, b0, b1 = _frame_solid_interval(
            oo_n, dd_n, oo_l, dd_l, oo_v, dd_v,
            0.5 * gates.depths[g], ow, oh,
            gates.inner_half[g, 0], gates.inner_half[g, 1])
        band_any = band_any | _hits_solid(a0, a1, b0, b1, 1e-6, _BIG)

    # [F, H·ss, W·ss] -> average each ss x ss block into a soft [F, H, W] coverage
    band = band_any.astype(jnp.float32).reshape(fleet, H, ss, W, ss)
    return band.mean(axis=(2, 4))


def render_floor(
    cam: CameraModel,
    pos: jax.Array,
    quat: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    half_extent: float | None = None,
) -> jax.Array:
    """
    Render the ground-plane (world z = 0) coverage for a fleet — the floor channel.

    A pixel is covered when its camera ray crosses the ground plane in FRONT of the
    camera (the drone flies above ground, z-up z > 0, so a downward ray hits it);
    everything above the horizon misses and stays sky. Pose contract as in
    :func:`render_masks` (z-up FLU). ``half_extent`` optionally clips the floor to a
    ±half_extent metre square about the world origin; None = the whole ground plane.

    Returns coverage [F, H, W] float32 in [0, 1] (soft via cam.supersample).
    """
    H, W, ss = cam.height, cam.width, cam.supersample
    pos_ned, quat_ned = pose_ned(pos, quat)
    cam_origin, rays_world = _camera_rays(cam, pos_ned, quat_ned, R_body_from_cam)
    fleet = pos_ned.shape[0]

    oz = cam_origin[:, None, 2]                                     # [F, 1] NED z (down +)
    dz = rays_world[..., 2]                                         # [F, P]
    t = (0.0 - oz) / jnp.where(jnp.abs(dz) < 1e-9, 1e-9, dz)
    hit = (t > 1e-6) & (jnp.abs(dz) > 1e-9)                         # ground in front of the camera
    if half_extent is not None:
        nx = cam_origin[:, None, 0] + t * rays_world[..., 0]
        ny = cam_origin[:, None, 1] + t * rays_world[..., 1]
        hit = hit & (jnp.abs(nx) <= half_extent) & (jnp.abs(ny) <= half_extent)
    return hit.astype(jnp.float32).reshape(fleet, H, ss, W, ss).mean(axis=(2, 4))


def render_masks_perworld(
    cam: CameraModel,
    centers: jax.Array, normals: jax.Array, laterals: jax.Array, verticals: jax.Array,
    inner_half: jax.Array, outer_half: jax.Array, depths: jax.Array,
    pos: jax.Array, quat: jax.Array,
    *,
    R_body_from_cam: jax.Array | None = None,
    outer_grow: jax.Array | None = None,
) -> jax.Array:
    """
    :func:`render_masks` with PER-WORLD (per-episode) gate geometry.

    Same ray-cast against each gate's solid frame and the same OR-union over gates, but
    every gate array carries a leading FLEET axis so each world can see gates in a
    different place (:func:`render_masks` bakes a single fixed GateSet shared by the whole
    fleet). Used by tasks that RANDOMIZE gate placement per episode.

    All gate geometry is world z-up FLU (converted internally once): ``centers`` /
    ``normals`` / ``laterals`` / ``verticals`` are [F, G, 3]; ``inner_half`` /
    ``outer_half`` [F, G, 2] metres; ``depths`` [F, G] metres. The plane axes must follow
    the GateSet convention read back by its ``*_world`` properties (normal from yaw about
    +z then pitch up-positive; laterals horizontal; verticals in-plane). ``pos`` [F, 3],
    ``quat`` [F, 4] wxyz body FLU → world; ``R_body_from_cam`` [F, 3, 3] and
    ``outer_grow`` [F] as in :func:`render_masks`. Returns the mask [F, H, W] in [0, 1].
    """
    H, W, ss = cam.height, cam.width, cam.supersample
    pos_ned, quat_ned = pose_ned(pos, quat)
    # world z-up gate geometry → internal NED: the same (x, -y, -z) flip on every world
    # vector (orthogonal, so all gate-frame dot products below are preserved exactly)
    centers = flip_xyz(centers)
    normals = flip_xyz(normals)
    laterals = flip_xyz(laterals)
    verticals = flip_xyz(verticals)
    cam_origin, rays_world = _camera_rays(cam, pos_ned, quat_ned, R_body_from_cam)
    fleet = pos_ned.shape[0]

    grow = None if outer_grow is None else outer_grow[:, None]       # [F, 1]
    G = centers.shape[1]
    band_any = jnp.zeros(rays_world.shape[:2], bool)                 # [F, P]
    for g in range(G):
        c = centers[:, g]                                           # [F, 3]
        nrm, lat, ver = normals[:, g], laterals[:, g], verticals[:, g]  # [F, 3] each
        oc = cam_origin - c                                         # [F, 3]
        oo_n = jnp.sum(oc * nrm, axis=-1)[:, None]                  # [F, 1]
        oo_l = jnp.sum(oc * lat, axis=-1)[:, None]
        oo_v = jnp.sum(oc * ver, axis=-1)[:, None]
        dd_n = jnp.sum(rays_world * nrm[:, None, :], axis=-1)       # [F, P]
        dd_l = jnp.sum(rays_world * lat[:, None, :], axis=-1)
        dd_v = jnp.sum(rays_world * ver[:, None, :], axis=-1)
        ow, oh = outer_half[:, g, 0][:, None], outer_half[:, g, 1][:, None]   # [F, 1]
        if grow is not None:
            ow, oh = ow + grow, oh + grow
        a0, a1, b0, b1 = _frame_solid_interval(
            oo_n, dd_n, oo_l, dd_l, oo_v, dd_v,
            0.5 * depths[:, g][:, None], ow, oh,
            inner_half[:, g, 0][:, None], inner_half[:, g, 1][:, None])
        band_any = band_any | _hits_solid(a0, a1, b0, b1, 1e-6, _BIG)

    band = band_any.astype(jnp.float32).reshape(fleet, H, ss, W, ss)
    return band.mean(axis=(2, 4))
