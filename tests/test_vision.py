"""
Vision suite (DESIGN.md §11): camera ray invariants, renderer behavior at the z-up FLU
boundary, mask-noise range/persistence/identity, and figure_eight course geometry.

CPU, deterministic keys. The analytic ray-cast geometry is cross-checked against MuJoCo
segmentation renders; what these tests pin is the SkyFlow surface — the frame conversion
at every public entry point (world z-up FLU in, DESIGN.md §3), shapes and dtypes, and the
figure-eight builder (DESIGN.md §9).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyflow.vision import (
    CameraModel,
    GateSet,
    classify_crossings,
    corrupt_mask,
    erasure_at,
    figure_eight,
    fresh_noise_keys,
    noise_state_init,
    noise_state_step,
    render_floor,
    render_masks,
    render_masks_perworld,
)
from skyflow.vision.mask_noise import _blob_field1, _blob_field_at

IDENTITY_QUAT = jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32)


def _random_quats(key, n: int) -> jnp.ndarray:
    """n uniform-ish random unit quaternions wxyz [n, 4] (normalized Gaussians)."""
    q = jax.random.normal(key, (n, 4), jnp.float32)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


# -- camera ray invariants -----------------------------------------------------------


def test_mean_ray_is_optical_axis():
    """The supersampled grid is symmetric about the principal point, so the mean ray
    direction IS the optical axis: (0, 0, 1) in the camera frame, body +x for a level
    mount, and pitched up by 25 deg for the default -25 deg mount (z-up FLU: z > 0)."""
    level = CameraModel(mount_pitch_deg=0.0)
    mean_dir = np.asarray(level.ray_dirs_cam.reshape(-1, 3).mean(axis=0))
    assert np.allclose(mean_dir, [0.0, 0.0, 1.0], atol=1e-5)

    axis_flu = np.asarray(level.R_body_from_cam) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(axis_flu, [1.0, 0.0, 0.0], atol=1e-6)

    tilted = CameraModel()  # default -25 deg = 25 deg up-tilt
    axis_flu = np.asarray(tilted.R_body_from_cam) @ np.array([0.0, 0.0, 1.0])
    t = math.radians(25.0)
    assert np.allclose(axis_flu, [math.cos(t), 0.0, math.sin(t)], atol=1e-6)
    # rotation sanity: orthonormal, right-handed
    r = np.asarray(tilted.R_body_from_cam, np.float64)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_fov_edges():
    """Focal lengths encode the configured FOV exactly (f = (size/2)/tan(fov/2)); the
    outermost subrays sit within half a subpixel inside the FOV boundary, symmetric."""
    cam = CameraModel()
    fx, fy = cam.focal
    assert np.isclose(2.0 * math.degrees(math.atan((cam.width / 2.0) / fx)), cam.fov_x_deg)
    assert np.isclose(2.0 * math.degrees(math.atan((cam.height / 2.0) / fy)), cam.fov_y_deg)

    rays = np.asarray(cam.ray_dirs_cam)
    u, v = rays[..., 0], rays[..., 1]
    assert np.isclose(u.max(), -u.min())  # left/right symmetry about the principal point
    assert np.isclose(v.max(), -v.min())  # up/down symmetry
    half_x = math.radians(cam.fov_x_deg) / 2.0
    edge = math.atan(u.max())
    subpixel = math.atan(0.5 / (cam.supersample * fx))
    assert half_x - subpixel * 1.01 < edge < half_x


# -- renderer at the z-up FLU boundary -------------------------------------------------


def test_gateset_world_accessors():
    """Builders take z-up world definitions; *_world properties read them back."""
    gs = GateSet.single(center=(3.0, 0.0, 1.5), yaw=0.0)
    assert np.allclose(np.asarray(gs.centers_world), [[3.0, 0.0, 1.5]])
    assert np.allclose(np.asarray(gs.normals_world), [[1.0, 0.0, 0.0]], atol=1e-6)
    assert np.allclose(np.asarray(gs.verticals_world), [[0.0, 0.0, 1.0]], atol=1e-6)

    gs = GateSet.single(center=(0.0, 2.0, 1.0), yaw=math.pi / 2)
    assert np.allclose(np.asarray(gs.normals_world), [[0.0, 1.0, 0.0]], atol=1e-6)

    # pitch is up-positive: the through axis gains an upward (+z) component
    gs = GateSet.single(center=(3.0, 0.0, 1.5), yaw=0.0, pitch=0.3)
    n = np.asarray(gs.normals_world)[0]
    assert np.allclose(n, [math.cos(0.3), 0.0, math.sin(0.3)], atol=1e-6)


def test_dead_ahead_gate_centroid_at_image_center():
    """A gate dead ahead of a level camera renders symmetric: mask centroid within half a
    pixel of the image center."""
    cam = CameraModel(mount_pitch_deg=0.0, offset_body=(0.0, 0.0, 0.0))
    gates = GateSet.single(center=(3.0, 0.0, 1.5), yaw=0.0)
    pos = jnp.array([[0.0, 0.0, 1.5]], jnp.float32)
    mask = render_masks(cam, gates, pos, IDENTITY_QUAT)

    assert mask.shape == (1, cam.height, cam.width)
    assert mask.dtype == jnp.float32
    m = np.asarray(mask[0])
    total = m.sum()
    assert total > 0.0  # the gate is visible
    rows = np.arange(cam.height)
    cols = np.arange(cam.width)
    cy = (m.sum(axis=1) * rows).sum() / total
    cx = (m.sum(axis=0) * cols).sum() / total
    assert abs(cy - (cam.height - 1) / 2.0) < 0.5
    assert abs(cx - (cam.width - 1) / 2.0) < 0.5


def test_gate_behind_camera_renders_zero():
    """Rays only march forward (t > 0): a gate behind the camera marks nothing."""
    cam = CameraModel(mount_pitch_deg=0.0, offset_body=(0.0, 0.0, 0.0))
    gates = GateSet.single(center=(-3.0, 0.0, 1.5), yaw=0.0)
    pos = jnp.array([[0.0, 0.0, 1.5]], jnp.float32)
    mask = render_masks(cam, gates, pos, IDENTITY_QUAT)
    assert np.all(np.asarray(mask) == 0.0)


def test_mask_in_unit_interval_random_poses(key):
    """Soft coverage stays in [0, 1] float32 for arbitrary poses over a full course."""
    f = 6
    gates = figure_eight(2, lobe_radius_m=2.0, alt_m=1.5)
    kp, kq = jax.random.split(key)
    pos = jax.random.uniform(kp, (f, 3), jnp.float32, minval=-5.0, maxval=5.0)
    pos = pos.at[:, 2].set(jnp.abs(pos[:, 2]))
    quat = _random_quats(kq, f)
    mask = render_masks(CameraModel(), gates, pos, quat)
    assert mask.shape == (f, 64, 64)
    assert mask.dtype == jnp.float32
    m = np.asarray(mask)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_floor_horizon():
    """Ground plane z = 0 (z-up): sky above the horizon renders 0, ground below is full
    coverage; a half_extent clip can only remove coverage."""
    cam = CameraModel()  # 25 deg up-tilt: horizon well inside the frame
    pos = jnp.array([[0.0, 0.0, 1.5]], jnp.float32)
    floor = np.asarray(render_floor(cam, pos, IDENTITY_QUAT))
    assert floor.min() >= 0.0 and floor.max() <= 1.0
    assert floor[0, 0, :].max() == 0.0       # top row: sky
    assert floor[0, -1, :].min() >= 0.999    # bottom row: ground, every subray hits
    clipped = np.asarray(render_floor(cam, pos, IDENTITY_QUAT, half_extent=2.0))
    assert np.all(clipped <= floor + 1e-6)
    assert clipped.sum() < floor.sum()


def test_perworld_matches_shared_renderer(key):
    """render_masks_perworld with each world's arrays set to the shared course (read back
    through the z-up accessors) reproduces render_masks exactly — pins the per-world
    z-up boundary conversion against the GateSet one."""
    f = 3
    gates = figure_eight(2, lobe_radius_m=2.0, alt_m=1.5)
    g = len(gates)
    kp, kq = jax.random.split(key)
    pos = jax.random.uniform(kp, (f, 3), jnp.float32, minval=-3.0, maxval=3.0)
    pos = pos.at[:, 2].set(1.0 + jnp.abs(pos[:, 2]))
    quat = _random_quats(kq, f)
    cam = CameraModel()

    def tile(a):
        return jnp.broadcast_to(a, (f, *a.shape))

    per = render_masks_perworld(
        cam,
        tile(gates.centers_world), tile(gates.normals_world),
        tile(gates.laterals_world), tile(gates.verticals_world),
        tile(gates.inner_half), tile(gates.outer_half),
        jnp.broadcast_to(gates.depths, (f, g)),
        pos, quat)
    shared = render_masks(cam, gates, pos, quat)
    assert np.allclose(np.asarray(per), np.asarray(shared), atol=1e-6)


# -- crossing classification (z-up boundary) --------------------------------------------


def test_classify_crossings_pass_hit_miss():
    gates = GateSet.single(center=(2.0, 0.0, 1.5), yaw=0.0)
    segs = jnp.array(
        [
            [[0.0, 0.0, 1.5], [4.0, 0.0, 1.5]],    # through the opening, forward
            [[4.0, 0.0, 1.5], [0.0, 0.0, 1.5]],    # same path, backward
            [[0.0, 0.4, 1.5], [4.0, 0.4, 1.5]],    # 0.4 m lateral: inside the frame band
            [[0.0, 0.0, 2.5], [4.0, 0.0, 2.5]],    # 1.0 m above: clears the outer edge
        ],
        jnp.float32,
    )
    fwd, bwd, hit = classify_crossings(segs[:, 0], segs[:, 1], gates)
    assert np.array_equal(np.asarray(fwd[:, 0]), [True, False, False, False])
    assert np.array_equal(np.asarray(bwd[:, 0]), [False, True, False, False])
    assert np.array_equal(np.asarray(hit[:, 0]), [False, False, True, False])


# -- figure_eight builder (DESIGN.md §9) -------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4])
def test_figure_eight_counts_altitude_extent(k):
    r, alt = 2.0, 1.3
    gs = figure_eight(k, lobe_radius_m=r, alt_m=alt)
    assert len(gs) == 2 * k
    c = np.asarray(gs.centers_world)
    assert np.allclose(c[:, 2], alt, atol=1e-6)          # all gates at alt_m
    assert np.max(np.abs(c[:, 0])) <= 2 * r + 1e-5       # lobes span 2*lobe_radius in x
    if k % 2 == 1:  # odd k puts a gate exactly on each lobe apex
        assert np.isclose(np.max(c[:, 0]), 2 * r, atol=1e-5)
        assert np.isclose(np.min(c[:, 0]), -2 * r, atol=1e-5)
    n = np.asarray(gs.normals_world)
    assert np.allclose(n[:, 2], 0.0, atol=1e-6)          # flat course: horizontal normals
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("k", [2, 3])
def test_figure_eight_alternating_center_transits(k):
    """Consecutive transits of the crossover run along the two DIFFERENT diagonals: both
    lobe-to-lobe chords pass exactly through the course center (point symmetry of the
    lemniscate) and cross each other at a substantial angle."""
    gs = figure_eight(k, lobe_radius_m=2.0, alt_m=1.0)
    c = np.asarray(gs.centers_world, np.float64)
    d1 = c[k] - c[k - 1]          # last right-lobe gate -> first left-lobe gate
    d2 = c[0] - c[2 * k - 1]      # wrap: last left-lobe gate -> first right-lobe gate
    for p, d in ((c[k - 1], d1), (c[2 * k - 1], d2)):
        tstar = -np.dot(p[:2], d[:2]) / np.dot(d[:2], d[:2])
        assert 0.0 < tstar < 1.0                                   # crossover inside the chord
        assert np.linalg.norm(p[:2] + tstar * d[:2]) < 1e-4        # ... exactly at the center
        assert abs(d[2]) < 1e-6                                    # flat course
    u1 = d1[:2] / np.linalg.norm(d1[:2])
    u2 = d2[:2] / np.linalg.norm(d2[:2])
    assert abs(u1[0] * u2[1] - u1[1] * u2[0]) > 0.5  # crossed diagonals, not (anti)parallel


def test_figure_eight_closed_loop_fly_through():
    """Fly the closed gate-to-gate polyline: every gate passes forward exactly once, no
    backward passes, no frame hits — the loop closes and gates face the travel
    direction."""
    k = 3
    gs = figure_eight(k, lobe_radius_m=2.0, alt_m=1.5)
    centers = np.asarray(gs.centers_world, np.float64)
    g = len(gs)

    steps = 32
    pts = []
    for i in range(g):
        p0, p1 = centers[i], centers[(i + 1) % g]
        for j in range(steps):
            f = (j + 0.5) / steps  # half-slot offset: no sample lands on a gate plane
            pts.append(p0 * (1.0 - f) + p1 * f)
    traj = jnp.asarray(np.stack(pts), jnp.float32)          # [T, 3] closed polyline
    nxt = jnp.roll(traj, -1, axis=0)

    fwd, bwd, hit = classify_crossings(traj, nxt, gs)       # [T, G]
    assert np.array_equal(np.asarray(fwd.sum(axis=0)), np.ones(g, np.int32))
    assert int(bwd.sum()) == 0
    assert int(hit.sum()) == 0


# -- mask noise --------------------------------------------------------------------------


def test_noise_output_in_unit_interval(key):
    f, h, w = 4, 64, 64
    kk, km = jax.random.split(key)
    mask = jax.random.uniform(km, (f, h, w), jnp.float32)
    for scale in (0.5, 1.0, 2.0):
        out = np.asarray(corrupt_mask(fresh_noise_keys(kk, f), mask, scale=scale))
        assert out.shape == (f, h, w)
        assert out.min() >= 0.0 and out.max() <= 1.0

    pts = jax.random.uniform(kk, (f, 5, 2), jnp.float32, minval=0.0, maxval=64.0)
    er = np.asarray(erasure_at(fresh_noise_keys(kk, f), pts, h, w))
    assert er.shape == (f, 5)
    assert er.min() >= 0.0 and er.max() <= 1.0


def test_noise_persistence_across_held_frames(key):
    """Artifacts are a pure function of the carried keys: while no ttl has expired the
    keys ride through noise_state_step unchanged and the corruption is bit-identical;
    expired slots resample."""
    f, h, w = 4, 64, 64
    k1, k2, km = jax.random.split(key, 3)
    mask = jax.random.uniform(km, (f, h, w), jnp.float32)

    keys, ttl = noise_state_init(k1, f, hold=8)             # ttl >= 1 everywhere
    keys2, ttl2 = noise_state_step(k2, keys, ttl)           # first step: nothing expires
    assert np.array_equal(np.asarray(keys2), np.asarray(keys))
    assert np.array_equal(np.asarray(ttl2), np.asarray(ttl) - 1)
    a = np.asarray(corrupt_mask(keys, mask))
    b = np.asarray(corrupt_mask(keys2, mask))
    assert np.array_equal(a, b)                             # held keys freeze the artifacts

    keys3, ttl3 = noise_state_step(k2, keys, jnp.zeros_like(ttl))  # force full expiry
    assert not np.array_equal(np.asarray(keys3), np.asarray(keys))
    assert np.all(np.asarray(ttl3) >= 0)


def test_noise_disabled_is_identity(key):
    f, h, w = 3, 64, 64
    kk, km = jax.random.split(key)
    mask = jax.random.uniform(km, (f, h, w), jnp.float32)
    keys = fresh_noise_keys(kk, f)
    assert np.array_equal(np.asarray(corrupt_mask(keys, mask, scale=0.0)), np.asarray(mask))
    pts = jnp.zeros((f, 4, 2), jnp.float32)
    assert np.all(np.asarray(erasure_at(keys, pts, h, w, scale=0.0)) == 0.0)


def test_blob_point_sampler_matches_grid_render(key):
    """_blob_field_at re-implements jax.image.resize's half-pixel-centre mapping; pin the
    point sampler to the grid render so the two definitions cannot drift."""
    h = w = 64
    grid = np.asarray(_blob_field1(key, h, w, cells=12))
    xs = jnp.arange(w, dtype=jnp.float32)[None, :]
    ys = jnp.arange(h, dtype=jnp.float32)[:, None]
    pts = np.asarray(_blob_field_at(key, h, w, 12, xs, ys))
    assert np.allclose(pts, grid, atol=1e-5)
