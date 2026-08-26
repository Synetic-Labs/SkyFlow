"""
Betaflight-in-the-loop seam for control="sticks" (DESIGN.md §10).

SkyFlow's public frames are world z-up / body FLU, but Betaflight wants NED/FRD sensors;
this module is one of the two places NED/FRD is allowed to exist (DESIGN.md §3). The
boundary layout is fixed by the cudaflight wheel:

    sensors [F, 7] f32 = gyro_FRD rad/s (3), specific force FRD m/s² (3), baro Pa (1)
                         (level hover ⇒ az = -9.81)
    sticks  [F, 4] f32 = AETR (roll, pitch, throttle, yaw) in [-1, 1]
    motors  [F, 4] f32 in [0, 1], QUADX order;  armed [F] u8

Both fleet classes implement `types.FirmwareFleet`: the firmware state is value-threaded
as a (blob, fwstate) pair so `fw_step` composes into the env's `lax.scan` as a pure-shaped
call. `CpuFirmwareFleet` (libcpuflight.so, ctypes) is complete and self-contained — any
fleet size, no CUDA, jits via ordered `io_callback`, but is NOT vmappable or replayable
because the real firmware state mutates host-side and the threaded pair is a zero-length
placeholder. `GpuFirmwareFleet` (libcudaflight.so, in-jit XLA FFI, genuinely donated
buffers) needs the `cudaflight.xla` FFI half (added in 0.3.3 — but the PACKAGED floor
is the pyproject `firmware` extra, cudaflight >= 0.6.0; per-feature version notes below
are history, not install targets). Any
external fleet that satisfies the protocol can be injected via
`SkyFlowEnv(cfg, firmware_fleet=...)`.

The two sensor helpers below are the ONLY sensor packaging DESIGN.md §10 authorizes in
this module: a frame flip and an isothermal barometer. Anything that models physics
(the IMU itself) comes from the generated backend through sensors.py.

cudaflight imports stay inside the constructors so `import skyflow.firmware` (and motors
mode, which never touches this module) works without the wheel installed.
"""

import ctypes
import os
import warnings
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.typing import ArrayLike

__all__ = ["GPU_FLEET_MIN", "CpuFirmwareFleet", "GpuFirmwareFleet", "baro_pa", "flu_to_frd"]

#: The GPU backend's minimum fleet: the runtime relocation table cannot be
#: discovered below 3 instances. THE definition — env's auto-selection and the
#: constructor guard both read it (smaller fleets belong to CpuFirmwareFleet).
GPU_FLEET_MIN = 3

_INSTALL_GUIDANCE = (
    'control="sticks" needs the cudaflight wheel (Betaflight SITL fleets). Install the '
    "firmware extra — `uv sync --extra firmware` / `pip install skyflow[firmware]` — or "
    "add the cudaflight wheel to the environment directly."
)

# Isothermal barometer constants for the firmware sensor boundary:
# P = 101325 Pa at sea level, scale height 8434 m.
_SEA_LEVEL_PA = 101325.0
_BARO_SCALE_M = 8434.0

# libcpuflight keeps ONE global SITL fleet per process ("render before create or
# after destroy"); a second live fleet corrupts the first. The reference is weak
# so the guard never keeps a dropped fleet alive past its __del__.
_LIVE_CPU_FLEET: "weakref.ref[CpuFirmwareFleet] | None" = None


def flu_to_frd(v: ArrayLike) -> jax.Array:
    """
    Flip [..., 3] vectors between FLU and FRD (equally z-up world and NED): (x, -y, -z).

    Harness-side sensor PACKAGING for the firmware boundary, not spec physics — the frames
    share the x axis and negate the other two, so the map is elementwise, dtype-preserving,
    and its own inverse. Used to hand body-FLU gyro/specific-force rows from the generated
    IMU to Betaflight as FRD. THE implementation is vision._ned.flip_xyz — this wrapper
    only names the boundary (DESIGN.md §10 authorizes the flip at exactly two homes:
    vision internals and here).
    """
    from skyflow.vision._ned import flip_xyz

    return flip_xyz(jnp.asarray(v))


def baro_pa(alt_m: ArrayLike) -> jax.Array:
    """
    Barometric pressure Pa from z-up altitude m: 101325·exp(-alt/8434).

    Harness-side sensor MODEL for the firmware boundary, not spec physics — an isothermal
    atmosphere, good to ~0.1% over indoor flight
    envelopes; Betaflight only differentiates it for altitude hold. Sea level ⇒ 101325 Pa.
    DESIGN.md §10 authorizes exactly this barometer here; a physical atmosphere model
    would go through the SkyFlow-Dynamics INTAKE protocol instead.
    """
    alt_m = jnp.asarray(alt_m)
    return _SEA_LEVEL_PA * jnp.exp(-alt_m / _BARO_SCALE_M)


class CpuFirmwareFleet:
    """
    Betaflight CPU SITL fleet (libcpuflight.so) — the self-contained sticks backend.

    Instances step sequentially in-process through ctypes, so any fleet size works (the
    GPU fleet refuses n < 3) and no CUDA is needed; at small interactive sizes this
    outruns a GPU dispatch. Every `fw_step`/`reset` crosses to the host through
    `io_callback(ordered=True)`, so the env's `lax.scan` jits unchanged — but the REAL
    firmware state lives in the ctypes handle and mutates in place. The threaded
    (blob, fwstate) pair is a zero-length uint8 placeholder carried only for
    `types.FirmwareFleet` parity: this fleet is NOT vmappable and NOT replayable
    (re-running a traced step re-mutates the one real fleet).

    Construction boots `fleet` firmware instances, lets them settle `settle_ms` virtual
    milliseconds, arms, and snapshots — `fresh_firmware_state()` and `reset()` restore
    that armed-on-ground snapshot. `eeprom` is a boot-ready Betaflight config image
    (None boots stock defaults); `lib` overrides the wheel's packaged libcpuflight.so
    path.

    Never pass a committed `.bin` as `eeprom`: an image one parameter-group version
    behind the wheel's firmware makes Betaflight factory-reset the whole config at
    boot, silently. Render the image from CLI dump text at use time with
    `cudaflight.render_eeprom()` (cudaflight >= 0.4.0) — it fails loudly on any
    setting the firmware rejects. See examples/configs/ for the pattern.
    """

    def __init__(
        self,
        fleet: int,
        *,
        settle_ms: int = 0,
        eeprom: str | os.PathLike[str] | None = None,
        lib: str | os.PathLike[str] | None = None,
    ) -> None:
        try:
            from cudaflight.lib import load_cpu
        except ImportError as e:
            raise ImportError(_INSTALL_GUIDANCE) from e

        global _LIVE_CPU_FLEET
        live = _LIVE_CPU_FLEET() if _LIVE_CPU_FLEET is not None else None
        if live is not None and live._h:
            raise RuntimeError(
                "a live CpuFirmwareFleet already exists in this process — libcpuflight "
                "keeps one global SITL fleet, and a second create corrupts the first. "
                "close() the existing fleet (or its SkyFlowEnv) before constructing "
                "another."
            )
        self.fleet = int(fleet)
        self._lib = load_cpu(lib)
        eeprom_arg = str(eeprom).encode() if eeprom else None
        self._h = self._lib.cpuflight_create_eeprom(self.fleet, settle_ms, eeprom_arg)
        if not self._h:
            raise RuntimeError(f"cpuflight_create failed: {self._lib.cpuflight_error().decode()}")
        self.act_dim = int(self._lib.cpuflight_act_dim(self._h))
        _LIVE_CPU_FLEET = weakref.ref(self)

    def _require_open(self) -> None:
        """Raise on a closed fleet. Called at the public entry points (clean eager /
        trace-time errors) AND inside the host halves below, so it also fires on
        re-runs of programs compiled before close() — where a raw call would hand
        ctypes a destroyed handle and segfault. A host-half raise poisons jax's
        ordered-callback token for the process; that path is already fatal misuse,
        and an error beats a segfault."""
        if not self._h:
            raise RuntimeError(
                "CpuFirmwareFleet is closed — its SITL instances are destroyed; "
                "construct a new fleet"
            )

    # -- host sides of the ordered io_callbacks --------------------------------------

    def _host_step(self, sticks: np.ndarray, sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One 1 kHz tick for the whole fleet through ctypes → (motors [F,4], armed [F])."""
        self._require_open()
        sticks = np.ascontiguousarray(sticks, np.float32)
        sensors = np.ascontiguousarray(sensors, np.float32)
        motors = np.empty((self.fleet, 4), np.float32)
        armed = np.empty((self.fleet,), np.uint8)
        rc = self._lib.cpuflight_fw_step(
            self._h,
            sticks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sensors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            motors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            armed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            1,
        )
        if rc != 0:  # 0 on success — fail loudly rather than fly on stale output buffers
            raise RuntimeError(
                f"cpuflight_fw_step failed ({rc}): {self._lib.cpuflight_error().decode()}"
            )
        return motors, armed

    def _host_reset(self, mask: np.ndarray) -> np.ndarray:
        """Restore the flagged instances (uint8 [F]) to the armed snapshot."""
        self._require_open()
        mask = np.ascontiguousarray(mask, np.uint8)
        rc = self._lib.cpuflight_reset_mask(
            self._h, mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        )
        if rc != 0:
            raise RuntimeError(
                f"cpuflight_reset_mask failed ({rc}): {self._lib.cpuflight_error().decode()}"
            )
        return np.zeros((0,), np.uint8)

    def _host_reset_all(self) -> np.ndarray:
        """Restore ALL instances to the armed snapshot."""
        self._require_open()
        self._lib.cpuflight_reset_all(self._h)
        return np.zeros((0,), np.uint8)

    # -- types.FirmwareFleet ----------------------------------------------------------

    def fresh_firmware_state(self) -> tuple[jax.Array, jax.Array]:
        """Restore ALL instances to the armed snapshot; placeholder (blob, fwstate)."""
        self._require_open()
        blob = io_callback(
            self._host_reset_all, jax.ShapeDtypeStruct((0,), jnp.uint8), ordered=True
        )
        return blob, jnp.zeros((0,), jnp.uint8)

    def fw_step(
        self, blob: jax.Array, fwstate: jax.Array, sticks: jax.Array, sensors: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """One 1 kHz firmware tick → (blob, fwstate, motors [F,4] in [0,1], armed u8 [F])."""
        self._require_open()
        motors, armed = io_callback(
            self._host_step,
            (
                jax.ShapeDtypeStruct((self.fleet, 4), jnp.float32),
                jax.ShapeDtypeStruct((self.fleet,), jnp.uint8),
            ),
            sticks,
            sensors,
            ordered=True,
        )
        return blob, fwstate, motors, armed

    def reset(
        self, blob: jax.Array, fwstate: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Restore the worlds selected by mask (u8 [F]) to the armed snapshot."""
        self._require_open()
        blob = io_callback(
            self._host_reset, jax.ShapeDtypeStruct((0,), jnp.uint8), mask, ordered=True
        )
        return blob, fwstate

    def close(self) -> None:
        """Destroy the SITL instances; the fleet is unusable afterwards."""
        if getattr(self, "_h", None):
            self._lib.cpuflight_destroy(self._h)
            self._h = None

    def __del__(self) -> None:  # best-effort teardown (interpreter-shutdown safe)
        try:
            self.close()
        except Exception:
            pass


class GpuFirmwareFleet:
    """
    Betaflight CUDA fleet (libcudaflight.so) — the in-jit sticks backend.

    Requires cudaflight >= 0.3.3: the wheel ships the XLA FFI half (`cudaflight.xla`,
    prebuilt handlers + source fallback), so the firmware kernels launch directly on
    XLA's compute stream inside the jitted program — zero host round trips per tick.
    This backend implements the construction sequence DESIGN.md §10 fixes:

    - `XLA_PYTHON_CLIENT_PREALLOCATE=false` must be set BEFORE jax touches the GPU, or
      XLA's arena leaves no VRAM for the firmware instance arrays. Export it in the
      LAUNCHER (the library never mutates the environment).
    - XLA claims the primary CUDA context first: the constructor touches the target
      device (`jnp.zeros(1, device=...)` + block_until_ready) before `cudaflight_create*`,
      so the firmware kernels share XLA's context instead of racing it for one.
    - Creates via `cudaflight_create_eeprom_ex(cubin, fleet, device_index, settle_ms,
      eeprom, with_grad=0)` — no differentiable-rollout scratch, ~1.5x more worlds —
      falling back to `cudaflight_create_eeprom` on older libraries.
    - `fleet >= 3`: the runtime relocation table cannot be discovered below 3
      instances; smaller fleets belong to CpuFirmwareFleet.
    - Firmware state is GENUINELY value-threaded, unlike the CPU placeholders: blob
      [F·stride] u8 and fwstate [F·state_size] u8 ride through `jax.ffi.ffi_call`
      with input_output_aliases (donated in place), copied at construction from the
      armed-on-ground snapshot buffers; `fresh_firmware_state`/`reset` restore from
      that snapshot, so this fleet IS replayable, unlike the CPU one.
    - bfSetBase runs before every step/reset launch: the handler points the global
      instance base at wherever XLA placed the donated blob and the kernel rebases
      on entry. One handle = one device = `fleet` worlds.
    - `eeprom` is a boot-ready config image (None boots stock defaults). Never pass
      a committed `.bin`: one parameter-group version of drift makes the firmware
      factory-reset the whole config at boot, silently. Render the image from CLI
      dump text with `cudaflight.render_eeprom()` (cudaflight >= 0.4.0); it fails
      loudly on rejected settings. See examples/configs/.
    """

    def __init__(
        self,
        fleet: int,
        *,
        device_index: int = 0,
        settle_ms: int = 0,
        eeprom: str | os.PathLike[str] | None = None,
        cubin: str | os.PathLike[str] | None = None,
        lib: str | os.PathLike[str] | None = None,
    ) -> None:
        if int(fleet) < GPU_FLEET_MIN:
            raise ValueError(
                f"GpuFirmwareFleet needs fleet >= {GPU_FLEET_MIN} (the runtime "
                f"relocation table cannot be discovered below that), got {fleet}; "
                f"use CpuFirmwareFleet for smaller fleets"
            )
        # The instance arrays share the device with XLA's arena. With default
        # preallocation XLA grabs ~90% of VRAM first and cudaflight_create OOMs —
        # historically that OOM was swallowed into a silent CPU fallback. Warn
        # BEFORE creating so the failure names its cause.
        prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower()
        if prealloc not in ("false", "0") and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
            warnings.warn(
                "XLA_PYTHON_CLIENT_PREALLOCATE is not 'false' and no MEM_FRACTION is "
                "set: XLA's arena will claim most of the VRAM before the firmware "
                "instance arrays allocate. Export XLA_PYTHON_CLIENT_PREALLOCATE=false "
                "before jax touches the GPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            from cudaflight import xla as _cfx
            from cudaflight.lib import default_fatbin_path, load
        except ImportError as e:
            raise ImportError(
                _INSTALL_GUIDANCE + " GpuFirmwareFleet additionally needs the "
                "cudaflight.xla FFI half (any wheel at the pyproject floor, "
                "cudaflight >= 0.6.0, ships it)."
            ) from e

        self.fleet = int(fleet)
        self.device = jax.devices("gpu")[device_index]
        # XLA claims the primary CUDA context before the firmware library does.
        jnp.zeros(1, device=self.device).block_until_ready()

        self._lib = load(lib)
        cubin_arg = str(cubin or default_fatbin_path())
        eeprom_arg = str(eeprom).encode() if eeprom else None
        # Non-differentiable create — no per-instance gradient scratch (~1.5x more
        # worlds); the differentiable rollout is out of scope here (§7: planned).
        if hasattr(self._lib, "cudaflight_create_eeprom_ex"):
            self._h = self._lib.cudaflight_create_eeprom_ex(
                cubin_arg.encode(), self.fleet, device_index, settle_ms, eeprom_arg, 0
            )
        else:
            self._h = self._lib.cudaflight_create_eeprom(
                cubin_arg.encode(), self.fleet, device_index, settle_ms, eeprom_arg
            )
        if not self._h:
            raise RuntimeError(
                f"cudaflight_create failed: {self._lib.cudaflight_error().decode()}"
            )

        self.act_dim = int(self._lib.cudaflight_act_dim(self._h))
        self._fw_pure = _cfx.fw_step_pure_call(self._lib, self._h)
        # the armed-on-ground episode-start snapshot, as fresh JAX buffers
        self._snap_blob, self._snap_state = _cfx.snapshot_state(
            self._lib, self._h, device_index
        )
        # cudaflight >= 0.3.4: the snapshot rides into the reset call as read-only
        # JAX buffer arguments, and the library-side copies are freed — every
        # firmware datum then lives in XLA buffers. Older wheels fall back to the
        # library-side snapshot pointers (baked device addresses).
        self._snapshot_args = bool(
            getattr(_cfx, "SUPPORTS_SNAPSHOT_ARGS", False)
            and hasattr(self._lib, "cudaflight_release_snapshots")
        )
        if self._snapshot_args:
            self._reset_pure = _cfx.reset_pure_call(
                self._lib, self._h, snapshot=(self._snap_blob, self._snap_state)
            )
            self._lib.cudaflight_release_snapshots(self._h)
        else:
            self._reset_pure = _cfx.reset_pure_call(self._lib, self._h)

    def _require_open(self) -> None:
        """Raise on a closed fleet. This fires at trace time and on eager calls; a
        program COMPILED before close() bypasses Python entirely and would launch
        the FFI kernels on the destroyed handle — never re-run one after close()."""
        if not self._h:
            raise RuntimeError(
                "GpuFirmwareFleet is closed — its device instances are destroyed; "
                "construct a new fleet (and never re-run programs compiled before "
                "close())"
            )

    # -- types.FirmwareFleet ----------------------------------------------------------

    def fresh_firmware_state(self) -> tuple[jax.Array, jax.Array]:
        """A fresh (blob, fwstate) pair copied from the armed snapshot."""
        self._require_open()
        return jnp.copy(self._snap_blob), jnp.copy(self._snap_state)

    def fw_step(
        self, blob: jax.Array, fwstate: jax.Array, sticks: jax.Array, sensors: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """One 1 kHz firmware tick → (blob, fwstate, motors [F,4] in [0,1], armed u8 [F])."""
        self._require_open()
        return self._fw_pure(blob, fwstate, sticks, sensors)

    def reset(
        self, blob: jax.Array, fwstate: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Restore the worlds selected by mask (u8 [F]) to the armed snapshot."""
        self._require_open()
        return self._reset_pure(blob, fwstate, mask)

    def close(self) -> None:
        """Destroy the firmware instances; the fleet is unusable afterwards."""
        if getattr(self, "_h", None):
            self._lib.cudaflight_destroy(self._h)
            self._h = None

    def __del__(self) -> None:  # best-effort teardown (interpreter-shutdown safe)
        try:
            self.close()
        except Exception:
            pass
