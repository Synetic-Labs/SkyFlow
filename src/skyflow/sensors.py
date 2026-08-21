"""
IMU packaging (DESIGN.md §2): the generated exact measurement plus the corruption hooks
the spec charter keeps harness-side. The spec supplies noiseless specific force and body
rate through dynamics.imu; additive Gaussian noise (and any future bias/scale/staleness
DR) lives here, never in the ODE.
"""

import jax

from skyflow import dynamics


def measure(
    plant,
    omega_cmd,
    wind_vel,
    params,
    *,
    key=None,
    accel_noise_std: float = 0.0,
    gyro_noise_std: float = 0.0,
    imu_bias=None,
):
    """
    IMU rows for the fleet → (accel [F,3], gyro [F,3]).

    accel is specific force in body FLU, m/s² — (0, 0, +g) at exact hover; gyro is body
    rate, rad/s (dynamics.imu: body origin, identity mount). Two independent corruption
    hooks (DomainRand, DESIGN.md §7): `imu_bias` [F,6] (accel(3) m/s², gyro(3) rad/s)
    adds the per-episode constant trait; with `key` given, zero-mean isotropic Gaussian
    noise with the per-sensor standard deviations adds the per-sample process. key=None
    and imu_bias=None return the exact measurement, keeping the noiseless path key-free.
    Corruption inherits the measurement dtype, so env-created float32 stays float32.
    """
    accel, gyro = dynamics.imu(plant, omega_cmd, wind_vel, params)
    if imu_bias is not None:
        accel = accel + imu_bias[:, 0:3].astype(accel.dtype)
        gyro = gyro + imu_bias[:, 3:6].astype(gyro.dtype)
    if key is None:
        return accel, gyro
    k_accel, k_gyro = jax.random.split(key)
    accel = accel + accel_noise_std * jax.random.normal(k_accel, accel.shape, accel.dtype)
    gyro = gyro + gyro_noise_std * jax.random.normal(k_gyro, gyro.shape, gyro.dtype)
    return accel, gyro
