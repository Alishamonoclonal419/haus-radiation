import numpy as np


def trajectory_constant_velocity(t: np.ndarray, v: float) -> tuple[np.ndarray, np.ndarray]:
    """
    1D constant-velocity motion along z.
    """
    z = v * t
    vz = np.full_like(t, v, dtype=float)
    return z, vz


def trajectory_sinusoidal(t: np.ndarray, d: float, omega0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    1D sinusoidal motion along z:
        z(t) = d sin(omega0 t)
        v(t) = d omega0 cos(omega0 t)
    """
    z = d * np.sin(omega0 * t)
    vz = d * omega0 * np.cos(omega0 * t)
    return z, vz