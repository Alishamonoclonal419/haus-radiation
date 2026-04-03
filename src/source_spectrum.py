import numpy as np
from windows import get_window


def compute_Jz_kw(
    t: np.ndarray,
    z: np.ndarray,
    vz: np.ndarray,
    kz_grid: np.ndarray,
    omega_grid: np.ndarray,
    q: float = 1.0,
    window: str = "rectangular",
    normalize: bool = True,
) -> np.ndarray:
    """
    Memory-safe computation of the 1D Fourier-domain source current

        J_z(k_z, omega) = q ∫ dt v(t) exp[-i k_z z(t)] exp[i omega t]

    The computation is done one k_z at a time, so we avoid building a huge
    (Nk, Nw, Nt) tensor in memory.

    Parameters
    ----------
    t : np.ndarray
        Time array, shape (Nt,).
    z : np.ndarray
        Position z(t), shape (Nt,).
    vz : np.ndarray
        Velocity v(t), shape (Nt,).
    kz_grid : np.ndarray
        k_z grid, shape (Nk,).
    omega_grid : np.ndarray
        omega grid, shape (Nw,).
    q : float
        Charge prefactor.
    window : str
        Window type: 'rectangular' or 'hann'.
    normalize : bool
        If True, divide by ∫ dt w(t).

    Returns
    -------
    J : np.ndarray
        Complex array of shape (Nk, Nw).
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    vz = np.asarray(vz, dtype=float)
    kz_grid = np.asarray(kz_grid, dtype=float)
    omega_grid = np.asarray(omega_grid, dtype=float)

    if t.ndim != 1 or z.ndim != 1 or vz.ndim != 1:
        raise ValueError("t, z, vz must all be 1D arrays.")
    if len(t) != len(z) or len(t) != len(vz):
        raise ValueError("t, z, vz must have the same length.")
    if len(t) < 2:
        raise ValueError("Need at least two time points.")

    w = get_window(t, kind=window)
    weighted_v = q * w * vz

    if normalize:
        norm = np.trapezoid(w, t)
        if norm == 0.0:
            raise ValueError("Window normalization is zero.")
    else:
        norm = 1.0

    # Precompute exp(i omega t) once: shape (Nw, Nt)
    exp_omega_t = np.exp(1j * omega_grid[:, None] * t[None, :])

    J = np.empty((len(kz_grid), len(omega_grid)), dtype=np.complex128)

    for i, kz in enumerate(kz_grid):
        exp_kz_z = np.exp(-1j * kz * z)              # shape (Nt,)
        integrand = exp_omega_t * (weighted_v * exp_kz_z)[None, :]
        J[i, :] = np.trapezoid(integrand, t, axis=1) / norm

    return J


def expected_constant_velocity_center(kz: np.ndarray, v: float) -> np.ndarray:
    """
    Expected ridge location for constant-velocity motion:
        omega = v * kz
    """
    kz = np.asarray(kz, dtype=float)
    return v * kz