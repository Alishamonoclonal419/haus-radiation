import numpy as np
from source_spectrum import compute_Jz_kw


def kz_on_vacuum_manifold(omega_grid: np.ndarray, theta: float, c: float = 1.0) -> np.ndarray:
    """
    For observation polar angle theta, compute k_z on the vacuum propagating manifold:

        k_z = (omega / c) cos(theta)
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    return (omega_grid / c) * np.cos(theta)


def structural_vacuum_intensity_for_z_motion(
    t: np.ndarray,
    z: np.ndarray,
    vz: np.ndarray,
    omega_grid: np.ndarray,
    theta_grid: np.ndarray,
    q: float = 1.0,
    c: float = 1.0,
    window: str = "hann",
    normalize: bool = True,
) -> np.ndarray:
    """
    Earlier structural vacuum radiation proxy for z-directed motion:

        I_struct(omega, theta) ∝ sin^2(theta) * |J_z(kz_rad, omega)|^2
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)

    I = np.zeros((len(theta_grid), len(omega_grid)), dtype=float)

    for i, theta in enumerate(theta_grid):
        kz_grid = kz_on_vacuum_manifold(omega_grid, theta=theta, c=c)

        J_matrix = compute_Jz_kw(
            t=t,
            z=z,
            vz=vz,
            kz_grid=kz_grid,
            omega_grid=omega_grid,
            q=q,
            window=window,
            normalize=normalize,
        )

        J_diag = np.diag(J_matrix)
        I[i, :] = (np.sin(theta) ** 2) * (np.abs(J_diag) ** 2)

    return I


def spectral_far_field_intensity_for_z_motion(
    t: np.ndarray,
    z: np.ndarray,
    vz: np.ndarray,
    omega_grid: np.ndarray,
    theta_grid: np.ndarray,
    q: float = 1.0,
    c: float = 1.0,
    window: str = "hann",
    normalize: bool = True,
) -> np.ndarray:
    """
    Upgraded vacuum far-field spectral observable for z-directed motion:

        I_spec(omega, theta) ∝ omega^2 sin^2(theta) |J_z(kz_rad, omega)|^2

    This is a more physically meaningful radiation-level quantity than the
    earlier structural proxy because it incorporates the far-field omega-scaling.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)

    I = np.zeros((len(theta_grid), len(omega_grid)), dtype=float)

    for i, theta in enumerate(theta_grid):
        kz_grid = kz_on_vacuum_manifold(omega_grid, theta=theta, c=c)

        J_matrix = compute_Jz_kw(
            t=t,
            z=z,
            vz=vz,
            kz_grid=kz_grid,
            omega_grid=omega_grid,
            q=q,
            window=window,
            normalize=normalize,
        )

        J_diag = np.diag(J_matrix)
        I[i, :] = (omega_grid ** 2) * (np.sin(theta) ** 2) * (np.abs(J_diag) ** 2)

    return I


def apply_dc_filter(
    omega_grid: np.ndarray,
    intensity: np.ndarray,
    omega_cut: float,
) -> np.ndarray:
    """
    Remove a symmetric band around omega = 0.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    filtered = np.array(intensity, copy=True)

    mask = np.abs(omega_grid) < omega_cut
    filtered[..., mask] = 0.0
    return filtered


def max_nonzero_frequency_intensity(
    omega_grid: np.ndarray,
    intensity: np.ndarray,
    omega_cut: float,
) -> float:
    """
    Return the maximum intensity outside the DC exclusion band.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    mask = np.abs(omega_grid) >= omega_cut
    if not np.any(mask):
        return 0.0
    return float(np.max(intensity[..., mask]))