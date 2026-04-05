import numpy as np


def build_point_baseline_intensity(harmonics, theta_grid, phi_grid):
    """
    Standalone structured point-source baseline.

    This is not your final Haus production kernel. It is a compact benchmark
    baseline that provides:
    - nontrivial theta structure
    - mild phi asymmetry
    - harmonic variation

    Returns
    -------
    cube : ndarray, shape (Nh, Ntheta, Nphi)
    """
    theta_obs, phi_obs = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    cube = []

    for n in harmonics:
        base = np.sin(theta_obs) ** 2
        mod_theta = 1.0 + 0.30 * np.cos((n - 1) * theta_obs) ** 2
        mod_phi = 1.0 + 0.10 * np.cos(phi_obs) * np.exp(-0.15 * (n - 1))
        cube.append(base * mod_theta * mod_phi)

    return np.array(cube, dtype=float)