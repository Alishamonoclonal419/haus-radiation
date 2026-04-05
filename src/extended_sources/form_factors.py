import numpy as np


def shell_form_factor_anisotropic(k, theta_obs, phi_obs, theta_src, phi_src, weight_src, b):
    """
    Anisotropic shell form factor:
        F(theta, phi) = (1/N) ∫ W(theta',phi') exp[-ik b khat·rhat'] dΩ'
    """
    sin_ts = np.sin(theta_src)

    xs = sin_ts * np.cos(phi_src)
    ys = sin_ts * np.sin(phi_src)
    zs = np.cos(theta_src)

    xo = np.sin(theta_obs) * np.cos(phi_obs)
    yo = np.sin(theta_obs) * np.sin(phi_obs)
    zo = np.cos(theta_obs)

    dtheta = float(theta_src[1, 0] - theta_src[0, 0])
    dphi = float(phi_src[0, 1] - phi_src[0, 0])

    norm = np.sum(weight_src * sin_ts) * dtheta * dphi

    F = np.zeros_like(theta_obs, dtype=np.complex128)

    for i in range(theta_obs.shape[0]):
        for j in range(theta_obs.shape[1]):
            mu = xo[i, j] * xs + yo[i, j] * ys + zo[i, j] * zs
            phase = np.exp(-1j * k * b * mu)
            F[i, j] = np.sum(weight_src * phase * sin_ts) * dtheta * dphi / norm

    return F


def volume_form_factor_anisotropic(k, theta_obs, phi_obs, r_grid, theta_src, phi_src, weight_ang):
    """
    Anisotropic solid-sphere form factor with angular anisotropy only:
        F(theta, phi) = (1/N) ∫_0^b ∫ W(theta',phi') exp[-ik r khat·rhat'] r^2 dΩ' dr
    """
    sin_ts = np.sin(theta_src)

    xs = sin_ts * np.cos(phi_src)
    ys = sin_ts * np.sin(phi_src)
    zs = np.cos(theta_src)

    xo = np.sin(theta_obs) * np.cos(phi_obs)
    yo = np.sin(theta_obs) * np.sin(phi_obs)
    zo = np.cos(theta_obs)

    dr = float(r_grid[1] - r_grid[0])
    dtheta = float(theta_src[1, 0] - theta_src[0, 0])
    dphi = float(phi_src[0, 1] - phi_src[0, 0])

    norm_ang = np.sum(weight_ang * sin_ts) * dtheta * dphi
    norm_rad = np.sum(r_grid**2) * dr
    norm = norm_ang * norm_rad

    F = np.zeros_like(theta_obs, dtype=np.complex128)

    for i in range(theta_obs.shape[0]):
        for j in range(theta_obs.shape[1]):
            mu = xo[i, j] * xs + yo[i, j] * ys + zo[i, j] * zs

            total = 0.0j
            for r in r_grid:
                phase = np.exp(-1j * k * r * mu)
                ang_part = np.sum(weight_ang * phase * sin_ts) * dtheta * dphi
                total += (r**2) * ang_part * dr

            F[i, j] = total / norm

    return F


def apply_form_factors_to_baseline(point_cube, form_factors):
    """
    point_cube:   (Nh, Ntheta, Nphi)
    form_factors: (Nh, Ntheta, Nphi), complex

    Returns
    -------
    cube : ndarray, shape (Nh, Ntheta, Nphi)
    """
    out = np.empty_like(point_cube)
    for i in range(point_cube.shape[0]):
        out[i] = point_cube[i] * np.abs(form_factors[i]) ** 2
    return out