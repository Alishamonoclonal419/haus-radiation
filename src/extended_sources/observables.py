import numpy as np


def ratio_safe(num, den, eps=1e-30):
    return float(num / max(den, eps))


def tradeoff_q(rdet, rff):
    return ratio_safe(rdet, rff)


def detector_score(intensity_cube, theta_grid, theta_min, theta_max, phi_index=0):
    """
    intensity_cube shape: (Nh, Ntheta, Nphi)
    """
    mask = (theta_grid >= theta_min) & (theta_grid <= theta_max)
    dtheta = float(theta_grid[1] - theta_grid[0])

    score = 0.0
    for n in range(intensity_cube.shape[0]):
        score += np.sum(intensity_cube[n, mask, phi_index]) * dtheta
    return float(score)


def full_3d_score(intensity_cube, theta_grid, phi_grid):
    """
    S_ff^3D = sum_n ∫∫ I_n(theta,phi) sin(theta) dtheta dphi
    """
    dtheta = float(theta_grid[1] - theta_grid[0])
    dphi = float(phi_grid[1] - phi_grid[0])
    sin_theta = np.sin(theta_grid)[:, None]

    score = 0.0
    for n in range(intensity_cube.shape[0]):
        score += np.sum(intensity_cube[n] * sin_theta) * dtheta * dphi
    return float(score)


def angular_profile_fixed_phi(intensity_cube, phi_index):
    """
    Sum over harmonics, return profile vs theta at a fixed phi index.
    """
    return np.sum(intensity_cube[:, :, phi_index], axis=0)