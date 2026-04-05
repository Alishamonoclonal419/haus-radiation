#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ============================================================
# Task 13B-shell-fast: anisotropic shell-only benchmark
# ============================================================

# ----------------------------
# Configuration
# ----------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Units
C = 1.0
W0 = 1.0
T0 = 2.0 * np.pi / W0

# Reduced harmonic set
N_MAX = 6
HARMONICS = jnp.arange(1, N_MAX + 1, dtype=jnp.float32)

# Reduced observation grid
N_THETA = 41
N_PHI = 37
theta_grid = jnp.linspace(0.0, jnp.pi, N_THETA)
phi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N_PHI)
theta_obs, phi_obs = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")

# Detector window
THETA_MIN = 0.55
THETA_MAX = 0.80
PHI_DET = 0.0
phi_det_index = int(np.argmin(np.abs(np.array(phi_grid) - PHI_DET)))

# Reduced source integration grid
N_THETA_SRC = 31
N_PHI_SRC = 31
theta_src = jnp.linspace(0.0, jnp.pi, N_THETA_SRC)
phi_src = jnp.linspace(0.0, 2.0 * jnp.pi, N_PHI_SRC)
theta_src_m, phi_src_m = jnp.meshgrid(theta_src, phi_src, indexing="ij")

# Small first-pass scan
chi_shell = jnp.array([0.46, 0.50, 0.54], dtype=jnp.float32)
betas = jnp.array([-0.8, 0.0, 0.8], dtype=jnp.float32)

# Robustness neighborhood
ROBUST_DELTA_CHI = 0.02


# ----------------------------
# Helper math
# ----------------------------
def legendre_p2(x):
    return 0.5 * (3.0 * x**2 - 1.0)


def weight_p2(theta, beta):
    return 1.0 + beta * legendre_p2(jnp.cos(theta))


def validate_nonnegative(weight, tol=1e-12):
    return bool(jnp.min(weight) >= -tol)


def ratio_safe(num, den, eps=1e-30):
    return float(num / max(den, eps))


def tradeoff_q(rdet, rff):
    return ratio_safe(rdet, rff)


# ----------------------------
# Baseline point-source model
# ----------------------------
@jax.jit
def build_point_baseline_intensity(harmonics, theta_grid, phi_grid):
    theta_obs, phi_obs = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")

    def one_harmonic(n):
        base = jnp.sin(theta_obs) ** 2
        mod_theta = 1.0 + 0.30 * jnp.cos((n - 1.0) * theta_obs) ** 2
        mod_phi = 1.0 + 0.10 * jnp.cos(phi_obs) * jnp.exp(-0.15 * (n - 1.0))
        return base * mod_theta * mod_phi

    return jax.vmap(one_harmonic)(harmonics)


# ----------------------------
# Precompute geometry
# ----------------------------
sin_ts = jnp.sin(theta_src_m)

xs = sin_ts * jnp.cos(phi_src_m)
ys = sin_ts * jnp.sin(phi_src_m)
zs = jnp.cos(theta_src_m)

xo = jnp.sin(theta_obs) * jnp.cos(phi_obs)
yo = jnp.sin(theta_obs) * jnp.sin(phi_obs)
zo = jnp.cos(theta_obs)

MU = (
    xo[:, :, None, None] * xs[None, None, :, :]
    + yo[:, :, None, None] * ys[None, None, :, :]
    + zo[:, :, None, None] * zs[None, None, :, :]
)

dtheta_src = float(theta_src[1] - theta_src[0])
dphi_src = float(phi_src[1] - phi_src[0])

dtheta_obs = float(theta_grid[1] - theta_grid[0])
dphi_obs = float(phi_grid[1] - phi_grid[0])


# ----------------------------
# Shell form factor
# ----------------------------
@jax.jit
def shell_form_factor_anisotropic_single(k, weight_src, b):
    norm = jnp.sum(weight_src * sin_ts) * dtheta_src * dphi_src
    phase = jnp.exp(-1j * k * b * MU)
    integ = jnp.sum(weight_src[None, None, :, :] * phase * sin_ts[None, None, :, :], axis=(2, 3))
    return integ * dtheta_src * dphi_src / norm


@jax.jit
def apply_form_factors_to_baseline(point_cube, form_factors):
    return point_cube * jnp.abs(form_factors) ** 2


# ----------------------------
# Observables
# ----------------------------
@jax.jit
def detector_score(intensity_cube, theta_grid, theta_min, theta_max, phi_index):
    mask = ((theta_grid >= theta_min) & (theta_grid <= theta_max)).astype(jnp.float32)
    selected = intensity_cube[:, :, phi_index]   # (Nh, Ntheta)
    return jnp.sum(selected * mask[None, :]) * dtheta_obs


@jax.jit
def full_3d_score(intensity_cube, theta_grid):
    sin_theta = jnp.sin(theta_grid).astype(jnp.float32)[:, None]
    return jnp.sum(intensity_cube * sin_theta[None, :, :]) * dtheta_obs * dphi_obs


def angular_profile_fixed_phi(intensity_cube, phi_index):
    return np.array(jnp.sum(intensity_cube[:, :, phi_index], axis=0))


# ----------------------------
# Plotting
# ----------------------------
def save_heatmap(arr, xvals, yvals, title, cbar_label, fname):
    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        np.array(arr),
        origin="lower",
        aspect="auto",
        extent=[float(xvals[0]), float(xvals[-1]), float(yvals[0]), float(yvals[-1])],
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\chi$")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_profile(theta_grid_np, point_profile, ext_profile, title, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid_np, point_profile, label="point baseline")
    plt.plot(theta_grid_np, ext_profile, label="best anisotropic shell")
    plt.axvspan(THETA_MIN, THETA_MAX, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_robustness_plot(chis, rdet_vals, rff_vals, title, fname):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(chis, rdet_vals, marker="o", label=r"$R_{\det}$")
    plt.plot(chis, rff_vals, marker="s", label=r"$R_{\mathrm{ff}}^{3D}$")
    plt.xlabel(r"$\chi$")
    plt.ylabel("Ratio to point baseline")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


# ----------------------------
# Family runner
# ----------------------------
def run_shell_family(point_cube, Sdet_point, Sff_point):
    Rdet = np.full((len(chi_shell), len(betas)), np.nan)
    Rff = np.full((len(chi_shell), len(betas)), np.nan)
    Q = np.full((len(chi_shell), len(betas)), np.nan)

    best = None

    for ic, chi in enumerate(np.array(chi_shell)):
        for ib, beta in enumerate(np.array(betas)):
            weight = weight_p2(theta_src_m, beta)

            if not validate_nonnegative(weight):
                continue

            form_factors = jax.vmap(lambda n: shell_form_factor_anisotropic_single(n, weight, chi))(HARMONICS)
            cube = apply_form_factors_to_baseline(point_cube, form_factors)

            Sdet = float(detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
            Sff = float(full_3d_score(cube, theta_grid))

            rdet = ratio_safe(Sdet, Sdet_point)
            rff = ratio_safe(Sff, Sff_point)
            q = tradeoff_q(rdet, rff)

            Rdet[ic, ib] = rdet
            Rff[ic, ib] = rff
            Q[ic, ib] = q

            if best is None or q < best["Q"]:
                best = {
                    "chi": chi,
                    "beta": beta,
                    "Q": q,
                    "Rdet": rdet,
                    "Rff": rff,
                    "cube": np.array(cube),
                }

    return Rdet, Rff, Q, best


def robustness_check(best_chi, best_beta, point_cube, Sdet_point, Sff_point):
    chis = np.array([
        best_chi - ROBUST_DELTA_CHI,
        best_chi - 0.5 * ROBUST_DELTA_CHI,
        best_chi,
        best_chi + 0.5 * ROBUST_DELTA_CHI,
        best_chi + ROBUST_DELTA_CHI,
    ], dtype=np.float32)

    weight = weight_p2(theta_src_m, best_beta)

    rdet_vals = []
    rff_vals = []

    for chi in chis:
        form_factors = jax.vmap(lambda n: shell_form_factor_anisotropic_single(n, weight, chi))(HARMONICS)
        cube = apply_form_factors_to_baseline(point_cube, form_factors)

        Sdet = float(detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
        Sff = float(full_3d_score(cube, theta_grid))

        rdet_vals.append(ratio_safe(Sdet, Sdet_point))
        rff_vals.append(ratio_safe(Sff, Sff_point))

    return chis, np.array(rdet_vals), np.array(rff_vals)


# ----------------------------
# Main
# ----------------------------
def main():
    print("Starting Task 13B shell-fast anisotropic scan (JAX)...", flush=True)

    point_cube = build_point_baseline_intensity(HARMONICS, theta_grid, phi_grid)
    Sdet_point = float(detector_score(point_cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
    Sff_point = float(full_3d_score(point_cube, theta_grid))
    point_profile = angular_profile_fixed_phi(point_cube, phi_det_index)

    print(f"Sdet(point) = {Sdet_point:.6e}")
    print(f"Sff(point)  = {Sff_point:.6e}")

    shell_Rdet, shell_Rff, shell_Q, best_shell = run_shell_family(point_cube, Sdet_point, Sff_point)

    print("\nBest shell case:")
    print(
        f"  chi={best_shell['chi']:.3f}, beta={best_shell['beta']:.3f}, "
        f"Rdet={best_shell['Rdet']:.6e}, Rff3D={best_shell['Rff']:.6e}, Q={best_shell['Q']:.6e}"
    )

    save_heatmap(shell_Rdet, np.array(betas), np.array(chi_shell), "Task 13B shell-fast: $R_{\\det}$", r"$R_{\det}$", "task13B_shell_fast_Rdet.png")
    save_heatmap(shell_Rff, np.array(betas), np.array(chi_shell), "Task 13B shell-fast: $R_{\\mathrm{ff}}^{3D}$", r"$R_{\mathrm{ff}}^{3D}$", "task13B_shell_fast_Rff3D.png")
    save_heatmap(shell_Q, np.array(betas), np.array(chi_shell), "Task 13B shell-fast: $Q=R_{\\det}/R_{\\mathrm{ff}}^{3D}$", r"$Q$", "task13B_shell_fast_Q.png")

    best_shell_profile = angular_profile_fixed_phi(best_shell["cube"], phi_det_index)
    save_profile(
        np.array(theta_grid),
        point_profile,
        best_shell_profile,
        rf"Task 13B shell-fast best profile ($\chi={best_shell['chi']:.2f}$, $\beta={best_shell['beta']:.2f}$)",
        "task13B_shell_fast_best_profile.png",
    )

    rob_chis, rob_rdet, rob_rff = robustness_check(best_shell["chi"], best_shell["beta"], point_cube, Sdet_point, Sff_point)
    save_robustness_plot(
        rob_chis,
        rob_rdet,
        rob_rff,
        "Task 13B shell-fast robustness",
        "task13B_shell_fast_robustness.png",
    )

    print("\nTask 13B shell-fast complete.", flush=True)


if __name__ == "__main__":
    main()