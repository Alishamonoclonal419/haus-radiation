#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ============================================================
# Task 14A: finite coherence / retardation-envelope benchmark
# Point-source baseline + harmonic coherence envelope
# ============================================================

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Physical / numerical settings
# ----------------------------
W0 = 1.0
N_MAX = 8
HARMONICS = jnp.arange(1, N_MAX + 1, dtype=jnp.float32)

N_THETA = 81
N_PHI = 73

theta_grid = jnp.linspace(0.0, jnp.pi, N_THETA)
phi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N_PHI)
theta_obs, phi_obs = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")

dtheta = float(theta_grid[1] - theta_grid[0])
dphi = float(phi_grid[1] - phi_grid[0])

THETA_MIN = 0.55
THETA_MAX = 0.80
PHI_DET = 0.0
phi_det_index = int(np.argmin(np.abs(np.array(phi_grid) - PHI_DET)))

# coherence parameter eta = omega * tau_c
ETA_SCAN = jnp.array([0.10, 0.30, 0.50, 1.00, 1.50, 2.00], dtype=jnp.float32)

# robustness around best eta
ROBUSTNESS_DETA = 0.20


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
# Coherence envelopes
# ----------------------------
@jax.jit
def exponential_power_envelope(harmonics, eta):
    # |G_n|^2 = 1 / (1 + (n eta)^2)
    return 1.0 / (1.0 + (harmonics * eta) ** 2)


@jax.jit
def gaussian_power_envelope(harmonics, eta):
    # |G_n|^2 = exp[-(n eta)^2]
    return jnp.exp(- (harmonics * eta) ** 2)


@jax.jit
def apply_harmonic_power_weights(point_cube, power_weights):
    return point_cube * power_weights[:, None, None]


# ----------------------------
# Observables
# ----------------------------
@jax.jit
def detector_score(intensity_cube, theta_grid, theta_min, theta_max, phi_index):
    mask = ((theta_grid >= theta_min) & (theta_grid <= theta_max)).astype(jnp.float32)
    selected = intensity_cube[:, :, phi_index]  # (Nh, Ntheta)
    return jnp.sum(selected * mask[None, :]) * dtheta


@jax.jit
def full_3d_score(intensity_cube, theta_grid):
    sin_theta = jnp.sin(theta_grid).astype(jnp.float32)[:, None]
    return jnp.sum(intensity_cube * sin_theta[None, :, :]) * dtheta * dphi


def angular_profile_fixed_phi(intensity_cube, phi_index):
    return np.array(jnp.sum(intensity_cube[:, :, phi_index], axis=0))


def harmonic_content(intensity_cube):
    # integrated power per harmonic over full sphere
    vals = []
    for n in range(intensity_cube.shape[0]):
        vals.append(float(full_3d_score(intensity_cube[n:n+1, :, :], theta_grid)))
    return np.array(vals)


def ratio_safe(num, den, eps=1e-30):
    return float(num / max(den, eps))


# ----------------------------
# Plot helpers
# ----------------------------
def save_curve(x, y, xlabel, ylabel, title, fname, marker="o"):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_profile(theta_grid_np, point_profile, coh_profile, eta_best, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid_np, point_profile, label="point baseline")
    plt.plot(theta_grid_np, coh_profile, label=f"best coherence case ($\\eta={eta_best:.2f}$)")
    plt.axvspan(THETA_MIN, THETA_MAX, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title("Task 14A: best-profile comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_harmonic_weights(harmonics_np, power_weights, title, fname):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(harmonics_np, power_weights, marker="o")
    plt.xlabel(r"Harmonic index $n$")
    plt.ylabel(r"$|G_n|^2$")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_harmonic_content(harmonics_np, point_h, coh_h, eta_best, fname):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(harmonics_np, point_h, marker="o", label="point baseline")
    plt.plot(harmonics_np, coh_h, marker="s", label=f"best coherence case ($\\eta={eta_best:.2f}$)")
    plt.xlabel(r"Harmonic index $n$")
    plt.ylabel("Integrated harmonic power")
    plt.title("Task 14A: harmonic-content comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


# ----------------------------
# Main run
# ----------------------------
def run_family(point_cube, Sdet_point, Sff_point, envelope_kind="exponential"):
    results = []

    best = None

    for eta in np.array(ETA_SCAN):
        eta_j = jnp.array(eta, dtype=jnp.float32)

        if envelope_kind == "exponential":
            weights = exponential_power_envelope(HARMONICS, eta_j)
        elif envelope_kind == "gaussian":
            weights = gaussian_power_envelope(HARMONICS, eta_j)
        else:
            raise ValueError("Unknown envelope_kind")

        cube = apply_harmonic_power_weights(point_cube, weights)

        Sdet = float(detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
        Sff = float(full_3d_score(cube, theta_grid))

        Rdet = ratio_safe(Sdet, Sdet_point)
        Rff = ratio_safe(Sff, Sff_point)
        Q = ratio_safe(Rdet, Rff)

        entry = {
            "eta": float(eta),
            "weights": np.array(weights),
            "cube": np.array(cube),
            "Rdet": Rdet,
            "Rff": Rff,
            "Q": Q,
        }
        results.append(entry)

        if best is None or Q < best["Q"]:
            best = entry

    return results, best


def robustness_scan(point_cube, Sdet_point, Sff_point, eta0, envelope_kind="exponential"):
    etas = np.array([
        max(0.02, eta0 - ROBUSTNESS_DETA),
        max(0.02, eta0 - 0.5 * ROBUSTNESS_DETA),
        eta0,
        eta0 + 0.5 * ROBUSTNESS_DETA,
        eta0 + ROBUSTNESS_DETA,
    ], dtype=np.float32)

    Rdet_vals = []
    Rff_vals = []

    for eta in etas:
        eta_j = jnp.array(eta, dtype=jnp.float32)

        if envelope_kind == "exponential":
            weights = exponential_power_envelope(HARMONICS, eta_j)
        else:
            weights = gaussian_power_envelope(HARMONICS, eta_j)

        cube = apply_harmonic_power_weights(point_cube, weights)

        Sdet = float(detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
        Sff = float(full_3d_score(cube, theta_grid))

        Rdet_vals.append(ratio_safe(Sdet, Sdet_point))
        Rff_vals.append(ratio_safe(Sff, Sff_point))

    return etas, np.array(Rdet_vals), np.array(Rff_vals)


def main():
    print("Starting Task 14A coherence-envelope scan (JAX)...", flush=True)

    point_cube = build_point_baseline_intensity(HARMONICS, theta_grid, phi_grid)
    Sdet_point = float(detector_score(point_cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
    Sff_point = float(full_3d_score(point_cube, theta_grid))

    print(f"Sdet(point) = {Sdet_point:.6e}")
    print(f"Sff(point)  = {Sff_point:.6e}")

    point_profile = angular_profile_fixed_phi(point_cube, phi_det_index)
    point_hcontent = harmonic_content(np.array(point_cube))

    # First pass: exponential coherence envelope
    results, best = run_family(point_cube, Sdet_point, Sff_point, envelope_kind="exponential")

    print("\nBest exponential-envelope case:")
    print(
        f"  eta={best['eta']:.3f}, "
        f"Rdet={best['Rdet']:.6e}, "
        f"Rff3D={best['Rff']:.6e}, "
        f"Q={best['Q']:.6e}"
    )

    etas = np.array([r["eta"] for r in results])
    Rdet_vals = np.array([r["Rdet"] for r in results])
    Rff_vals = np.array([r["Rff"] for r in results])
    Q_vals = np.array([r["Q"] for r in results])

    save_curve(
        etas, Rdet_vals,
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$R_{\det}$",
        title="Task 14A: detector-window ratio vs coherence parameter",
        fname="task14A_Rdet_vs_eta.png"
    )

    save_curve(
        etas, Rff_vals,
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$R_{\mathrm{ff}}^{3D}$",
        title="Task 14A: total-radiation ratio vs coherence parameter",
        fname="task14A_Rff3D_vs_eta.png"
    )

    save_curve(
        etas, Q_vals,
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$Q=R_{\det}/R_{\mathrm{ff}}^{3D}$",
        title="Task 14A: local/global tradeoff vs coherence parameter",
        fname="task14A_Q_vs_eta.png"
    )

    best_profile = angular_profile_fixed_phi(best["cube"], phi_det_index)
    save_profile(
        np.array(theta_grid),
        point_profile,
        best_profile,
        best["eta"],
        "task14A_best_profile.png"
    )

    save_harmonic_weights(
        np.array(HARMONICS),
        best["weights"],
        title=rf"Task 14A: best coherence envelope ($\eta={best['eta']:.2f}$)",
        fname="task14A_best_envelope_weights.png"
    )

    best_hcontent = harmonic_content(best["cube"])
    save_harmonic_content(
        np.array(HARMONICS),
        point_hcontent,
        best_hcontent,
        best["eta"],
        "task14A_harmonic_content_comparison.png"
    )

    rob_etas, rob_rdet, rob_rff = robustness_scan(
        point_cube, Sdet_point, Sff_point, best["eta"], envelope_kind="exponential"
    )

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(rob_etas, rob_rdet, marker="o", label=r"$R_{\det}$")
    plt.plot(rob_etas, rob_rff, marker="s", label=r"$R_{\mathrm{ff}}^{3D}$")
    plt.xlabel(r"$\eta=\omega\tau_c$")
    plt.ylabel("Ratio to point baseline")
    plt.title("Task 14A: robustness around best coherence parameter")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "task14A_robustness.png")
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")

    print("\nTask 14A complete.", flush=True)


if __name__ == "__main__":
    main()