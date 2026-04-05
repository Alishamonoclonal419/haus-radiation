#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ============================================================
# Task 14B: extended source + finite coherence
# Fresh-start JAX benchmark
# ============================================================

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# grids / constants
# ----------------------------
N_MAX = 8
N_THETA = 81
N_PHI = 73

theta_grid = jnp.linspace(0.0, jnp.pi, N_THETA)
phi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N_PHI)
theta_obs, phi_obs = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")

harmonics = jnp.arange(1, N_MAX + 1, dtype=jnp.float32)

dtheta = float(theta_grid[1] - theta_grid[0])
dphi = float(phi_grid[1] - phi_grid[0])

THETA_MIN = 0.55
THETA_MAX = 0.80
PHI_DET = 0.0
phi_det_index = int(np.argmin(np.abs(np.array(phi_grid) - PHI_DET)))

# modest scan first; widen later if needed
CHI_SHELL_SCAN = jnp.array([0.20, 0.30, 0.40, 0.50, 0.60], dtype=jnp.float32)
CHI_VOLUME_SCAN = jnp.array([0.60, 0.80, 1.00, 1.20, 1.40], dtype=jnp.float32)
ETA_SCAN = jnp.array([0.10, 0.30, 0.50, 1.00, 1.50], dtype=jnp.float32)

# ----------------------------
# baseline point-source angular cube
# ----------------------------
@jax.jit
def point_intensity_cube(harmonics, theta_grid, phi_grid):
    theta_obs, phi_obs = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")

    def one_harmonic(n):
        base = jnp.sin(theta_obs) ** 2
        mod_theta = 1.0 + 0.30 * jnp.cos((n - 1.0) * theta_obs) ** 2
        mod_phi = 1.0 + 0.10 * jnp.cos(phi_obs) * jnp.exp(-0.15 * (n - 1.0))
        return base * mod_theta * mod_phi

    return jax.vmap(one_harmonic)(harmonics)


# ----------------------------
# spatial form factors
# ----------------------------
@jax.jit
def shell_form_factor_power(n, chi):
    """
    Thin spherical shell:
        A_n ~ sin(n chi)/(n chi)
    We use power weight |A_n|^2.
    """
    x = n * chi
    val = jnp.where(jnp.abs(x) < 1e-7, 1.0, jnp.sin(x) / x)
    return val ** 2


@jax.jit
def volume_form_factor_power(n, chi):
    """
    Uniform sphere:
        A_n ~ 3 [sin(n chi) - n chi cos(n chi)] / (n chi)^3
    Again use |A_n|^2.
    """
    x = n * chi
    small = jnp.abs(x) < 1e-6
    amp = jnp.where(
        small,
        1.0,
        3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x ** 3)
    )
    return amp ** 2


@jax.jit
def shell_spatial_weights(harmonics, chi):
    return jax.vmap(lambda n: shell_form_factor_power(n, chi))(harmonics)


@jax.jit
def volume_spatial_weights(harmonics, chi):
    return jax.vmap(lambda n: volume_form_factor_power(n, chi))(harmonics)


# ----------------------------
# coherence envelope
# ----------------------------
@jax.jit
def coherence_weights(harmonics, eta):
    # exponential/Lorentzian-type power envelope
    return 1.0 / (1.0 + (harmonics * eta) ** 2)


# ----------------------------
# combined weighted cube
# ----------------------------
@jax.jit
def apply_weights(cube, harmonic_power_weights):
    return cube * harmonic_power_weights[:, None, None]


# ----------------------------
# observables
# ----------------------------
@jax.jit
def detector_score(cube, theta_grid, theta_min, theta_max, phi_index):
    mask = ((theta_grid >= theta_min) & (theta_grid <= theta_max)).astype(jnp.float32)
    selected = cube[:, :, phi_index]
    return jnp.sum(selected * mask[None, :]) * dtheta


@jax.jit
def full_3d_score(cube, theta_grid):
    sin_theta = jnp.sin(theta_grid).astype(jnp.float32)[:, None]
    return jnp.sum(cube * sin_theta[None, :, :]) * dtheta * dphi


def ratio_safe(a, b, eps=1e-30):
    return float(a / max(b, eps))


def fixed_phi_profile(cube, phi_index):
    return np.array(jnp.sum(cube[:, :, phi_index], axis=0))


def harmonic_content(cube):
    vals = []
    for i in range(cube.shape[0]):
        vals.append(float(full_3d_score(cube[i:i+1], theta_grid)))
    return np.array(vals)


# ----------------------------
# plotting helpers
# ----------------------------
def save_heatmap(arr, xvals, yvals, xlabel, ylabel, cbar, title, fname):
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]]
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label=cbar)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_profile(theta_np, point_prof, best_prof, label, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(theta_np, point_prof, label="point baseline")
    plt.plot(theta_np, best_prof, label=label)
    plt.axvspan(THETA_MIN, THETA_MAX, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\phi$ angular intensity")
    plt.title("Task 14B: best-profile comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_curve(x, y, xlabel, ylabel, title, fname, marker="o"):
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_harmonic_compare(hn, point_h, best_h, label, fname):
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(hn, point_h, marker="o", label="point baseline")
    plt.plot(hn, best_h, marker="s", label=label)
    plt.xlabel(r"Harmonic index $n$")
    plt.ylabel("Integrated harmonic power")
    plt.title("Task 14B: harmonic-content comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


# ----------------------------
# scan engine
# ----------------------------
def scan_family(point_cube, family_name, chi_scan):
    results = []

    if family_name == "shell":
        spatial_fun = shell_spatial_weights
    elif family_name == "volume":
        spatial_fun = volume_spatial_weights
    else:
        raise ValueError("family_name must be shell or volume")

    for chi in np.array(chi_scan):
        chi_j = jnp.array(chi, dtype=jnp.float32)
        space_w = spatial_fun(harmonics, chi_j)

        row_Q = []
        row_Rdet = []
        row_Rff = []

        for eta in np.array(ETA_SCAN):
            eta_j = jnp.array(eta, dtype=jnp.float32)
            coh_w = coherence_weights(harmonics, eta_j)

            total_w = space_w * coh_w
            cube = apply_weights(point_cube, total_w)

            Sdet = float(detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
            Sff = float(full_3d_score(cube, theta_grid))

            Rdet = ratio_safe(Sdet, SDET_POINT)
            Rff = ratio_safe(Sff, SFF_POINT)
            Q = ratio_safe(Rdet, Rff)

            results.append({
                "family": family_name,
                "chi": float(chi),
                "eta": float(eta),
                "Rdet": Rdet,
                "Rff": Rff,
                "Q": Q,
                "cube": np.array(cube),
                "weights": np.array(total_w),
            })

            row_Q.append(Q)
            row_Rdet.append(Rdet)
            row_Rff.append(Rff)

        heat_Q[family_name].append(row_Q)
        heat_Rdet[family_name].append(row_Rdet)
        heat_Rff[family_name].append(row_Rff)

    best_by_Q = min(results, key=lambda x: x["Q"])
    best_by_Rdet = min(results, key=lambda x: x["Rdet"])
    return results, best_by_Q, best_by_Rdet


# ----------------------------
# main
# ----------------------------
def main():
    global SDET_POINT, SFF_POINT, heat_Q, heat_Rdet, heat_Rff

    print("Starting Task 14B spatiotemporal extended-source scan (JAX)...", flush=True)

    point_cube = point_intensity_cube(harmonics, theta_grid, phi_grid)

    SDET_POINT = float(detector_score(point_cube, theta_grid, THETA_MIN, THETA_MAX, phi_det_index))
    SFF_POINT = float(full_3d_score(point_cube, theta_grid))

    print(f"Sdet(point) = {SDET_POINT:.6e}")
    print(f"Sff(point)  = {SFF_POINT:.6e}")

    point_prof = fixed_phi_profile(point_cube, phi_det_index)
    point_h = harmonic_content(np.array(point_cube))

    heat_Q = {"shell": [], "volume": []}
    heat_Rdet = {"shell": [], "volume": []}
    heat_Rff = {"shell": [], "volume": []}

    # shell
    shell_results, shell_best_Q, shell_best_Rdet = scan_family(point_cube, "shell", CHI_SHELL_SCAN)

    # volume
    volume_results, volume_best_Q, volume_best_Rdet = scan_family(point_cube, "volume", CHI_VOLUME_SCAN)

    print("\nBest shell by Q:")
    print(
        f"  chi={shell_best_Q['chi']:.3f}, eta={shell_best_Q['eta']:.3f}, "
        f"Rdet={shell_best_Q['Rdet']:.6e}, Rff3D={shell_best_Q['Rff']:.6e}, Q={shell_best_Q['Q']:.6e}"
    )
    print("Best shell by Rdet:")
    print(
        f"  chi={shell_best_Rdet['chi']:.3f}, eta={shell_best_Rdet['eta']:.3f}, "
        f"Rdet={shell_best_Rdet['Rdet']:.6e}, Rff3D={shell_best_Rdet['Rff']:.6e}, Q={shell_best_Rdet['Q']:.6e}"
    )

    print("\nBest volume by Q:")
    print(
        f"  chi={volume_best_Q['chi']:.3f}, eta={volume_best_Q['eta']:.3f}, "
        f"Rdet={volume_best_Q['Rdet']:.6e}, Rff3D={volume_best_Q['Rff']:.6e}, Q={volume_best_Q['Q']:.6e}"
    )
    print("Best volume by Rdet:")
    print(
        f"  chi={volume_best_Rdet['chi']:.3f}, eta={volume_best_Rdet['eta']:.3f}, "
        f"Rdet={volume_best_Rdet['Rdet']:.6e}, Rff3D={volume_best_Rdet['Rff']:.6e}, Q={volume_best_Rdet['Q']:.6e}"
    )

    # heatmaps
    save_heatmap(
        np.array(heat_Q["shell"]),
        np.array(ETA_SCAN), np.array(CHI_SHELL_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$Q=R_{\det}/R_{\mathrm{ff}}^{3D}$",
        title="Task 14B shell: local/global tradeoff",
        fname="task14B_shell_Q_heatmap.png"
    )

    save_heatmap(
        np.array(heat_Rdet["shell"]),
        np.array(ETA_SCAN), np.array(CHI_SHELL_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$R_{\det}$",
        title="Task 14B shell: detector ratio",
        fname="task14B_shell_Rdet_heatmap.png"
    )

    save_heatmap(
        np.array(heat_Rff["shell"]),
        np.array(ETA_SCAN), np.array(CHI_SHELL_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$R_{\mathrm{ff}}^{3D}$",
        title="Task 14B shell: total-radiation ratio",
        fname="task14B_shell_Rff_heatmap.png"
    )

    save_heatmap(
        np.array(heat_Q["volume"]),
        np.array(ETA_SCAN), np.array(CHI_VOLUME_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$Q=R_{\det}/R_{\mathrm{ff}}^{3D}$",
        title="Task 14B volume: local/global tradeoff",
        fname="task14B_volume_Q_heatmap.png"
    )

    save_heatmap(
        np.array(heat_Rdet["volume"]),
        np.array(ETA_SCAN), np.array(CHI_VOLUME_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$R_{\det}$",
        title="Task 14B volume: detector ratio",
        fname="task14B_volume_Rdet_heatmap.png"
    )

    save_heatmap(
        np.array(heat_Rff["volume"]),
        np.array(ETA_SCAN), np.array(CHI_VOLUME_SCAN),
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$\chi=b/(cT)$",
        cbar=r"$R_{\mathrm{ff}}^{3D}$",
        title="Task 14B volume: total-radiation ratio",
        fname="task14B_volume_Rff_heatmap.png"
    )

    # best profile and harmonic comparison using best Q case among shell/volume
    all_best = [shell_best_Q, volume_best_Q]
    overall_best = min(all_best, key=lambda x: x["Q"])

    label = f"{overall_best['family']} best-Q case ($\\chi={overall_best['chi']:.2f},\\ \\eta={overall_best['eta']:.2f}$)"
    best_prof = fixed_phi_profile(overall_best["cube"], phi_det_index)
    best_h = harmonic_content(overall_best["cube"])

    save_profile(
        np.array(theta_grid),
        point_prof,
        best_prof,
        label,
        "task14B_best_profile.png"
    )

    save_harmonic_compare(
        np.array(harmonics),
        point_h,
        best_h,
        label,
        "task14B_harmonic_content_comparison.png"
    )

    # Q vs eta along best chi slices
    shell_best_chi = shell_best_Q["chi"]
    shell_eta = []
    shell_Qslice = []
    for r in shell_results:
        if abs(r["chi"] - shell_best_chi) < 1e-9:
            shell_eta.append(r["eta"])
            shell_Qslice.append(r["Q"])
    idx = np.argsort(shell_eta)
    save_curve(
        np.array(shell_eta)[idx],
        np.array(shell_Qslice)[idx],
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$Q$",
        title=rf"Task 14B shell: $Q$ vs $\eta$ at best $\chi={shell_best_chi:.2f}$",
        fname="task14B_shell_Q_vs_eta_bestchi.png"
    )

    volume_best_chi = volume_best_Q["chi"]
    volume_eta = []
    volume_Qslice = []
    for r in volume_results:
        if abs(r["chi"] - volume_best_chi) < 1e-9:
            volume_eta.append(r["eta"])
            volume_Qslice.append(r["Q"])
    idx = np.argsort(volume_eta)
    save_curve(
        np.array(volume_eta)[idx],
        np.array(volume_Qslice)[idx],
        xlabel=r"$\eta=\omega\tau_c$",
        ylabel=r"$Q$",
        title=rf"Task 14B volume: $Q$ vs $\eta$ at best $\chi={volume_best_chi:.2f}$",
        fname="task14B_volume_Q_vs_eta_bestchi.png"
    )

    print("\nTask 14B complete.", flush=True)


if __name__ == "__main__":
    main()