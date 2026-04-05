import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Task 13B: Anisotropic extended source benchmark (standalone)
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

# Harmonics
N_MAX = 8
HARMONICS = np.arange(1, N_MAX + 1, dtype=float)

# Observation grid
N_THETA = 81
N_PHI = 73
theta_grid = np.linspace(0.0, np.pi, N_THETA)
phi_grid = np.linspace(0.0, 2.0 * np.pi, N_PHI)

theta_obs, phi_obs = np.meshgrid(theta_grid, phi_grid, indexing="ij")

# Detector window
THETA_MIN = 0.55
THETA_MAX = 0.80
PHI_DET = 0.0
phi_det_index = int(np.argmin(np.abs(phi_grid - PHI_DET)))

# Source integration grids
N_THETA_SRC = 61
N_PHI_SRC = 61
theta_src = np.linspace(0.0, np.pi, N_THETA_SRC)
phi_src = np.linspace(0.0, 2.0 * np.pi, N_PHI_SRC)
theta_src_m, phi_src_m = np.meshgrid(theta_src, phi_src, indexing="ij")

# Volume radial grid
N_R = 41
r_unit = np.linspace(0.0, 1.0, N_R)

# Scan ranges
chi_shell = np.array([0.42, 0.46, 0.50, 0.54, 0.58])
chi_volume = np.array([1.00, 1.10, 1.20, 1.30, 1.40])
betas = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

# Robustness neighborhood half-width
ROBUST_DELTA_CHI = 0.02


# ----------------------------
# Helper math
# ----------------------------
def legendre_p2(x):
    return 0.5 * (3.0 * x**2 - 1.0)


def weight_p2(theta, beta):
    return 1.0 + beta * legendre_p2(np.cos(theta))


def validate_nonnegative(w, tol=1e-12):
    return np.min(w) >= -tol


def ratio_safe(num, den, eps=1e-30):
    return float(num / max(den, eps))


def tradeoff_q(rdet, rff):
    return ratio_safe(rdet, rff)


# ----------------------------
# Baseline point-source model
# ----------------------------
def build_point_baseline_intensity(harmonics, theta_grid, phi_grid):
    """
    Standalone baseline model.

    This is intentionally simple but structured:
    - dominant sin^2(theta) radiation envelope
    - harmonic-dependent angular modulation
    - slight phi dependence so detector-window logic is meaningful
    """
    theta_obs, phi_obs = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    cube = []

    for n in harmonics:
        base = np.sin(theta_obs) ** 2
        mod_theta = 1.0 + 0.30 * np.cos((n - 1) * theta_obs) ** 2
        mod_phi = 1.0 + 0.10 * np.cos(phi_obs) * np.exp(-0.15 * (n - 1))
        profile = base * mod_theta * mod_phi
        cube.append(profile)

    return np.array(cube, dtype=float)


# ----------------------------
# Anisotropic form factors
# ----------------------------
def shell_form_factor_anisotropic(k, theta_obs, phi_obs, theta_src, phi_src, weight_src, b):
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


def apply_form_factor_to_baseline(point_cube, form_factors):
    out = np.empty_like(point_cube)
    for i in range(point_cube.shape[0]):
        out[i] = point_cube[i] * np.abs(form_factors[i]) ** 2
    return out


# ----------------------------
# Observables
# ----------------------------
def detector_score(intensity_cube, theta_grid, theta_min, theta_max, phi_index=0):
    mask = (theta_grid >= theta_min) & (theta_grid <= theta_max)
    dtheta = float(theta_grid[1] - theta_grid[0])
    score = 0.0
    for n in range(intensity_cube.shape[0]):
        score += np.sum(intensity_cube[n, mask, phi_index]) * dtheta
    return float(score)


def full_3d_score(intensity_cube, theta_grid, phi_grid):
    dtheta = float(theta_grid[1] - theta_grid[0])
    dphi = float(phi_grid[1] - phi_grid[0])
    sin_theta = np.sin(theta_grid)[:, None]
    score = 0.0
    for n in range(intensity_cube.shape[0]):
        score += np.sum(intensity_cube[n] * sin_theta) * dtheta * dphi
    return float(score)


def angular_profile_fixed_phi(intensity_cube, phi_index):
    return np.sum(intensity_cube[:, :, phi_index], axis=0)


# ----------------------------
# Plotting
# ----------------------------
def save_heatmap(arr, xvals, yvals, title, cbar_label, fname):
    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\chi$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()
    print(f"[Saved] {os.path.join(FIG_DIR, fname)}")


def save_profile(theta_grid, point_profile, ext_profile, title, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, point_profile, label="point baseline")
    plt.plot(theta_grid, ext_profile, label="best anisotropic source")
    plt.axvspan(THETA_MIN, THETA_MAX, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()
    print(f"[Saved] {os.path.join(FIG_DIR, fname)}")


def save_robustness_plot(chis, rdet_vals, rff_vals, title, fname):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(chis, rdet_vals, marker="o", label=r"$R_{\det}$")
    plt.plot(chis, rff_vals, marker="s", label=r"$R_{\mathrm{ff}}^{3D}$")
    plt.xlabel(r"$\chi$")
    plt.ylabel("Ratio to point baseline")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=220)
    plt.close()
    print(f"[Saved] {os.path.join(FIG_DIR, fname)}")


# ----------------------------
# Robustness check
# ----------------------------
def robustness_check(kind, chi0, beta0, point_cube, Sdet_point, Sff_point, theta_obs, phi_obs, theta_src_m, phi_src_m):
    chis = np.array([chi0 - ROBUST_DELTA_CHI, chi0 - 0.5 * ROBUST_DELTA_CHI, chi0, chi0 + 0.5 * ROBUST_DELTA_CHI, chi0 + ROBUST_DELTA_CHI])
    weight = weight_p2(theta_src_m, beta0)

    rdet_vals = []
    rff_vals = []

    for chi in chis:
        form_factors = []
        for n in HARMONICS:
            k_n = float(n)
            if kind == "shell":
                F = shell_form_factor_anisotropic(
                    k=k_n,
                    theta_obs=theta_obs,
                    phi_obs=phi_obs,
                    theta_src=theta_src_m,
                    phi_src=phi_src_m,
                    weight_src=weight,
                    b=chi,
                )
            else:
                r_grid = chi * r_unit
                F = volume_form_factor_anisotropic(
                    k=k_n,
                    theta_obs=theta_obs,
                    phi_obs=phi_obs,
                    r_grid=r_grid,
                    theta_src=theta_src_m,
                    phi_src=phi_src_m,
                    weight_ang=weight,
                )
            form_factors.append(F)

        form_factors = np.array(form_factors)
        cube = apply_form_factor_to_baseline(point_cube, form_factors)
        Sdet = detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_index=phi_det_index)
        Sff = full_3d_score(cube, theta_grid, phi_grid)

        rdet_vals.append(ratio_safe(Sdet, Sdet_point))
        rff_vals.append(ratio_safe(Sff, Sff_point))

    return chis, np.array(rdet_vals), np.array(rff_vals)


# ----------------------------
# Main
# ----------------------------
def run_family(kind, chi_values, point_cube, Sdet_point, Sff_point, theta_obs, phi_obs, theta_src_m, phi_src_m):
    Rdet = np.full((len(chi_values), len(betas)), np.nan)
    Rff = np.full((len(chi_values), len(betas)), np.nan)
    Q = np.full((len(chi_values), len(betas)), np.nan)

    best = None

    for ic, chi in enumerate(chi_values):
        for ib, beta in enumerate(betas):
            w = weight_p2(theta_src_m, beta)
            if not validate_nonnegative(w):
                continue

            form_factors = []
            for n in HARMONICS:
                k_n = float(n)
                if kind == "shell":
                    F = shell_form_factor_anisotropic(
                        k=k_n,
                        theta_obs=theta_obs,
                        phi_obs=phi_obs,
                        theta_src=theta_src_m,
                        phi_src=phi_src_m,
                        weight_src=w,
                        b=chi,
                    )
                else:
                    r_grid = chi * r_unit
                    F = volume_form_factor_anisotropic(
                        k=k_n,
                        theta_obs=theta_obs,
                        phi_obs=phi_obs,
                        r_grid=r_grid,
                        theta_src=theta_src_m,
                        phi_src=phi_src_m,
                        weight_ang=w,
                    )

                form_factors.append(F)

            form_factors = np.array(form_factors)
            cube = apply_form_factor_to_baseline(point_cube, form_factors)

            Sdet = detector_score(cube, theta_grid, THETA_MIN, THETA_MAX, phi_index=phi_det_index)
            Sff = full_3d_score(cube, theta_grid, phi_grid)

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
                    "cube": cube,
                }

    return Rdet, Rff, Q, best


def main():
    print("Starting Task 13B anisotropic extended-source scan...", flush=True)

    point_cube = build_point_baseline_intensity(HARMONICS, theta_grid, phi_grid)
    Sdet_point = detector_score(point_cube, theta_grid, THETA_MIN, THETA_MAX, phi_index=phi_det_index)
    Sff_point = full_3d_score(point_cube, theta_grid, phi_grid)
    point_profile = angular_profile_fixed_phi(point_cube, phi_det_index)

    print(f"Sdet(point) = {Sdet_point:.6e}")
    print(f"Sff(point)  = {Sff_point:.6e}")

    shell_Rdet, shell_Rff, shell_Q, best_shell = run_family(
        "shell", chi_shell, point_cube, Sdet_point, Sff_point, theta_obs, phi_obs, theta_src_m, phi_src_m
    )

    volume_Rdet, volume_Rff, volume_Q, best_volume = run_family(
        "volume", chi_volume, point_cube, Sdet_point, Sff_point, theta_obs, phi_obs, theta_src_m, phi_src_m
    )

    print("\nBest shell case:")
    print(
        f"  chi={best_shell['chi']:.3f}, beta={best_shell['beta']:.3f}, "
        f"Rdet={best_shell['Rdet']:.6e}, Rff3D={best_shell['Rff']:.6e}, Q={best_shell['Q']:.6e}"
    )

    print("Best volume case:")
    print(
        f"  chi={best_volume['chi']:.3f}, beta={best_volume['beta']:.3f}, "
        f"Rdet={best_volume['Rdet']:.6e}, Rff3D={best_volume['Rff']:.6e}, Q={best_volume['Q']:.6e}"
    )

    # Heatmaps
    save_heatmap(shell_Rdet, betas, chi_shell, "Task 13B shell: $R_{\\det}$", r"$R_{\det}$", "task13B_shell_Rdet.png")
    save_heatmap(shell_Rff, betas, chi_shell, "Task 13B shell: $R_{\\mathrm{ff}}^{3D}$", r"$R_{\mathrm{ff}}^{3D}$", "task13B_shell_Rff3D.png")
    save_heatmap(shell_Q, betas, chi_shell, "Task 13B shell: $Q=R_{\\det}/R_{\\mathrm{ff}}^{3D}$", r"$Q$", "task13B_shell_Q.png")

    save_heatmap(volume_Rdet, betas, chi_volume, "Task 13B volume: $R_{\\det}$", r"$R_{\det}$", "task13B_volume_Rdet.png")
    save_heatmap(volume_Rff, betas, chi_volume, "Task 13B volume: $R_{\\mathrm{ff}}^{3D}$", r"$R_{\mathrm{ff}}^{3D}$", "task13B_volume_Rff3D.png")
    save_heatmap(volume_Q, betas, chi_volume, "Task 13B volume: $Q=R_{\\det}/R_{\\mathrm{ff}}^{3D}$", r"$Q$", "task13B_volume_Q.png")

    # Best-case profiles
    best_shell_profile = angular_profile_fixed_phi(best_shell["cube"], phi_det_index)
    best_volume_profile = angular_profile_fixed_phi(best_volume["cube"], phi_det_index)

    save_profile(
        theta_grid,
        point_profile,
        best_shell_profile,
        rf"Task 13B shell best profile ($\chi={best_shell['chi']:.2f}$, $\beta={best_shell['beta']:.2f}$)",
        "task13B_shell_best_profile.png",
    )
    save_profile(
        theta_grid,
        point_profile,
        best_volume_profile,
        rf"Task 13B volume best profile ($\chi={best_volume['chi']:.2f}$, $\beta={best_volume['beta']:.2f}$)",
        "task13B_volume_best_profile.png",
    )

    # Robustness
    shell_chis, shell_rdet_rob, shell_rff_rob = robustness_check(
        "shell", best_shell["chi"], best_shell["beta"], point_cube, Sdet_point, Sff_point,
        theta_obs, phi_obs, theta_src_m, phi_src_m
    )
    volume_chis, volume_rdet_rob, volume_rff_rob = robustness_check(
        "volume", best_volume["chi"], best_volume["beta"], point_cube, Sdet_point, Sff_point,
        theta_obs, phi_obs, theta_src_m, phi_src_m
    )

    save_robustness_plot(
        shell_chis, shell_rdet_rob, shell_rff_rob,
        "Task 13B shell robustness",
        "task13B_shell_robustness.png",
    )
    save_robustness_plot(
        volume_chis, volume_rdet_rob, volume_rff_rob,
        "Task 13B volume robustness",
        "task13B_volume_robustness.png",
    )

    print("\nTask 13B complete.", flush=True)


if __name__ == "__main__":
    main()