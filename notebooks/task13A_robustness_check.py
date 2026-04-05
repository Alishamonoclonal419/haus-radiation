import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# USER SETTINGS: replace these imports/functions with yours
# ------------------------------------------------------------

# You need to replace these with the actual functions from your codebase.
#
# Required functions:
#
# 1. point_source_intensity_map(phi_fixed=0.0)
#    -> returns I_point with shape (n_theta, n_harmonics)
#
# 2. extended_source_intensity_map(kind, chi, phi_fixed=0.0)
#    -> returns I_ext with shape (n_theta, n_harmonics)
#
# 3. detector_window_ratio(I_ext, I_point, theta_grid, theta_min, theta_max)
#    -> scalar R_det
#
# 4. full_radiation_ratio(I_ext, I_point, theta_grid)
#    -> scalar R_ff_3D
#
# 5. form_factor_sq(kind, chi, n_array)
#    -> array |F_n|^2 for n=1..N
#
# If your code already computes I_ext by multiplying point harmonics
# by |F_n|^2, this diagnostic is exactly what you want.


# ---------------- MOCK INTERFACE PLACEHOLDERS ----------------
# DELETE these after wiring to your own code.

def point_source_intensity_map(phi_fixed=0.0):
    theta = np.linspace(0.0, np.pi, 181)
    n = np.arange(1, 9)
    I = np.zeros((len(theta), len(n)))
    for j, nj in enumerate(n):
        I[:, j] = (np.sin(theta) ** 2) * np.exp(-0.25 * (nj - 1))
    return theta, n, I

def shell_form_factor_sq(chi, n_array):
    x = chi * n_array * np.pi
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-14
    out[mask] = (np.sin(x[mask]) / x[mask]) ** 2
    return out

def volume_form_factor_sq(chi, n_array):
    x = chi * n_array * np.pi
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-14
    xm = x[mask]
    out[mask] = (3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm ** 3)) ** 2
    return out

def form_factor_sq(kind, chi, n_array):
    if kind == "shell":
        return shell_form_factor_sq(chi, n_array)
    elif kind == "volume":
        return volume_form_factor_sq(chi, n_array)
    raise ValueError(f"Unknown kind: {kind}")

def extended_source_intensity_map(kind, chi, phi_fixed=0.0):
    theta, n, I_point = point_source_intensity_map(phi_fixed=phi_fixed)
    F2 = form_factor_sq(kind, chi, n)
    I_ext = I_point * F2[None, :]
    return theta, n, I_ext

def detector_window_ratio(I_ext, I_point, theta_grid, theta_min, theta_max):
    mask = (theta_grid >= theta_min) & (theta_grid <= theta_max)
    s_ext = np.sum(I_ext[mask, :])
    s_pt = np.sum(I_point[mask, :])
    return s_ext / s_pt if s_pt > 0 else np.nan

def full_radiation_ratio(I_ext, I_point, theta_grid):
    # crude 3D weighting
    w = np.sin(theta_grid)[:, None]
    s_ext = np.sum(I_ext * w)
    s_pt = np.sum(I_point * w)
    return s_ext / s_pt if s_pt > 0 else np.nan
# ------------------------------------------------------------


def profile_from_map(I_map):
    return np.sum(I_map, axis=1)


def run_robustness(kind, chi0, dchi, theta_min, theta_max, phi_fixed=0.0):
    theta, n, I_point = point_source_intensity_map(phi_fixed=phi_fixed)

    chi_values = np.array([
        chi0 - dchi,
        chi0 - 0.5 * dchi,
        chi0,
        chi0 + 0.5 * dchi,
        chi0 + dchi
    ])

    rows = []

    print(f"\n=== {kind.upper()} ROBUSTNESS CHECK ===")
    print(f"chi0 = {chi0:.5f}, dchi = {dchi:.5f}")
    print(f"detector window = [{theta_min:.3f}, {theta_max:.3f}]")
    print()

    for chi in chi_values:
        _, _, I_ext = extended_source_intensity_map(kind, chi, phi_fixed=phi_fixed)
        R_det = detector_window_ratio(I_ext, I_point, theta, theta_min, theta_max)
        R_ff = full_radiation_ratio(I_ext, I_point, theta)
        F2 = form_factor_sq(kind, chi, n)

        rows.append((chi, R_det, R_ff, F2, I_ext))

        print(f"chi = {chi:.5f} | R_det = {R_det:.6e} | R_ff3D = {R_ff:.6e}")
        print("  |F_n|^2 =", "  ".join([f"n={int(nn)}:{ff:.3e}" for nn, ff in zip(n, F2)]))

    return theta, n, I_point, rows


def plot_robustness(kind, theta, I_point, rows, theta_min, theta_max, outprefix):
    plt.figure(figsize=(8, 5))
    plt.axvspan(theta_min, theta_max, alpha=0.15, label="detector window")
    plt.plot(theta, profile_from_map(I_point), linewidth=2, label="point baseline")

    for chi, _, _, _, I_ext in rows:
        plt.plot(theta, profile_from_map(I_ext), linewidth=1.8, label=f"{kind}, chi={chi:.3f}")

    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title(f"{kind.capitalize()} robustness profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outprefix}_{kind}_profiles.png", dpi=200)
    plt.close()

    chis = [r[0] for r in rows]
    Rdet = [r[1] for r in rows]
    Rff = [r[2] for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(chis, Rdet, marker="o", label=r"$R_{\rm det}$")
    plt.plot(chis, Rff, marker="s", label=r"$R_{\rm ff}^{3D}$")
    plt.xlabel(r"$\chi$")
    plt.ylabel("Ratio to point baseline")
    plt.title(f"{kind.capitalize()} robustness vs finite-size parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outprefix}_{kind}_ratios.png", dpi=200)
    plt.close()


def robustness_summary(rows):
    Rdet = np.array([r[1] for r in rows])
    Rff = np.array([r[2] for r in rows])

    return {
        "Rdet_min": float(np.min(Rdet)),
        "Rdet_max": float(np.max(Rdet)),
        "Rff_min": float(np.min(Rff)),
        "Rff_max": float(np.max(Rff)),
        "Rdet_spread_factor": float(np.max(Rdet) / max(np.min(Rdet), 1e-300)),
        "Rff_spread_factor": float(np.max(Rff) / max(np.min(Rff), 1e-300)),
    }


def main():
    theta_min = 0.55
    theta_max = 0.80
    phi_fixed = 0.0

    shell_theta, shell_n, I_point, shell_rows = run_robustness(
        kind="shell",
        chi0=0.50,
        dchi=0.02,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_fixed=phi_fixed
    )

    _, _, _, volume_rows = run_robustness(
        kind="volume",
        chi0=1.20,
        dchi=0.02,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_fixed=phi_fixed
    )

    plot_robustness("shell", shell_theta, I_point, shell_rows, theta_min, theta_max, "figures/task13A")
    plot_robustness("volume", shell_theta, I_point, volume_rows, theta_min, theta_max, "figures/task13A")

    shell_sum = robustness_summary(shell_rows)
    vol_sum = robustness_summary(volume_rows)

    print("\n=== ROBUSTNESS SUMMARY ===")
    print("Shell :", shell_sum)
    print("Volume:", vol_sum)

    print("\nInterpretation rule:")
    print("- If spread factors are ~1 to 3: reasonably robust")
    print("- If spread factors are 10x, 100x, or worse: node-tuned / fragile")
    print("- If one chi gives ~1e-32 and neighbors give ~1e-2 or ~1e-1: reject as non-robust")


if __name__ == "__main__":
    main()