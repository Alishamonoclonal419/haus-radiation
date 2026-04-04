import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from jax_radiation import make_jax_grids


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def trapz_weights_np(x):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if len(x) > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def make_ff_weights(omega_grid, omega_cut=0.25):
    w = trapz_weights_np(omega_grid)
    mask = (np.abs(omega_grid) > omega_cut).astype(float)
    return w * mask


def make_theta_sector_weights(theta_grid, theta_lo=0.55, theta_hi=0.80):
    w = trapz_weights_np(theta_grid)
    mask = ((theta_grid >= theta_lo) & (theta_grid <= theta_hi)).astype(float)
    return w * mask


def score_from_map(I_map, w_omega, w_theta):
    score_theta = np.sum(I_map * w_omega[None, :], axis=1)
    return float(np.sum(score_theta * w_theta))


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def constrained_three_harmonic_trajectory(t, r2, phi2, r3, phi3, omega1=1.0, vrms_target=None):
    """
    Raw ansatz:
        z_raw = sin(w t) + r2 sin(2 w t + phi2) + r3 sin(3 w t + phi3)

    Then rescale by alpha so RMS(v) = vrms_target.
    """
    t = np.asarray(t, dtype=float)
    omega2 = 2.0 * omega1
    omega3 = 3.0 * omega1

    z_raw = (
        np.sin(omega1 * t)
        + r2 * np.sin(omega2 * t + phi2)
        + r3 * np.sin(omega3 * t + phi3)
    )

    v_raw = (
        omega1 * np.cos(omega1 * t)
        + r2 * omega2 * np.cos(omega2 * t + phi2)
        + r3 * omega3 * np.cos(omega3 * t + phi3)
    )

    if vrms_target is None:
        return z_raw, v_raw, 1.0

    vrms_raw = rms(v_raw)
    if vrms_raw <= 1e-14:
        alpha = 0.0
    else:
        alpha = vrms_target / vrms_raw

    z = alpha * z_raw
    v = alpha * v_raw
    return z, v, alpha


def compute_map_from_zv(grids, z, v, c=1.0):
    t = np.array(grids["t"])
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])
    w_t = np.array(grids["w_t"])

    amps = np.zeros((len(theta_grid), len(omega_grid)), dtype=np.complex128)

    for i, theta in enumerate(theta_grid):
        tau = t - np.cos(theta) * z / c
        phase = np.exp(1j * omega_grid[:, None] * tau[None, :])
        amps[i, :] = np.sum((v * w_t)[None, :] * phase, axis=1)

    I = (
        (omega_grid[None, :] ** 2)
        * (np.sin(theta_grid)[:, None] ** 2)
        * (np.abs(amps) ** 2)
    )
    return I


def main():
    print("Starting Task 11B constrained 3-harmonic optimization...", flush=True)

    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    ensure_dir(fig_dir)

    grids = make_jax_grids(
        T=60.0,
        Nt=1200,
        theta_min=0.0,
        theta_max=np.pi,
        Ntheta=61,
        omega_min=-5.0,
        omega_max=5.0,
        Nomega=121,
        omega_cut=0.25,
    )

    t_grid = np.array(grids["t"])
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])
    w_theta_full = np.array(grids["w_theta"])

    omega1 = 1.0
    omega_cut = 0.25
    theta_lo = 0.55
    theta_hi = 0.80

    w_omega_ff = make_ff_weights(omega_grid, omega_cut=omega_cut)
    w_theta_sector = make_theta_sector_weights(theta_grid, theta_lo=theta_lo, theta_hi=theta_hi)

    # Baseline
    z_base_raw = np.sin(omega1 * t_grid)
    v_base_raw = omega1 * np.cos(omega1 * t_grid)
    vrms_target = rms(v_base_raw)

    z_base, v_base, alpha_base = constrained_three_harmonic_trajectory(
        t_grid, r2=0.0, phi2=0.0, r3=0.0, phi3=0.0, omega1=omega1, vrms_target=vrms_target
    )
    I_base = compute_map_from_zv(grids, z_base, v_base, c=1.0)

    Ssector_base = score_from_map(I_base, w_omega_ff, w_theta_sector)
    Sff_base = score_from_map(I_base, w_omega_ff, w_theta_full)

    print(f"vrms target   = {vrms_target:.6e}", flush=True)
    print(f"Ssector(base) = {Ssector_base:.6e}", flush=True)
    print(f"Sff(base)     = {Sff_base:.6e}", flush=True)
    print(f"Narrow sector = [{theta_lo:.3f}, {theta_hi:.3f}] rad", flush=True)

    # Search grid
    r2_vals = np.linspace(0.0, 0.50, 9)
    r3_vals = np.linspace(0.0, 0.40, 9)
    phi2_vals = np.linspace(0.0, 2*np.pi, 18, endpoint=False)
    phi3_vals = np.linspace(0.0, 2*np.pi, 18, endpoint=False)

    # We will keep best cases only, and also make a reduced summary heatmap:
    # best Rtheta over phi2,phi3 for each (r2,r3)
    best_Rtheta_by_r2r3 = np.full((len(r2_vals), len(r3_vals)), np.inf)
    best_Rff_by_r2r3 = np.full((len(r2_vals), len(r3_vals)), np.inf)

    all_cases = []
    t0 = time.time()

    for i, r2 in enumerate(r2_vals):
        print(f"Scanning r2 = {r2:.3f}", flush=True)
        for j, r3 in enumerate(r3_vals):
            local_best_Rtheta = np.inf
            local_best_Rff = np.inf

            for phi2 in phi2_vals:
                for phi3 in phi3_vals:
                    z, v, alpha = constrained_three_harmonic_trajectory(
                        t_grid, r2=r2, phi2=phi2, r3=r3, phi3=phi3,
                        omega1=omega1, vrms_target=vrms_target
                    )

                    I = compute_map_from_zv(grids, z, v, c=1.0)

                    Ssector = score_from_map(I, w_omega_ff, w_theta_sector)
                    Sff = score_from_map(I, w_omega_ff, w_theta_full)

                    Rtheta = Ssector / Ssector_base
                    Rff = Sff / Sff_base

                    if Rtheta < local_best_Rtheta:
                        local_best_Rtheta = Rtheta
                        local_best_Rff = Rff

                    all_cases.append({
                        "r2": float(r2),
                        "phi2": float(phi2),
                        "r3": float(r3),
                        "phi3": float(phi3),
                        "alpha": float(alpha),
                        "Ssector": float(Ssector),
                        "Sff": float(Sff),
                        "Rtheta": float(Rtheta),
                        "Rff": float(Rff),
                    })

            best_Rtheta_by_r2r3[i, j] = local_best_Rtheta
            best_Rff_by_r2r3[i, j] = local_best_Rff

    t1 = time.time()
    print(f"Total optimization scan time: {t1 - t0:.2f} s", flush=True)

    all_cases = sorted(all_cases, key=lambda d: d["Rtheta"])
    top = all_cases[:10]

    print("\nTop 10 constrained 3-harmonic candidates:", flush=True)
    for k, case in enumerate(top, start=1):
        print(
            f"{k}. r2={case['r2']:.3f}, "
            f"phi2={case['phi2']:.3f}, "
            f"r3={case['r3']:.3f}, "
            f"phi3={case['phi3']:.3f}, "
            f"alpha={case['alpha']:.6f}, "
            f"Ssector={case['Ssector']:.6e}, "
            f"Rtheta={case['Rtheta']:.6e}, "
            f"(1-Rtheta)={1.0 - case['Rtheta']:.6e}, "
            f"Sff={case['Sff']:.6e}, "
            f"Rff={case['Rff']:.6e}",
            flush=True
        )

    best = top[0]
    z_best, v_best, alpha_best = constrained_three_harmonic_trajectory(
        t_grid,
        r2=best["r2"], phi2=best["phi2"],
        r3=best["r3"], phi3=best["phi3"],
        omega1=omega1, vrms_target=vrms_target
    )
    I_best = compute_map_from_zv(grids, z_best, v_best, c=1.0)

    extent = [omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]]

    # Heatmap 1: best directional suppression over phases for each (r2,r3)
    sector_supp = 1.0 - best_Rtheta_by_r2r3
    vmax1 = max(0.01, float(np.max(sector_supp)))

    plt.figure(figsize=(8, 6))
    plt.imshow(
        sector_supp.T,
        extent=[r2_vals[0], r2_vals[-1], r3_vals[0], r3_vals[-1]],
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=vmax1,
    )
    plt.colorbar(label=r"$1 - R_\Theta$")
    plt.xlabel(r"$r_2 = a_2/a_1$")
    plt.ylabel(r"$r_3 = a_3/a_1$")
    plt.title("Task 11B: Best narrow-sector suppression over phase choices")
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_sector_suppression.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Heatmap 2: best total-radiation ratio paired to best local directional score
    plt.figure(figsize=(8, 6))
    plt.imshow(
        best_Rff_by_r2r3.T,
        extent=[r2_vals[0], r2_vals[-1], r3_vals[0], r3_vals[-1]],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(label=r"$R_{\rm ff}=S_{\rm ff}/S_{\rm ff}^{\rm base}$")
    plt.xlabel(r"$r_2 = a_2/a_1$")
    plt.ylabel(r"$r_3 = a_3/a_1$")
    plt.title("Task 11B: Total-radiation ratio at best local phase choice")
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_total_ratio.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Baseline map
    plt.figure(figsize=(10, 5))
    plt.imshow(I_base, extent=extent, origin="lower", aspect="auto")
    plt.colorbar(label=r"$I(\omega,\theta)$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\theta$")
    plt.title("Task 11B: Constrained baseline map")
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_baseline_map.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Best map
    plt.figure(figsize=(10, 5))
    plt.imshow(I_best, extent=extent, origin="lower", aspect="auto")
    plt.colorbar(label=r"$I(\omega,\theta)$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\theta$")
    plt.title("Task 11B: Constrained best-candidate map")
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_best_candidate_map.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Angular profile comparison
    Ibase_profile = np.sum(I_base * w_omega_ff[None, :], axis=1)
    Ibest_profile = np.sum(I_best * w_omega_ff[None, :], axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, Ibase_profile, label="Baseline angular profile")
    plt.plot(theta_grid, Ibest_profile, label="Best 3-harmonic candidate")
    plt.axvspan(theta_lo, theta_hi, alpha=0.15, label="narrow sector")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Finite-frequency angular intensity")
    plt.title("Task 11B: Constrained angular-profile comparison")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_sector_profile_comparison.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Trajectory comparison
    plt.figure(figsize=(10, 5))
    plt.plot(t_grid, z_base, label="Baseline sinusoid")
    plt.plot(t_grid, z_best, label="Best 3-harmonic candidate")
    plt.xlabel("t")
    plt.ylabel("z(t)")
    plt.title("Task 11B: Constrained trajectory comparison")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, "task11B_three_harmonic_trajectory_comparison.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    print("\nTask 11B complete.", flush=True)


if __name__ == "__main__":
    main()