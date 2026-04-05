import os
import sys
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


def rms_speed(vx, vy, vz):
    return float(np.sqrt(np.mean(vx**2 + vy**2 + vz**2)))


def normalize_to_vrms(vx, vy, vz, x, y, z, vrms_target):
    current = rms_speed(vx, vy, vz)
    alpha = 0.0 if current <= 1e-14 else vrms_target / current
    return alpha * x, alpha * y, alpha * z, alpha * vx, alpha * vy, alpha * vz, alpha


def trajectory_baseline_1d(t, omega=1.0):
    x = np.sin(omega * t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    vx = omega * np.cos(omega * t)
    vy = np.zeros_like(t)
    vz = np.zeros_like(t)
    return x, y, z, vx, vy, vz


def trajectory_planar_circular(t, omega=1.0):
    x = np.cos(omega * t)
    y = np.sin(omega * t)
    z = np.zeros_like(t)
    vx = -omega * np.sin(omega * t)
    vy =  omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return x, y, z, vx, vy, vz


def trajectory_true_3d_helix(t, omega=1.0, az_ratio=0.25, ratio_z=1.0, phi_z=np.pi):
    """
    True bounded 3D helix-like motion:
        x = cos(omega t)
        y = sin(omega t)
        z = az_ratio * sin(ratio_z * omega * t + phi_z)
    """
    wz = ratio_z * omega

    x = np.cos(omega * t)
    y = np.sin(omega * t)
    z = az_ratio * np.sin(wz * t + phi_z)

    vx = -omega * np.sin(omega * t)
    vy =  omega * np.cos(omega * t)
    vz = az_ratio * wz * np.cos(wz * t + phi_z)

    return x, y, z, vx, vy, vz


def compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=0.0, c=1.0):
    t = np.array(grids["t"])
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])
    w_t = np.array(grids["w_t"])

    I = np.zeros((len(theta_grid), len(omega_grid)), dtype=float)

    cos_phi = np.cos(phi_obs)
    sin_phi = np.sin(phi_obs)

    for i, theta in enumerate(theta_grid):
        nx = np.sin(theta) * cos_phi
        ny = np.sin(theta) * sin_phi
        nz = np.cos(theta)

        ndotr = nx * x + ny * y + nz * z
        tau = t - ndotr / c

        ndotv = nx * vx + ny * vy + nz * vz
        vpx = vx - nx * ndotv
        vpy = vy - ny * ndotv
        vpz = vz - nz * ndotv

        phase = np.exp(1j * omega_grid[:, None] * tau[None, :])

        Ax = np.sum((vpx * w_t)[None, :] * phase, axis=1)
        Ay = np.sum((vpy * w_t)[None, :] * phase, axis=1)
        Az = np.sum((vpz * w_t)[None, :] * phase, axis=1)

        I[i, :] = (omega_grid**2) * (np.abs(Ax)**2 + np.abs(Ay)**2 + np.abs(Az)**2)

    return I


def compute_total_ff_3d(grids, x, y, z, vx, vy, vz, omega_cut=0.25, n_phi=36, c=1.0):
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])

    w_omega_ff = make_ff_weights(omega_grid, omega_cut=omega_cut)
    w_theta = trapz_weights_np(theta_grid) * np.sin(theta_grid)

    phi_grid = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    total = 0.0
    for phi_obs in phi_grid:
        I = compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=phi_obs, c=c)
        ang_freq = np.sum(I * w_omega_ff[None, :], axis=1)
        total += dphi * np.sum(ang_freq * w_theta)

    return float(total)


def compute_detector_window_ff_3d(
    grids, x, y, z, vx, vy, vz,
    theta_lo=0.55, theta_hi=0.80,
    phi_lo=-0.35, phi_hi=0.35,
    omega_cut=0.25,
    n_phi=48,
    c=1.0
):
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])

    w_omega_ff = make_ff_weights(omega_grid, omega_cut=omega_cut)
    w_theta = trapz_weights_np(theta_grid) * np.sin(theta_grid)
    theta_mask = ((theta_grid >= theta_lo) & (theta_grid <= theta_hi)).astype(float)
    w_theta_win = w_theta * theta_mask

    phi_grid = np.linspace(phi_lo, phi_hi, n_phi)
    w_phi = trapz_weights_np(phi_grid)

    total = 0.0
    for phi_obs, wp in zip(phi_grid, w_phi):
        I = compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=phi_obs, c=c)
        ang_freq = np.sum(I * w_omega_ff[None, :], axis=1)
        total += wp * np.sum(ang_freq * w_theta_win)

    return float(total)


def save_fixed_phi_plots(fig_dir, tag, grids, I_case, I_base, theta_lo, theta_hi):
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])
    w_omega_ff = make_ff_weights(omega_grid, omega_cut=0.25)

    extent = [omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]]

    plt.figure(figsize=(10, 5))
    plt.imshow(I_case, extent=extent, origin="lower", aspect="auto")
    plt.colorbar(label=r"$I(\omega,\theta;\phi=0)$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\theta$")
    plt.title(f"Task 12B: {tag} fixed-$\\phi$ map")
    plt.tight_layout()
    out = os.path.join(fig_dir, f"task12B_{tag}_map_phi0.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    prof_case = np.sum(I_case * w_omega_ff[None, :], axis=1)
    prof_base = np.sum(I_base * w_omega_ff[None, :], axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, prof_base, label="baseline 1D, fixed φ")
    plt.plot(theta_grid, prof_case, label=f"{tag}, fixed φ")
    plt.axvspan(theta_lo, theta_hi, alpha=0.15, label="θ window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Finite-frequency angular intensity")
    plt.title(f"Task 12B: {tag} fixed-$\\phi$ profile")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, f"task12B_{tag}_profile_phi0.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)


def main():
    print("Starting Task 12B true 3D helix benchmark...", flush=True)

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

    theta_lo = 0.55
    theta_hi = 0.80
    phi_lo = -0.35
    phi_hi = 0.35
    omega_cut = 0.25

    # Baseline 1D
    x0, y0, z0, vx0, vy0, vz0 = trajectory_baseline_1d(t_grid, omega=1.0)
    vrms_target = rms_speed(vx0, vy0, vz0)
    x0, y0, z0, vx0, vy0, vz0, alpha0 = normalize_to_vrms(vx0, vy0, vz0, x0, y0, z0, vrms_target)

    Iphi0_base = compute_map_3d_vector_fixed_phi(grids, x0, y0, z0, vx0, vy0, vz0, phi_obs=0.0)
    Sdet_base = compute_detector_window_ff_3d(
        grids, x0, y0, z0, vx0, vy0, vz0,
        theta_lo=theta_lo, theta_hi=theta_hi,
        phi_lo=phi_lo, phi_hi=phi_hi,
        omega_cut=omega_cut, n_phi=48
    )
    Sff_base = compute_total_ff_3d(
        grids, x0, y0, z0, vx0, vy0, vz0,
        omega_cut=omega_cut, n_phi=36
    )

    print(f"Baseline detector-window score = {Sdet_base:.6e}", flush=True)
    print(f"Baseline full 3D Sff          = {Sff_base:.6e}", flush=True)
    print(f"vrms target                   = {vrms_target:.6e}", flush=True)

    # Planar circular reference
    xc, yc, zc, vxc, vyc, vzc = trajectory_planar_circular(t_grid, omega=1.0)
    xc, yc, zc, vxc, vyc, vzc, alphac = normalize_to_vrms(vxc, vyc, vzc, xc, yc, zc, vrms_target)

    Iphi0_circ = compute_map_3d_vector_fixed_phi(grids, xc, yc, zc, vxc, vyc, vzc, phi_obs=0.0)
    Sdet_circ = compute_detector_window_ff_3d(
        grids, xc, yc, zc, vxc, vyc, vzc,
        theta_lo=theta_lo, theta_hi=theta_hi,
        phi_lo=phi_lo, phi_hi=phi_hi,
        omega_cut=omega_cut, n_phi=48
    )
    Sff_circ = compute_total_ff_3d(
        grids, xc, yc, zc, vxc, vyc, vzc,
        omega_cut=omega_cut, n_phi=36
    )

    print(
        f"planar_circular  Sdet={Sdet_circ:.6e}  Rdet={Sdet_circ/Sdet_base:.6e}  "
        f"Sff3D={Sff_circ:.6e}  Rff3D={Sff_circ/Sff_base:.6e}  alpha={alphac:.6e}",
        flush=True
    )

    save_fixed_phi_plots(fig_dir, "planar_circular", grids, Iphi0_circ, Iphi0_base, theta_lo, theta_hi)

    # True 3D helix family
    az_vals = [0.15, 0.30, 0.50]
    ratio_vals = [1.0, 2.0]
    phi_vals = [0.0, 0.5*np.pi, np.pi]

    results = []

    for az_ratio in az_vals:
        for ratio_z in ratio_vals:
            for phi_z in phi_vals:
                tag = f"helix_az{az_ratio:.2f}_rz{ratio_z:.1f}_ph{phi_z/np.pi:.2f}pi".replace(".", "p")

                x, y, z, vx, vy, vz = trajectory_true_3d_helix(
                    t_grid, omega=1.0, az_ratio=az_ratio, ratio_z=ratio_z, phi_z=phi_z
                )
                x, y, z, vx, vy, vz, alpha = normalize_to_vrms(vx, vy, vz, x, y, z, vrms_target)

                Iphi0 = compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=0.0)

                Sdet = compute_detector_window_ff_3d(
                    grids, x, y, z, vx, vy, vz,
                    theta_lo=theta_lo, theta_hi=theta_hi,
                    phi_lo=phi_lo, phi_hi=phi_hi,
                    omega_cut=omega_cut, n_phi=48
                )
                Sff = compute_total_ff_3d(
                    grids, x, y, z, vx, vy, vz,
                    omega_cut=omega_cut, n_phi=36
                )

                Rdet = Sdet / Sdet_base
                Rff = Sff / Sff_base

                results.append({
                    "tag": tag,
                    "az_ratio": az_ratio,
                    "ratio_z": ratio_z,
                    "phi_z": phi_z,
                    "alpha": alpha,
                    "Sdet": Sdet,
                    "Rdet": Rdet,
                    "Sff": Sff,
                    "Rff": Rff,
                    "Iphi0": Iphi0,
                })

                print(
                    f"{tag:28s}  Sdet={Sdet:.6e}  Rdet={Rdet:.6e}  "
                    f"Sff3D={Sff:.6e}  Rff3D={Rff:.6e}  alpha={alpha:.6e}",
                    flush=True
                )

    # Best by detector score
    results_sorted = sorted(results, key=lambda d: d["Rdet"])
    best = results_sorted[0]

    print("\nTop 5 true-3D helix cases by detector suppression:", flush=True)
    for i, d in enumerate(results_sorted[:5], start=1):
        print(
            f"{i}. az_ratio={d['az_ratio']:.2f}, ratio_z={d['ratio_z']:.1f}, "
            f"phi_z={d['phi_z']:.3f}, Rdet={d['Rdet']:.6e}, Rff3D={d['Rff']:.6e}",
            flush=True
        )

    save_fixed_phi_plots(fig_dir, best["tag"], grids, best["Iphi0"], Iphi0_base, theta_lo, theta_hi)

    # Summary plot: planar circular vs best true-3D
    labels = ["planar circular", "best true 3D"]
    Rdet_vals = [Sdet_circ / Sdet_base, best["Rdet"]]
    Rff_vals = [Sff_circ / Sff_base, best["Rff"]]

    xloc = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(xloc - width/2, Rdet_vals, width, label=r"$R_{\rm det}$")
    plt.bar(xloc + width/2, Rff_vals, width, label=r"$R_{\rm ff}^{3D}$")
    plt.xticks(xloc, labels)
    plt.ylabel("Ratio to 1D baseline")
    plt.title("Task 12B: planar circular vs best true 3D helix")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, "task12B_true3d_summary.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    # Heatmap over az_ratio and phi_z for ratio_z=1 and ratio_z=2 separately
    for rz in ratio_vals:
        subset = [d for d in results if abs(d["ratio_z"] - rz) < 1e-12]
        phi_unique = sorted(set(d["phi_z"] for d in subset))
        az_unique = sorted(set(d["az_ratio"] for d in subset))

        Z = np.zeros((len(az_unique), len(phi_unique)))
        for ia, az in enumerate(az_unique):
            for ip, ph in enumerate(phi_unique):
                match = [d for d in subset if abs(d["az_ratio"]-az)<1e-12 and abs(d["phi_z"]-ph)<1e-12]
                Z[ia, ip] = match[0]["Rdet"]

        plt.figure(figsize=(7, 5))
        plt.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[0, len(phi_unique)-1, az_unique[0], az_unique[-1]]
        )
        plt.colorbar(label=r"$R_{\rm det}$")
        plt.xticks(range(len(phi_unique)), [f"{ph/np.pi:.2f}π" for ph in phi_unique])
        plt.xlabel(r"$\phi_z$")
        plt.ylabel(r"$A_z/R$")
        plt.title(f"Task 12B: detector ratio heatmap for $\\Omega/\\omega={rz:.1f}$")
        plt.tight_layout()
        out = os.path.join(fig_dir, f"task12B_heatmap_rz_{rz:.1f}.png".replace(".", "p"))
        plt.savefig(out, dpi=220)
        plt.close()
        print(f"[Saved] {out}", flush=True)

    print("\nTask 12B complete.", flush=True)


if __name__ == "__main__":
    main()