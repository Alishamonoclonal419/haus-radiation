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


def make_theta_weights_with_sin(theta_grid):
    w_theta = trapz_weights_np(theta_grid)
    return w_theta * np.sin(theta_grid)


def make_phi_weights(phi_grid):
    return trapz_weights_np(phi_grid)


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


def trajectory_circular(t, omega=1.0):
    x = np.cos(omega * t)
    y = np.sin(omega * t)
    z = np.zeros_like(t)
    vx = -omega * np.sin(omega * t)
    vy =  omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return x, y, z, vx, vy, vz


def trajectory_elliptical(t, a=1.0, b=0.5, omega=1.0):
    x = a * np.cos(omega * t)
    y = b * np.sin(omega * t)
    z = np.zeros_like(t)
    vx = -a * omega * np.sin(omega * t)
    vy =  b * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return x, y, z, vx, vy, vz


def trajectory_lissajous(t, Ax=1.0, Ay=0.35, omega=1.0, phi=np.pi):
    x = Ax * np.sin(omega * t)
    y = Ay * np.sin(2 * omega * t + phi)
    z = np.zeros_like(t)
    vx = Ax * omega * np.cos(omega * t)
    vy = Ay * 2 * omega * np.cos(2 * omega * t + phi)
    vz = np.zeros_like(t)
    return x, y, z, vx, vy, vz


def compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=0.0, c=1.0):
    """
    Returns I(theta, omega) for a fixed azimuth phi_obs, using the
    physically correct transverse VECTOR amplitude:
        A(omega, n) = ∫ dt v_perp(t) exp(i omega tau)
    with
        v_perp = v - n (n·v)
    and then
        I ~ omega^2 |A|^2
    """
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
    """
    Full 3D finite-frequency total:
        S_ff = ∫ dphi ∫ dtheta sin(theta) ∫_{|omega|>omega_cut} dω I(ω,theta,phi)
    """
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])

    w_omega_ff = make_ff_weights(omega_grid, omega_cut=omega_cut)
    w_theta_sin = make_theta_weights_with_sin(theta_grid)

    phi_grid = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = make_phi_weights(np.r_[phi_grid, 2 * np.pi])[:-1]  # periodic approximation
    if len(w_phi) != len(phi_grid):
        w_phi = np.full_like(phi_grid, 2 * np.pi / len(phi_grid), dtype=float)

    total = 0.0
    for phi_obs, wp in zip(phi_grid, w_phi):
        I = compute_map_3d_vector_fixed_phi(grids, x, y, z, vx, vy, vz, phi_obs=phi_obs, c=c)
        ang_freq = np.sum(I * w_omega_ff[None, :], axis=1)
        total += wp * np.sum(ang_freq * w_theta_sin)

    return float(total)


def compute_detector_window_ff_3d(
    grids, x, y, z, vx, vy, vz,
    theta_lo=0.55, theta_hi=0.80,
    phi_lo=-0.35, phi_hi=0.35,
    omega_cut=0.25,
    n_phi=48,
    c=1.0
):
    """
    Detector-window score:
        theta in [theta_lo, theta_hi]
        phi in [phi_lo, phi_hi]
    centered around phi=0 for now.
    """
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


def plot_fixed_phi_map_and_profile(fig_dir, name, grids, Iphi0, Iphi0_base, theta_lo, theta_hi):
    theta_grid = np.array(grids["theta_grid"])
    omega_grid = np.array(grids["omega_grid"])
    w_omega_ff = make_ff_weights(omega_grid, omega_cut=0.25)

    extent = [omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]]

    plt.figure(figsize=(10, 5))
    plt.imshow(Iphi0, extent=extent, origin="lower", aspect="auto")
    plt.colorbar(label=r"$I(\omega,\theta;\phi=0)$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\theta$")
    plt.title(f"Task 12A-fix: {name} fixed-$\\phi$ map")
    plt.tight_layout()
    out = os.path.join(fig_dir, f"task12A_fix_{name}_map_phi0.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    prof = np.sum(Iphi0 * w_omega_ff[None, :], axis=1)
    prof_base = np.sum(Iphi0_base * w_omega_ff[None, :], axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, prof_base, label="baseline 1D, fixed φ")
    plt.plot(theta_grid, prof, label=f"{name}, fixed φ")
    plt.axvspan(theta_lo, theta_hi, alpha=0.15, label="θ window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Finite-frequency angular intensity")
    plt.title(f"Task 12A-fix: {name} fixed-$\\phi$ profile")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, f"task12A_fix_{name}_profile_phi0.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)


def main():
    print("Starting Task 12A-fix 3D vector benchmark...", flush=True)

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

    cases = []

    # Circular
    x, y, z, vx, vy, vz = trajectory_circular(t_grid, omega=1.0)
    x, y, z, vx, vy, vz, alpha = normalize_to_vrms(vx, vy, vz, x, y, z, vrms_target)
    cases.append(("circular", x, y, z, vx, vy, vz, alpha))

    # Elliptical
    x, y, z, vx, vy, vz = trajectory_elliptical(t_grid, a=1.0, b=0.5, omega=1.0)
    x, y, z, vx, vy, vz, alpha = normalize_to_vrms(vx, vy, vz, x, y, z, vrms_target)
    cases.append(("elliptical", x, y, z, vx, vy, vz, alpha))

    # Lissajous
    x, y, z, vx, vy, vz = trajectory_lissajous(t_grid, Ax=1.0, Ay=0.35, omega=1.0, phi=np.pi)
    x, y, z, vx, vy, vz, alpha = normalize_to_vrms(vx, vy, vz, x, y, z, vrms_target)
    cases.append(("lissajous", x, y, z, vx, vy, vz, alpha))

    results = []

    for name, x, y, z, vx, vy, vz, alpha in cases:
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

        results.append((name, Sdet, Rdet, Sff, Rff, alpha))

        print(
            f"{name:12s}  Sdet={Sdet:.6e}  Rdet={Rdet:.6e}  "
            f"Sff3D={Sff:.6e}  Rff3D={Rff:.6e}  alpha={alpha:.6e}",
            flush=True
        )

        plot_fixed_phi_map_and_profile(fig_dir, name, grids, Iphi0, Iphi0_base, theta_lo, theta_hi)

    # Summary plot
    names = [r[0] for r in results]
    Rdet_vals = [r[2] for r in results]
    Rff_vals = [r[4] for r in results]

    xloc = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(xloc - width/2, Rdet_vals, width, label=r"$R_{\rm det}$")
    plt.bar(xloc + width/2, Rff_vals, width, label=r"$R_{\rm ff}^{3D}$")
    plt.xticks(xloc, names)
    plt.ylabel("Ratio to 1D baseline")
    plt.title("Task 12A-fix: 3D vector benchmark summary")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, "task12A_fix_3d_summary.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[Saved] {out}", flush=True)

    print("\nTask 12A-fix complete.", flush=True)


if __name__ == "__main__":
    main()