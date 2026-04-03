import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from trajectories import trajectory_constant_velocity, trajectory_sinusoidal
from vacuum_radiation import (
    structural_vacuum_intensity_for_z_motion,
    apply_dc_filter,
    max_nonzero_frequency_intensity,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def nearest_index(xgrid: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(xgrid - x)))


def symmetry_error(theta_grid: np.ndarray, profile: np.ndarray) -> float:
    """
    Compute max symmetry error under theta -> pi - theta.
    """
    mirrored = np.interp(np.pi - theta_grid, theta_grid, profile)
    return float(np.max(np.abs(profile - mirrored)))


def run_constant_velocity(omega_cut: float):
    q = 1.0
    c = 1.0
    v = 0.8

    T = 100.0
    Nt = 2500
    t = np.linspace(-T / 2, T / 2, Nt)
    z, vz = trajectory_constant_velocity(t, v=v)

    omega_grid = np.linspace(-10.0, 10.0, 301)
    theta_grid = np.linspace(0.0, np.pi, 181)

    I = structural_vacuum_intensity_for_z_motion(
        t=t,
        z=z,
        vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q,
        c=c,
        window="hann",
        normalize=True,
    )
    I_filt = apply_dc_filter(omega_grid, I, omega_cut=omega_cut)
    return omega_grid, theta_grid, I_filt


def run_sinusoidal(omega_cut: float):
    q = 1.0
    c = 1.0
    d = 1.0
    omega0 = 1.5

    T = 50.0 * (2.0 * np.pi / omega0)
    Nt = 4000
    t = np.linspace(-T / 2, T / 2, Nt)
    z, vz = trajectory_sinusoidal(t, d=d, omega0=omega0)

    omega_grid = np.linspace(-10.0 * omega0, 10.0 * omega0, 401)
    theta_grid = np.linspace(0.0, np.pi, 181)

    I = structural_vacuum_intensity_for_z_motion(
        t=t,
        z=z,
        vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q,
        c=c,
        window="hann",
        normalize=True,
    )
    I_filt = apply_dc_filter(omega_grid, I, omega_cut=omega_cut)
    return omega_grid, theta_grid, I_filt, omega0


def plot_harmonic_profiles(fig_dir, theta_grid, omega_grid, intensity, omega0, harmonics=(1, 2, 3)):
    plt.figure(figsize=(9, 6))
    for n in harmonics:
        idx_pos = nearest_index(omega_grid, n * omega0)
        plt.plot(theta_grid, intensity[:, idx_pos], label=fr"$\omega={n}\omega_0$")

    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$I_{\rm filt}(\omega,\theta)$")
    plt.title("Task 3: Angular Profiles at Selected Harmonics")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(fig_dir, "task3_harmonic_angular_profiles.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")


def plot_symmetry_check(fig_dir, theta_grid, omega_grid, intensity, omega0, harmonic=1):
    idx = nearest_index(omega_grid, harmonic * omega0)
    profile = intensity[:, idx]
    mirrored = np.interp(np.pi - theta_grid, theta_grid, profile)

    plt.figure(figsize=(9, 6))
    plt.plot(theta_grid, profile, label=fr"$I(\theta)$ at $\omega={harmonic}\omega_0$")
    plt.plot(theta_grid, mirrored, "--", label=r"$I(\pi-\theta)$")
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$I_{\rm filt}$")
    plt.title("Task 3: Forward/Backward Symmetry Check")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(fig_dir, "task3_symmetry_check.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")


def plot_constant_vs_sinusoidal_nonzero(fig_dir, omega_cv, I_cv, omega_sin, I_sin, theta_grid, omega0):
    idx = np.argmin(np.abs(theta_grid - np.pi / 2))

    plt.figure(figsize=(9, 6))
    plt.plot(omega_cv, I_cv[idx], label=r"Constant velocity, filtered, $\theta\approx\pi/2$")
    plt.plot(omega_sin / omega0, I_sin[idx], label=r"Sinusoidal, filtered, $\theta\approx\pi/2$")
    for n in range(-8, 9):
        plt.axvline(n, linestyle="--", linewidth=0.8, alpha=0.2)
    plt.xlabel(r"$\omega$ (constant velocity) / $\omega/\omega_0$ (sinusoidal)")
    plt.ylabel(r"$I_{\rm filt}$")
    plt.title("Task 3: Nonzero-Frequency Benchmark Comparison")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(fig_dir, "task3_nonzero_frequency_comparison.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    ensure_dir(fig_dir)

    omega_cut = 0.3

    omega_cv, theta_cv, I_cv = run_constant_velocity(omega_cut=omega_cut)
    omega_sin, theta_sin, I_sin, omega0 = run_sinusoidal(omega_cut=omega_cut)

    plot_harmonic_profiles(fig_dir, theta_sin, omega_sin, I_sin, omega0, harmonics=(1, 2, 3))
    plot_symmetry_check(fig_dir, theta_sin, omega_sin, I_sin, omega0, harmonic=1)
    plot_constant_vs_sinusoidal_nonzero(fig_dir, omega_cv, I_cv, omega_sin, I_sin, theta_sin, omega0)

    # Quantitative diagnostics
    harmonics_to_check = [1, 2, 3]
    print("\nTask 3 diagnostics:")
    for n in harmonics_to_check:
        idx = nearest_index(omega_sin, n * omega0)
        profile = I_sin[:, idx]

        on_axis_max = max(profile[0], profile[-1])
        off_axis_max = np.max(profile)
        sym_err = symmetry_error(theta_sin, profile)

        print(f"\nHarmonic n = {n}")
        print(f"  on-axis max:      {on_axis_max:.6e}")
        print(f"  off-axis max:     {off_axis_max:.6e}")
        print(f"  symmetry error:   {sym_err:.6e}")
        if off_axis_max > 0:
            print(f"  on/off ratio:     {on_axis_max / off_axis_max:.6e}")

    max_cv_nonzero = max_nonzero_frequency_intensity(omega_cv, I_cv, omega_cut=omega_cut)
    max_sin_nonzero = max_nonzero_frequency_intensity(omega_sin, I_sin, omega_cut=omega_cut)

    print("\nNonzero-frequency benchmark:")
    print(f"  constant velocity max outside DC band: {max_cv_nonzero:.6e}")
    print(f"  sinusoidal max outside DC band:        {max_sin_nonzero:.6e}")

    print("\nTask 3 pass condition:")
    print("1. Harmonic angular profiles vanish on-axis")
    print("2. Profiles are symmetric under theta -> pi - theta")
    print("3. Constant-velocity nonzero-frequency signal remains negligible")
    print("4. Sinusoidal nonzero-frequency harmonics remain strong")


if __name__ == "__main__":
    main()