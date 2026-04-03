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
    spectral_far_field_intensity_for_z_motion,
    max_nonzero_frequency_intensity,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_map(omega_grid, theta_grid, intensity, title, outpath):
    plt.figure(figsize=(9, 6))
    extent = [omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]]
    plt.imshow(intensity, extent=extent, origin="lower", aspect="auto")
    plt.colorbar(label=r"$I(\omega,\theta)$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")


def run_constant_velocity(fig_dir: str):
    q = 1.0
    c = 1.0
    v = 0.8

    T = 100.0
    Nt = 2500
    t = np.linspace(-T / 2, T / 2, Nt)
    z, vz = trajectory_constant_velocity(t, v=v)

    omega_grid = np.linspace(-10.0, 10.0, 301)
    theta_grid = np.linspace(0.0, np.pi, 181)

    I_struct = structural_vacuum_intensity_for_z_motion(
        t=t, z=z, vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q, c=c,
        window="hann", normalize=True,
    )

    I_spec = spectral_far_field_intensity_for_z_motion(
        t=t, z=z, vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q, c=c,
        window="hann", normalize=True,
    )

    plot_map(
        omega_grid, theta_grid, I_struct,
        "Task 4: Structural Proxy — Constant Velocity",
        os.path.join(fig_dir, "task4_constant_velocity_structural.png")
    )
    plot_map(
        omega_grid, theta_grid, I_spec,
        "Task 4: Spectral Far-Field Observable — Constant Velocity",
        os.path.join(fig_dir, "task4_constant_velocity_spectral.png")
    )

    return omega_grid, theta_grid, I_struct, I_spec


def run_sinusoidal(fig_dir: str):
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

    I_struct = structural_vacuum_intensity_for_z_motion(
        t=t, z=z, vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q, c=c,
        window="hann", normalize=True,
    )

    I_spec = spectral_far_field_intensity_for_z_motion(
        t=t, z=z, vz=vz,
        omega_grid=omega_grid,
        theta_grid=theta_grid,
        q=q, c=c,
        window="hann", normalize=True,
    )

    plot_map(
        omega_grid, theta_grid, I_struct,
        "Task 4: Structural Proxy — Sinusoidal Motion",
        os.path.join(fig_dir, "task4_sinusoidal_structural.png")
    )
    plot_map(
        omega_grid, theta_grid, I_spec,
        "Task 4: Spectral Far-Field Observable — Sinusoidal Motion",
        os.path.join(fig_dir, "task4_sinusoidal_spectral.png")
    )

    return omega_grid, theta_grid, I_struct, I_spec, omega0


def compare_fixed_angle(fig_dir, omega_cv, Icv_struct, Icv_spec, omega_sin, Isin_struct, Isin_spec, theta_grid, omega0):
    idx = np.argmin(np.abs(theta_grid - np.pi / 2))

    plt.figure(figsize=(10, 6))
    plt.plot(omega_cv, Icv_struct[idx], label="Const velocity, structural")
    plt.plot(omega_cv, Icv_spec[idx], label="Const velocity, spectral")
    plt.plot(omega_sin / omega0, Isin_struct[idx], label="Sinusoidal, structural")
    plt.plot(omega_sin / omega0, Isin_spec[idx], label="Sinusoidal, spectral")
    for n in range(-8, 9):
        plt.axvline(n, linestyle="--", linewidth=0.8, alpha=0.2)
    plt.xlabel(r"$\omega$ (constant velocity) / $\omega/\omega_0$ (sinusoidal)")
    plt.ylabel(r"$I$")
    plt.title("Task 4: Structural vs Spectral Observable at Fixed Angle")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(fig_dir, "task4_fixed_angle_comparison.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    ensure_dir(fig_dir)

    omega_cv, theta_grid, Icv_struct, Icv_spec = run_constant_velocity(fig_dir)
    omega_sin, theta_grid2, Isin_struct, Isin_spec, omega0 = run_sinusoidal(fig_dir)

    compare_fixed_angle(
        fig_dir,
        omega_cv, Icv_struct, Icv_spec,
        omega_sin, Isin_struct, Isin_spec,
        theta_grid, omega0
    )

    omega_cut = 0.3

    print("\nTask 4 diagnostics:")
    print(f"Constant velocity structural max outside DC band: {max_nonzero_frequency_intensity(omega_cv, Icv_struct, omega_cut):.6e}")
    print(f"Constant velocity spectral   max outside DC band: {max_nonzero_frequency_intensity(omega_cv, Icv_spec, omega_cut):.6e}")
    print(f"Sinusoidal structural max outside DC band:       {max_nonzero_frequency_intensity(omega_sin, Isin_struct, omega_cut):.6e}")
    print(f"Sinusoidal spectral   max outside DC band:       {max_nonzero_frequency_intensity(omega_sin, Isin_spec, omega_cut):.6e}")

    print("\nTask 4 pass condition:")
    print("1. The spectral observable suppresses low-frequency contamination more naturally than the structural proxy")
    print("2. The sinusoidal case retains harmonic finite-frequency structure")
    print("3. The angular suppression on-axis remains intact")


if __name__ == "__main__":
    main()