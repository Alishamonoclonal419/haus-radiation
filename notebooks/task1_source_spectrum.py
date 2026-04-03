import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from trajectories import trajectory_constant_velocity, trajectory_sinusoidal
from source_spectrum import compute_Jz_kw, expected_constant_velocity_center


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def constant_velocity_heatmap(fig_dir: str, window: str = "rectangular") -> None:
    """
    Source-spectrum test only.
    Expected ridge near omega = v k_z.
    """
    q = 1.0
    v = 0.8
    T = 100.0
    Nt = 4000

    t = np.linspace(-T / 2, T / 2, Nt)
    z, vz = trajectory_constant_velocity(t, v=v)

    kz_grid = np.linspace(-10.0, 10.0, 401)
    omega_grid = np.linspace(-10.0, 10.0, 401)

    J = compute_Jz_kw(
        t, z, vz, kz_grid, omega_grid,
        q=q, window=window, normalize=True
    )
    J_abs = np.abs(J)

    plt.figure(figsize=(8, 6))
    extent = [omega_grid[0], omega_grid[-1], kz_grid[0], kz_grid[-1]]
    plt.imshow(
        J_abs,
        extent=extent,
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(label=r"$|J_z(k_z,\omega)|$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$k_z$")
    plt.title(f"Task 1: Constant-Velocity Source Spectrum ({window} window)")

    kz_line = np.linspace(kz_grid[0], kz_grid[-1], 400)
    omega_line = expected_constant_velocity_center(kz_line, v=v)
    plt.plot(omega_line, kz_line, "--", linewidth=1.5, label=fr"$\omega = v k_z,\ v={v}$")
    plt.legend()

    outpath = os.path.join(fig_dir, f"task1_constant_velocity_heatmap_{window}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"[Saved] {outpath}")


def constant_velocity_width_check(fig_dir: str, window: str = "rectangular") -> None:
    """
    Increasing total time window should sharpen the ridge in omega.
    """
    q = 1.0
    v = 0.8
    kz_fixed = np.array([3.0])
    omega_grid = np.linspace(-5.0, 5.0, 2501)
    time_windows = [20.0, 50.0, 100.0]
    Nt = 4000

    plt.figure(figsize=(8, 5))
    for T in time_windows:
        t = np.linspace(-T / 2, T / 2, Nt)
        z, vz = trajectory_constant_velocity(t, v=v)
        J = compute_Jz_kw(
            t, z, vz, kz_fixed, omega_grid,
            q=q, window=window, normalize=True
        )
        plt.plot(omega_grid, np.abs(J[0]), label=fr"$T={T}$")

    omega_expected = v * kz_fixed[0]
    plt.axvline(omega_expected, linestyle="--", linewidth=1.2, label=fr"expected center = {omega_expected:.2f}")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$|J_z|$")
    plt.title(f"Task 1: Constant-Velocity Peak Narrowing ({window} window)")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(fig_dir, f"task1_constant_velocity_widthcheck_{window}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"[Saved] {outpath}")


def sinusoidal_spectrum(fig_dir: str, window: str = "hann") -> None:
    """
    Source-spectrum test for oscillatory motion.
    Expected harmonic peaks near integer multiples of omega0.
    """
    q = 1.0
    d = 1.0
    omega0 = 1.5
    T = 40.0 * (2.0 * np.pi / omega0)
    Nt = 8000

    t = np.linspace(-T / 2, T / 2, Nt)
    z, vz = trajectory_sinusoidal(t, d=d, omega0=omega0)

    kz_values = np.array([0.5, 1.0, 2.0])
    omega_grid = np.linspace(-12.0 * omega0, 12.0 * omega0, 4001)

    J = compute_Jz_kw(
        t, z, vz, kz_values, omega_grid,
        q=q, window=window, normalize=True
    )

    plt.figure(figsize=(9, 6))
    x = omega_grid / omega0
    for i, kz in enumerate(kz_values):
        plt.plot(x, np.abs(J[i]), label=fr"$k_z={kz}$")

    max_harmonic = 8
    for m in range(-max_harmonic, max_harmonic + 1):
        plt.axvline(m, linestyle="--", linewidth=0.8, alpha=0.35)

    plt.xlabel(r"$\omega/\omega_0$")
    plt.ylabel(r"$|J_z(k_z,\omega)|$")
    plt.title(f"Task 1: Sinusoidal Source Spectrum ({window} window)")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(fig_dir, f"task1_sinusoidal_spectrum_{window}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"[Saved] {outpath}")


def main() -> None:
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    ensure_dir(fig_dir)

    constant_velocity_heatmap(fig_dir, window="rectangular")
    constant_velocity_heatmap(fig_dir, window="hann")
    constant_velocity_width_check(fig_dir, window="rectangular")
    sinusoidal_spectrum(fig_dir, window="hann")


if __name__ == "__main__":
    main()