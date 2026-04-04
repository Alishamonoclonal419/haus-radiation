import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from trajectories import trajectory_constant_velocity
from medium_radiation import spectral_far_field_intensity_nondispersive_z_motion


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalized(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    m = np.max(arr)
    if m <= 0:
        return arr.copy()
    return arr / m


def cherenkov_angle(n_medium: float, beta: float):
    x = n_medium * beta
    if x <= 1.0:
        return None
    return float(np.arccos(1.0 / x))


def integrated_nonzero_frequency_strength(
    intensity_map: np.ndarray,
    omega_grid: np.ndarray,
    omega_cut: float,
) -> np.ndarray:
    mask = np.abs(omega_grid) > omega_cut
    if not np.any(mask):
        raise ValueError("omega_cut removed all frequencies.")
    return np.trapezoid(intensity_map[:, mask], omega_grid[mask], axis=1)


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    ensure_dir(fig_dir)

    c = 1.0
    q = 1.0
    n_medium = 1.5
    beta_list = [0.60, 0.68, 0.80]

    # Keep this genuinely cheap
    T = 60.0
    Nt = 1800
    t = np.linspace(-T / 2, T / 2, Nt)

    theta_grid = np.linspace(0.0, np.pi / 2, 81)
    omega_grid = np.linspace(-4.0, 4.0, 201)
    omega_cut = 0.25

    profiles = {}
    peak_report = []

    saved_map = None
    saved_beta = 0.80

    print("Starting Task 8 fast...")
    print(f"n = {n_medium}, beta list = {beta_list}")
    print(f"Nt = {Nt}, Ntheta = {len(theta_grid)}, Nomega = {len(omega_grid)}")

    for beta in beta_list:
        print(f"\nComputing beta = {beta:.2f} ...")
        v = beta * c
        z, vz = trajectory_constant_velocity(t, v=v)

        I_med = spectral_far_field_intensity_nondispersive_z_motion(
            t=t,
            z=z,
            vz=vz,
            omega_grid=omega_grid,
            theta_grid=theta_grid,
            n_medium=n_medium,
            q=q,
            c=c,
            window="hann",
            normalize=True,
        )

        S_theta = integrated_nonzero_frequency_strength(
            intensity_map=I_med,
            omega_grid=omega_grid,
            omega_cut=omega_cut,
        )
        profiles[beta] = S_theta

        theta_pred = cherenkov_angle(n_medium=n_medium, beta=beta)
        theta_num = float(theta_grid[np.argmax(S_theta)])
        peak_report.append((beta, theta_pred, theta_num))

        if abs(beta - saved_beta) < 1e-12:
            saved_map = I_med.copy()

        print(f"Finished beta = {beta:.2f}")

    # Angular strength plot
    plt.figure(figsize=(9, 6))
    for beta in beta_list:
        plt.plot(theta_grid, normalized(profiles[beta]), label=fr"$\beta={beta:.2f}$")
        theta_pred = cherenkov_angle(n_medium, beta)
        if theta_pred is not None:
            plt.axvline(theta_pred, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"Normalized $S(\theta)$")
    plt.title(fr"Task 8 (fast): Threshold Angular Strength in Medium ($n={n_medium}$)")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(fig_dir, "task8_fast_threshold_angular_profiles.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"\n[Saved] {out1}")

    # One above-threshold map only
    if saved_map is not None:
        plt.figure(figsize=(9, 6))
        extent = [omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]]
        plt.imshow(saved_map, extent=extent, origin="lower", aspect="auto")
        plt.colorbar(label=r"$I(\omega,\theta)$")
        theta_pred = cherenkov_angle(n_medium, saved_beta)
        if theta_pred is not None:
            plt.axhline(
                theta_pred,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=fr"predicted $\theta_C$ for $\beta={saved_beta:.2f}$",
            )
            plt.legend()
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\theta$ [rad]")
        plt.title(fr"Task 8 (fast): Above-Threshold Map ($n={n_medium}$, $\beta={saved_beta:.2f}$)")
        plt.tight_layout()
        out2 = os.path.join(fig_dir, "task8_fast_above_threshold_map.png")
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"[Saved] {out2}")

    print("\nTask 8 (fast) diagnostics:")
    print(f"Medium refractive index n = {n_medium:.3f}")
    print(f"Threshold beta = 1/n = {1.0 / n_medium:.6f}")
    print(f"Low-frequency cut omega_cut = {omega_cut:.3f}")

    for beta, theta_pred, theta_num in peak_report:
        print(f"\nBeta = {beta:.3f}, n*beta = {n_medium * beta:.6f}")
        if theta_pred is None:
            print("  predicted Cherenkov angle: none (below threshold)")
            print(f"  numerical peak angle:      {theta_num:.6f} rad")
        else:
            err = abs(theta_num - theta_pred)
            print(f"  predicted Cherenkov angle: {theta_pred:.6f} rad")
            print(f"  numerical peak angle:      {theta_num:.6f} rad")
            print(f"  absolute angle error:      {err:.6e} rad")

    print("\nTask 8 (fast) pass condition:")
    print("1. Below threshold: no sharp Cherenkov-like peak")
    print("2. Above threshold: angular peak appears")
    print("3. Peak is near predicted theta_C")


if __name__ == "__main__":
    main()