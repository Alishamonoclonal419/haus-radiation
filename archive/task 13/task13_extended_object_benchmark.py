#!/usr/bin/env python3
"""
Task 13A: classical extended-object benchmark

Physics idea
------------
For a rigid translated, spherically symmetric source,
    rho(r, t) = rho0(r - a(t))
    j(r, t)   = rho0(r - a(t)) * a_dot(t),

the spatial Fourier transform factorizes:
    J_ext(k, omega) = F(|k|) * J_point(k, omega),

where F is the static source form factor.

This script benchmarks whether finite spatial extent improves the
local/global radiation tradeoff relative to the point-source baseline.

Implemented source families
---------------------------
1. Spherical shell:
       F_shell(q) = sin(q b) / (q b)

2. Uniform spherical volume:
       F_vol(q) = 3 [sin(q b) - q b cos(q b)] / (q b)^3

Baseline center-of-mass motion
------------------------------
Planar circular motion:
    a(t) = (A cos(w0 t), A sin(w0 t), 0)

The code computes a 3D radiation map from the transverse Fourier current
on the radiation shell for harmonics n = 1, ..., n_max.

Output
------
- baseline radiation map at fixed phi = 0
- R_det and R_ff curves versus chi = b / (c T)
- tradeoff plot R_ff versus R_det
- best-candidate fixed-phi map and angular profile

This is a Task 13A benchmark, not the full Task 13 program.
It isolates the effect of finite size only.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# User-facing configuration
# ----------------------------
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Units: set c = 1 and fundamental angular frequency w0 = 1.
C = 1.0
W0 = 1.0
T0 = 2.0 * np.pi / W0

# Baseline center motion
A_CENTER = 1.0

# Fourier / angular resolution
N_T = 2048
N_MAX = 8
N_THETA = 64
N_PHI = 48

# Detector window
THETA_MIN = 0.55
THETA_MAX = 0.80
PHI_MIN = 0.0
PHI_MAX = 2.0 * np.pi

# Fixed-phi diagnostic slice
PHI_FIXED = 0.0

# Scan variable: chi = b / (c T)
CHI_GRID = np.linspace(0.0, 1.20, 49)

# To favor genuinely useful candidates instead of trivial local minima,
# rank candidates by a simple combined score:
#   score = R_det + lambda_penalty * max(R_ff - 1, 0)
LAMBDA_GLOBAL = 2.0


# ----------------------------
# Source form factors
# ----------------------------
def sinc1(x: np.ndarray) -> np.ndarray:
    """Unnormalized spherical sinc: sin(x)/x with stable x->0 limit."""
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-12
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def form_factor_shell(q: np.ndarray, b: float) -> np.ndarray:
    """Spherical shell form factor."""
    x = q * b
    return sinc1(x)


def form_factor_volume(q: np.ndarray, b: float) -> np.ndarray:
    """Uniform solid sphere form factor."""
    x = q * b
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-10
    xm = x[mask]
    out[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)
    return out


# ----------------------------
# Baseline center motion
# ----------------------------
def trajectory_planar_circular(t: np.ndarray, amp: float = A_CENTER, w0: float = W0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        r_t : shape (3, Nt)
        v_t : shape (3, Nt)
    """
    x = amp * np.cos(w0 * t)
    y = amp * np.sin(w0 * t)
    z = np.zeros_like(t)

    vx = -amp * w0 * np.sin(w0 * t)
    vy = amp * w0 * np.cos(w0 * t)
    vz = np.zeros_like(t)

    r_t = np.vstack([x, y, z])
    v_t = np.vstack([vx, vy, vz])
    return r_t, v_t


# ----------------------------
# Radiation engine
# ----------------------------
def build_angle_grids(n_theta: int = N_THETA, n_phi: int = N_PHI) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    th2d, ph2d = np.meshgrid(theta, phi, indexing="ij")

    dirs = np.stack(
        [
            np.sin(th2d) * np.cos(ph2d),
            np.sin(th2d) * np.sin(ph2d),
            np.cos(th2d),
        ],
        axis=-1,
    )  # shape (Ntheta, Nphi, 3)

    return theta, phi, dirs


def compute_point_harmonic_intensity(
    r_t: np.ndarray,
    v_t: np.ndarray,
    dirs: np.ndarray,
    t: np.ndarray,
    n_max: int = N_MAX,
    c: float = C,
    w0: float = W0,
) -> np.ndarray:
    """
    Compute baseline point-source intensity on the radiation shell.

    Returns
    -------
    intensity : ndarray, shape (n_max, Ntheta, Nphi)
        Harmonic-resolved intensity. Harmonic index i corresponds to n = i+1.
    """
    n_theta, n_phi, _ = dirs.shape
    m_dirs = n_theta * n_phi

    dir_flat = dirs.reshape(m_dirs, 3)
    r_dot = dir_flat @ r_t                  # (M, Nt)

    intens = np.zeros((n_max, m_dirs), dtype=float)

    for n in range(1, n_max + 1):
        omega_n = n * w0
        k_n = omega_n / c

        phase = np.exp(-1j * k_n * r_dot + 1j * omega_n * t[None, :])  # (M, Nt)
        # J_n ~ \int dt v(t) e^{-ik.r(t)} e^{i n w0 t}
        J = np.einsum("mt,it->mi", phase, v_t) / t.size                # (M, 3)

        khat = dir_flat
        J_dot_k = np.einsum("mi,mi->m", J, khat)
        J_perp = J - J_dot_k[:, None] * khat

        intens[n - 1] = (omega_n**2) * np.sum(np.abs(J_perp) ** 2, axis=1).real

    return intens.reshape(n_max, n_theta, n_phi)


def build_extended_intensity(
    point_intensity: np.ndarray,
    chi: float,
    source_kind: str,
    c: float = C,
    w0: float = W0,
) -> np.ndarray:
    """
    Apply the static source form factor to the point baseline.

    chi = b / (c T), with T = 2 pi / w0  =>  b = chi * c * T = 2 pi chi c / w0
    """
    b = chi * c * (2.0 * np.pi / w0)
    harmonics = np.arange(1, point_intensity.shape[0] + 1, dtype=float)
    q = harmonics * w0 / c

    if source_kind == "shell":
        F = form_factor_shell(q, b)
    elif source_kind == "volume":
        F = form_factor_volume(q, b)
    else:
        raise ValueError(f"unknown source_kind={source_kind!r}")

    return point_intensity * (F[:, None, None] ** 2)


# ----------------------------
# Scores and diagnostics
# ----------------------------
def angular_weights(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    if len(theta) < 2 or len(phi) < 2:
        raise ValueError("Need at least 2 theta and phi points.")
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    return np.sin(theta)[:, None] * dtheta * dphi


def compute_scores(
    intensity: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    phi_min: float = PHI_MIN,
    phi_max: float = PHI_MAX,
) -> tuple[float, float]:
    """
    Returns:
        S_det, S_ff
    """
    w = angular_weights(theta, phi)  # (Ntheta, Nphi)

    theta_mask = (theta[:, None] >= theta_min) & (theta[:, None] <= theta_max)
    phi_wrapped = (phi[None, :] % (2.0 * np.pi))
    phi_mask = (phi_wrapped >= (phi_min % (2.0 * np.pi))) & (phi_wrapped <= (phi_max % (2.0 * np.pi)))
    det_mask = theta_mask & phi_mask

    summed = intensity.sum(axis=0)  # sum over harmonics -> (Ntheta, Nphi)

    s_ff = float(np.sum(summed * w))
    s_det = float(np.sum(summed * w * det_mask))
    return s_det, s_ff


def fixed_phi_profile(intensity: np.ndarray, theta: np.ndarray, phi: np.ndarray, phi_fixed: float = PHI_FIXED) -> np.ndarray:
    idx = int(np.argmin(np.abs(phi - phi_fixed)))
    return intensity.sum(axis=0)[:, idx]


def fixed_phi_map(intensity: np.ndarray, phi: np.ndarray, phi_fixed: float = PHI_FIXED) -> np.ndarray:
    idx = int(np.argmin(np.abs(phi - phi_fixed)))
    return intensity.sum(axis=0)[:, idx]  # (Ntheta,)


# ----------------------------
# Plot helpers
# ----------------------------
def save_fixed_phi_map(point_or_ext: np.ndarray, theta: np.ndarray, title: str, filename: Path) -> None:
    # point_or_ext shape (nmax, ntheta, nphi) -> show summed intensity at fixed phi as (omega_n, theta)
    summed = point_or_ext
    nmax = summed.shape[0]
    harm = np.arange(1, nmax + 1)
    omega = harm * W0

    idx_phi = int(np.argmin(np.abs(PHI_GRID_CACHE - PHI_FIXED)))
    z = summed[:, :, idx_phi].T  # (theta, omega)

    plt.figure(figsize=(8.5, 5.0))
    plt.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=[omega[0], omega[-1], theta[0], theta[-1]],
    )
    plt.colorbar(label=r"$I(\omega_n,\theta;\phi=0)$")
    plt.xlabel(r"$\omega_n$")
    plt.ylabel(r"$\theta$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"[Saved] {filename}")


def save_profile_comparison(theta: np.ndarray, base_profile: np.ndarray, ext_profile: np.ndarray, title: str, filename: Path) -> None:
    plt.figure(figsize=(8.2, 5.0))
    plt.plot(theta, base_profile, label="point baseline")
    plt.plot(theta, ext_profile, label="extended source")
    plt.axvspan(THETA_MIN, THETA_MAX, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"[Saved] {filename}")


def save_scan_plot(x: np.ndarray, ys: dict[str, np.ndarray], ylabel: str, title: str, filename: Path) -> None:
    plt.figure(figsize=(8.4, 5.0))
    for label, y in ys.items():
        plt.plot(x, y, label=label)
    plt.axhline(1.0, ls="--", lw=1.0, color="gray")
    plt.xlabel(r"$\chi = b/(cT)$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"[Saved] {filename}")


def save_tradeoff_plot(
    rdet_shell: np.ndarray,
    rff_shell: np.ndarray,
    rdet_vol: np.ndarray,
    rff_vol: np.ndarray,
    chi: np.ndarray,
    filename: Path,
) -> None:
    plt.figure(figsize=(7.6, 5.6))
    plt.plot(rdet_shell, rff_shell, "-o", ms=3, label="shell")
    plt.plot(rdet_vol, rff_vol, "-s", ms=3, label="volume")
    # mark best points
    is_shell = np.argmin(rdet_shell + LAMBDA_GLOBAL * np.clip(rff_shell - 1.0, 0.0, None))
    is_vol = np.argmin(rdet_vol + LAMBDA_GLOBAL * np.clip(rff_vol - 1.0, 0.0, None))
    plt.scatter([rdet_shell[is_shell]], [rff_shell[is_shell]], s=70, marker="o")
    plt.scatter([rdet_vol[is_vol]], [rff_vol[is_vol]], s=70, marker="s")
    plt.xlabel(r"$R_{\rm det}$")
    plt.ylabel(r"$R_{\rm ff}^{3D}$")
    plt.title("Task 13A: local/global tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"[Saved] {filename}")


# cache used by plotting helper
PHI_GRID_CACHE = None


def main() -> None:
    print("Starting Task 13A extended-object benchmark...", flush=True)

    t = np.linspace(0.0, T0, N_T, endpoint=False)
    theta, phi, dirs = build_angle_grids(N_THETA, N_PHI)
    global PHI_GRID_CACHE
    PHI_GRID_CACHE = phi

    r_t, v_t = trajectory_planar_circular(t, amp=A_CENTER, w0=W0)

    print("Computing point-source baseline...", flush=True)
    point_intensity = compute_point_harmonic_intensity(r_t, v_t, dirs, t, n_max=N_MAX, c=C, w0=W0)
    sdet_base, sff_base = compute_scores(point_intensity, theta, phi)

    base_profile = fixed_phi_profile(point_intensity, theta, phi, phi_fixed=PHI_FIXED)

    print(f"Baseline S_det   = {sdet_base:.6e}")
    print(f"Baseline S_ff3D  = {sff_base:.6e}")
    print(f"Detector window  = [{THETA_MIN:.3f}, {THETA_MAX:.3f}] rad")
    print(f"Fixed-phi slice  = {PHI_FIXED:.3f} rad", flush=True)

    save_fixed_phi_map(
        point_intensity,
        theta,
        title="Task 13A: point-source baseline map at fixed $\\phi=0$",
        filename=OUT_DIR / "task13A_point_baseline_map_phi0.png",
    )

    shell_rdet = []
    shell_rff = []
    vol_rdet = []
    vol_rff = []

    shell_scores = []
    vol_scores = []

    best_shell = None
    best_vol = None

    for chi in CHI_GRID:
        shell_int = build_extended_intensity(point_intensity, chi=chi, source_kind="shell", c=C, w0=W0)
        vol_int = build_extended_intensity(point_intensity, chi=chi, source_kind="volume", c=C, w0=W0)

        sdet_shell, sff_shell = compute_scores(shell_int, theta, phi)
        sdet_vol, sff_vol = compute_scores(vol_int, theta, phi)

        rdet_s = sdet_shell / sdet_base
        rff_s = sff_shell / sff_base
        rdet_v = sdet_vol / sdet_base
        rff_v = sff_vol / sff_base

        shell_rdet.append(rdet_s)
        shell_rff.append(rff_s)
        vol_rdet.append(rdet_v)
        vol_rff.append(rff_v)

        score_s = rdet_s + LAMBDA_GLOBAL * max(rff_s - 1.0, 0.0)
        score_v = rdet_v + LAMBDA_GLOBAL * max(rff_v - 1.0, 0.0)
        shell_scores.append(score_s)
        vol_scores.append(score_v)

        if best_shell is None or score_s < best_shell["score"]:
            best_shell = {
                "chi": chi,
                "score": score_s,
                "rdet": rdet_s,
                "rff": rff_s,
                "intensity": shell_int,
            }

        if best_vol is None or score_v < best_vol["score"]:
            best_vol = {
                "chi": chi,
                "score": score_v,
                "rdet": rdet_v,
                "rff": rff_v,
                "intensity": vol_int,
            }

    shell_rdet = np.asarray(shell_rdet)
    shell_rff = np.asarray(shell_rff)
    vol_rdet = np.asarray(vol_rdet)
    vol_rff = np.asarray(vol_rff)

    print("\nBest shell candidate:")
    print(
        f"  chi={best_shell['chi']:.4f}, "
        f"R_det={best_shell['rdet']:.6e}, "
        f"R_ff3D={best_shell['rff']:.6e}, "
        f"score={best_shell['score']:.6e}"
    )
    print("Best volume candidate:")
    print(
        f"  chi={best_vol['chi']:.4f}, "
        f"R_det={best_vol['rdet']:.6e}, "
        f"R_ff3D={best_vol['rff']:.6e}, "
        f"score={best_vol['score']:.6e}"
    )

    save_scan_plot(
        CHI_GRID,
        {"shell": shell_rdet, "volume": vol_rdet},
        ylabel=r"$R_{\rm det}$",
        title="Task 13A: detector-window ratio vs finite-size parameter",
        filename=OUT_DIR / "task13A_Rdet_vs_chi.png",
    )
    save_scan_plot(
        CHI_GRID,
        {"shell": shell_rff, "volume": vol_rff},
        ylabel=r"$R_{\rm ff}^{3D}$",
        title="Task 13A: total-radiation ratio vs finite-size parameter",
        filename=OUT_DIR / "task13A_Rff_vs_chi.png",
    )
    save_tradeoff_plot(
        shell_rdet, shell_rff, vol_rdet, vol_rff, CHI_GRID,
        filename=OUT_DIR / "task13A_tradeoff_shell_vs_volume.png",
    )

    # Save best candidates
    for tag, info in [("shell", best_shell), ("volume", best_vol)]:
        ext_profile = fixed_phi_profile(info["intensity"], theta, phi, phi_fixed=PHI_FIXED)
        save_fixed_phi_map(
            info["intensity"],
            theta,
            title=rf"Task 13A: best {tag} map at fixed $\phi=0$ ($\chi={info['chi']:.3f}$)",
            filename=OUT_DIR / f"task13A_best_{tag}_map_phi0.png",
        )
        save_profile_comparison(
            theta,
            base_profile,
            ext_profile,
            title=rf"Task 13A: point baseline vs best {tag} profile ($\chi={info['chi']:.3f}$)",
            filename=OUT_DIR / f"task13A_best_{tag}_profile_phi0.png",
        )

    print("\nTask 13A complete.", flush=True)


if __name__ == "__main__":
    main()