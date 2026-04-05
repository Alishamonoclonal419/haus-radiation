import os
import matplotlib.pyplot as plt


def save_heatmap(fig_dir, arr, xvals, yvals, title, cbar_label, fname):
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
    path = os.path.join(fig_dir, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_profile(fig_dir, theta_grid, point_profile, ext_profile, theta_min, theta_max, title, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, point_profile, label="point baseline")
    plt.plot(theta_grid, ext_profile, label="best anisotropic source")
    plt.axvspan(theta_min, theta_max, alpha=0.15, label="detector window")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fixed-$\\phi$ angular intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(fig_dir, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_robustness_plot(fig_dir, chis, rdet_vals, rff_vals, title, fname):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(chis, rdet_vals, marker="o", label=r"$R_{\det}$")
    plt.plot(chis, rff_vals, marker="s", label=r"$R_{\mathrm{ff}}^{3D}$")
    plt.xlabel(r"$\chi$")
    plt.ylabel("Ratio to point baseline")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(fig_dir, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")