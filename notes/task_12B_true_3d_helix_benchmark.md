# Task 12B — True 3D helix benchmark

## Goal

Test whether genuine out-of-plane motion improves directional suppression beyond the planar trajectory results of Task 12A-fix.

Task 12A-fix used a 3D-correct radiation kernel, but the tested source motions were still planar. Task 12B is the first benchmark with a truly three-dimensional bounded source trajectory.

---

## Trajectory family

Use the helical-like bounded trajectory

$$
x(t)=R\cos(\omega t), \qquad
y(t)=R\sin(\omega t), \qquad
z(t)=A_z\sin(\Omega t+\phi_z).
$$

This is a true 3D trajectory because all three coordinates vary nontrivially in time.

---

## Constraint philosophy

As in Task 11A and Task 12A-fix, comparisons should be fair.

Each trajectory is rescaled to satisfy

$$
\sqrt{\langle v_x^2+v_y^2+v_z^2\rangle}=v_{\rm rms,target},
$$

where the target is taken from the 1D baseline sinusoid.

---

## Main question

The key question is:

> does genuine 3D point motion provide better detector-window suppression than planar circular motion without paying a larger global radiation penalty?

---

## Detector observable

Keep the same detector window used in Task 12A-fix:

- polar window:
  $$
  \theta \in [0.55,\ 0.80]
  $$
- azimuth window:
  $$
  \phi \in [-0.35,\ 0.35]
  $$

Define the detector-window score

$$
S_{\rm det}
=
\int_{\phi_1}^{\phi_2} d\phi
\int_{\theta_1}^{\theta_2} d\theta \,\sin\theta
\int_{|\omega|>\omega_{\rm cut}} d\omega\,
I(\omega,\theta,\phi).
$$

The detector suppression ratio is

$$
R_{\rm det}=\frac{S_{\rm det}}{S_{\rm det}^{\rm base}}.
$$

---

## Global observable

Also track the full 3D finite-frequency radiation

$$
S_{\rm ff}^{3D}
=
\int_0^{2\pi} d\phi
\int_0^\pi d\theta \,\sin\theta
\int_{|\omega|>\omega_{\rm cut}} d\omega\,
I(\omega,\theta,\phi),
$$

and

$$
R_{\rm ff}^{3D}
=
\frac{S_{\rm ff}^{3D}}{S_{\rm ff}^{3D,\rm base}}.
$$

This distinguishes genuine improvement from mere detector-window steering with a hidden global cost.

---

## Scan variables

A first benchmark scan should vary:

- the vertical amplitude ratio $A_z/R$,
- the frequency ratio $\Omega/\omega$,
- the vertical phase $\phi_z$.

This is enough to test whether true 3D motion changes the planar picture.

---

## What success means

Task 12B is successful if one or more helical trajectories:

1. achieve lower $R_{\rm det}$ than the planar circular benchmark,
2. do so without making $R_{\rm ff}^{3D}$ noticeably worse,
3. show a stable suppression pocket rather than a single accidental point.

---

## What failure would mean

If the true 3D helix family does not improve the tradeoff meaningfully, then the point-source branch is probably close to exhaustion. That would strengthen the case for moving to extended-source benchmarks.