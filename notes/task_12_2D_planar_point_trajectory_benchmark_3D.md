# Task 12 — 3D point-trajectory benchmark

## Goal

Extend the present 1D trajectory framework to fully 3D point motion and test whether geometric freedom in three spatial directions allows stronger directional suppression than the best 1D constrained results.

This task does **not** assume global suppression will appear. The first goal is more modest:

> determine whether 3D point trajectories can outperform 1D trajectories in directional quieting and channel shaping.

---

## Why this task is needed

The 1D trajectory branch established several useful facts:

- total finite-frequency suppression was **not** found,
- frequency-band suppression **was** found,
- narrow angular-sector suppression **was** found,
- constrained optimization improved directional suppression,
- all successful cases worked by **redistribution**, not global disappearance of radiation.

This suggests that the present 1D ansatz family may already be near its limit.

A natural next step is therefore to increase the geometric freedom of the source itself.

---

## Physical idea

A 1D point trajectory can only reshape radiation by changing its time dependence along one axis.

A 3D point trajectory may allow additional effects:

- directional interference between different velocity components,
- richer angular redistribution,
- suppression in one detector direction by steering radiation into others,
- geometric cancellation that is not available in 1D motion.

This still does **not** guarantee global suppression. But it is the correct next benchmark before moving to extended sources.

---

## Core question

The main question of Task 12 is:

> does 3D point motion allow substantially better directional suppression than the best constrained 1D trajectory?

A secondary question is:

> does 3D geometry qualitatively change the physics story, or does it still reduce to redistribution rather than true suppression?

---

## Candidate 3D trajectory families

The first benchmark set should be simple and interpretable.

### 1. Circular motion

$$
x(t)=R\cos(\omega t), \qquad
y(t)=R\sin(\omega t), \qquad
z(t)=0.
$$

This is the most basic 3D trajectory beyond the 1D line.

### 2. Elliptical motion

$$
x(t)=a\cos(\omega t), \qquad
y(t)=b\sin(\omega t), \qquad
z(t)=0.
$$

This introduces anisotropy and tests whether eccentricity changes directional quieting.

### 3. Helical motion

$$
x(t)=R\cos(\omega t), \qquad
y(t)=R\sin(\omega t), \qquad
z(t)=v_z t.
$$

This adds simultaneous transverse and longitudinal motion.

### 4. Lissajous motion

$$
x(t)=A_x\sin(\omega_x t+\phi_x), \qquad
y(t)=A_y\sin(\omega_y t+\phi_y), \qquad
z(t)=A_z\sin(\omega_z t+\phi_z).
$$

This is the natural richer family if the simpler families are promising.

---

## Constraint philosophy

As in Task 11A, comparisons should be fair.

The first constrained 3D benchmark should fix one kinematic scale such as:

$$
\sqrt{\langle v_x^2+v_y^2+v_z^2\rangle}=v_{\rm rms,target}.
$$

This prevents trivial wins from simply making the motion weaker or stronger overall.

If needed, later benchmarks can also test fixed RMS acceleration or fixed spatial extent.

---

## Primary observable

Use the same kind of directional objective that succeeded in the 1D branch.

For a chosen detector window

$$
\Theta_{\rm win}=[\theta_1,\theta_2],
$$

define

$$
S_\Theta=\int_{\theta_1}^{\theta_2} d\theta \int_{|\omega|>\omega_{\rm cut}} I(\omega,\theta)\,d\omega.
$$

The suppression factor is

$$
R_\Theta=\frac{S_\Theta}{S_\Theta^{\rm base}}.
$$

The baseline should be chosen carefully. At minimum, compare against the best 1D constrained baseline and against a simple 3D reference such as circular motion with the same RMS speed.

---

## Secondary observables

Also track:

### Total finite-frequency radiation

$$
S_{\rm ff}=\int_0^\pi d\theta \int_{|\omega|>\omega_{\rm cut}} I(\omega,\theta)\,d\omega,
$$

and

$$
R_{\rm ff}=\frac{S_{\rm ff}}{S_{\rm ff}^{\rm base}}.
$$

### Optional frequency-band score

If useful, also measure a band-limited score as in Task 10A to see whether 3D motion helps spectral shaping in addition to directional shaping.

---

## What success means

Task 12 is successful if one or more 3D point trajectories:

1. achieve lower $R_\Theta$ than the best constrained 1D result,
2. do so without an extreme increase in $R_{\rm ff}$,
3. show stable suppression pockets rather than one-point accidents,
4. reveal genuinely new angular structure rather than a trivial deformation of the 1D story.

---

## What failure would mean

If 3D point trajectories still do not improve the picture qualitatively, then that is also informative.

It would suggest that the limitation is not merely the dimensionality of the trajectory, but the fact that the source is still a single moving point.

That would strongly motivate the extended-source branch.

---

## Why this matters for the larger project

Task 12 is the natural bridge between:

- 1D point-trajectory redistribution,
- richer geometric point-source structure,
- and eventually extended-source suppression mechanisms.

It also provides a more defensible semiclassical bridge to the intuition that structured motion in space may radiate very differently from naive 1D accelerated motion.

This is still **not** a derivation of hydrogen stationary states. But it is a more serious structural benchmark than the 1D branch alone.