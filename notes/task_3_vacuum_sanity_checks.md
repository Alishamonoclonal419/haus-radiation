# Task 3 — Vacuum radiation sanity checks

## Goal

Strengthen the corrected vacuum radiation benchmark by checking geometric and symmetry constraints that must hold for z-directed motion in vacuum.

This task does not yet introduce a new radiation formula. It tests whether the current filtered vacuum observable behaves like a physically sensible transverse radiation quantity.

---

## Starting point

From Task 2, the working filtered observable is

\[
I_{\rm filt}(\omega,\theta)
\]

built from the structural vacuum quantity

\[
I(\omega,\theta)\propto \sin^2\theta
\left|
J_z\!\left(\frac{\omega}{c}\cos\theta,\omega\right)
\right|^2,
\]

with a small band around \(\omega=0\) removed.

---

## What must be true physically

For z-directed motion in vacuum:

### 1. On-axis suppression

Because the transverse factor contains \(\sin^2\theta\),

\[
I(\omega,0)=I(\omega,\pi)=0.
\]

So all harmonic angular profiles should vanish on the axis.

### 2. Reflection symmetry

For motion along the z-axis with no handedness or azimuthal structure, the radiation pattern should satisfy

\[
I(\omega,\theta)=I(\omega,\pi-\theta).
\]

So harmonic angular profiles should be symmetric about \(\theta=\pi/2\).

### 3. Finite-frequency harmonic structure

For sinusoidal motion, the filtered observable should remain nonzero at harmonics

\[
\omega = n\omega_0,
\qquad n\neq 0,
\]

while the constant-velocity benchmark remains negligible outside the DC exclusion band.

---

## What Task 3 should compute

1. The angular profile at selected harmonics \(n\omega_0\) for sinusoidal motion.
2. The maximum on-axis value compared with the maximum off-axis value.
3. The symmetry error

\[
\Delta_{\rm sym}(\omega,\theta)=|I(\omega,\theta)-I(\omega,\pi-\theta)|.
\]

4. A comparison showing that the constant-velocity case remains negligible at nonzero frequency.

---

## What success means

Task 3 is successful if:

- harmonic angular profiles vanish on-axis,
- the profiles are symmetric under \(\theta\to\pi-\theta\),
- the sinusoidal case remains nonzero at finite-frequency harmonics,
- the constant-velocity case remains negligible outside the DC band.

This would mean the corrected vacuum observable is not only qualitatively useful, but also respects the expected geometric structure of transverse radiation.