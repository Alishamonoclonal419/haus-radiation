# Task 1 Report — Fourier-Domain Source Spectrum of a Moving Point Charge

## Objective

Task 1 was designed to validate the first numerical object in the Haus-based project: the **source spectrum**, not yet the radiation field.

The quantity computed was

\[
J_z(k_z,\omega)=q\int dt\,v(t)e^{-ik_z z(t)}e^{i\omega t}.
\]

This quantity measures how strongly a prescribed 1D charge trajectory excites space-time Fourier modes \((k_z,\omega)\). The purpose of Task 1 was to check whether the code reproduces the expected spectral support structure for two benchmark trajectories:

1. constant velocity,
2. sinusoidal motion.

This stage does **not** yet compute radiation. It computes the source spectrum that later feeds the radiation extraction step.

---

## Physics basis

From Haus, the point-charge current density is

\[
\mathbf J(\mathbf r,t)=q\,\mathbf v(t)\,\delta\!\left(\mathbf r-\mathbf r_0(t)\right),
\]

which leads, after spatial and temporal Fourier transformation, to

\[
\mathbf J(\mathbf k,\omega)=q\int dt\,\mathbf v(t)e^{-i\mathbf k\cdot\mathbf r_0(t)}e^{i\omega t}.
\]

Restricting to 1D motion along \(z\),

\[
J_z(k_z,\omega)=q\int dt\,v(t)e^{-ik_z z(t)}e^{i\omega t}.
\]

This was the numerical target for Task 1.

---

## Numerical implementation

The implementation used:

- a sampled time grid,
- trapezoidal integration,
- optional time windowing,
- normalization by the integrated window weight,
- a memory-safe evaluation strategy that loops over \(k_z\) rather than building the full \((N_k,N_\omega,N_t)\) tensor.

Two windows were used:

- rectangular window,
- Hann window.

The rectangular window preserves the raw finite-time spectral structure but introduces stronger sidelobes. The Hann window suppresses leakage and produces a cleaner main ridge.

---

## Benchmark 1 — Constant velocity

### Trajectory

\[
z(t)=vt,
\qquad
v(t)=v.
\]

### Expected result

Substituting into the source spectrum gives

\[
J_z(k_z,\omega)=qv\int dt\,e^{i(\omega-k_z v)t}.
\]

For a finite observation window, this is a broadened peak centered at

\[
\omega=k_z v.
\]

So the expected 2D structure is a ridge in the \((k_z,\omega)\) plane with slope fixed by \(v\).

### Observed result

The rectangular-window heatmap showed a bright diagonal ridge exactly aligned with the overlay

\[
\omega=v k_z,
\qquad v=0.8.
\]

This confirms that the numerical Fourier transform is placing the source support at the correct kinematic location.

### Window comparison

The Hann-window heatmap showed the same ridge, but with much cleaner background and strongly reduced sidelobe structure.

Interpretation:
- the ridge location is physical,
- the faint diagonal haze in the rectangular case is finite-window leakage,
- the Hann window suppresses the numerical leakage without moving the ridge.

This is exactly the expected behavior.

---

## Benchmark 2 — Peak narrowing with increasing time window

### Expected result

For constant velocity and finite time window \(T\), the spectrum near fixed \(k_z\) should narrow in frequency as \(T\) increases. In other words, the finite-window approximation should converge toward a delta-like structure centered at

\[
\omega = v k_z.
\]

### Observed result

At fixed \(k_z=3\), the three curves for

- \(T=20\),
- \(T=50\),
- \(T=100\)

all peak at the expected center

\[
\omega = v k_z = 2.4,
\]

and the peak becomes sharper as \(T\) increases.

This confirms two things:

1. the center frequency is correct,
2. the finite-time broadening scales in the right direction.

The visible oscillatory sidelobes are expected for a rectangular window and are not a bug.

---

## Benchmark 3 — Sinusoidal motion

### Trajectory

\[
z(t)=d\sin(\omega_0 t),
\qquad
v(t)=d\omega_0\cos(\omega_0 t).
\]

### Expected result

Because the phase factor becomes

\[
e^{-ik_z d\sin(\omega_0 t)},
\]

the source spectrum contains harmonic structure. The oscillatory motion should therefore produce peaks near integer multiples of \(\omega_0\).

### Observed result

The Hann-window spectrum showed sharp peaks at integer values of

\[
\omega/\omega_0,
\]

with symmetric positive and negative harmonics, and amplitudes depending on \(k_z\).

Interpretation:
- the harmonic spacing is correct,
- the source spectrum is discrete rather than broadband,
- the dependence on \(k_z\) is physically reasonable,
- the Hann window gives a clean representation of the harmonic lines.

This is the expected qualitative behavior from the Haus oscillatory example.

---

## Physics verdict

Task 1 is a success.

The computed source spectrum reproduces the expected benchmark structures:

- **constant velocity:** ridge near \(\omega = v k_z\),
- **longer time window:** narrower peak in frequency,
- **sinusoidal motion:** harmonic spectral lines at integer multiples of \(\omega_0\).

So the code is correctly computing the **source-spectrum object**

\[
J_z(k_z,\omega),
\]

which is the right first numerical target from the Haus formulation.

---

## What Task 1 does *not* prove

Task 1 does **not** yet demonstrate radiation.

It demonstrates only that the source Fourier transform is being computed correctly.

To obtain radiation, later tasks must still:

1. restrict to the propagating manifold,
2. apply the transverse projection,
3. define a far-field spectral observable.

So the correct conclusion is:

> Task 1 validates the source spectrum, not yet the radiated field.

This distinction must be kept explicit in all future notes and code comments.

---

## Numerical observations

### Strengths

- Ridge location in the constant-velocity case is correct.
- Window effects behave as expected.
- Harmonic structure in the sinusoidal case is clear and stable.
- The memory-safe implementation is appropriate for future scaling.

### Remaining improvement opportunities

1. Add an analytic comparison for the sinusoidal case using the harmonic/Bessel expansion.
2. Add a short conventions note describing normalization and units.
3. Add automated tests that verify ridge location and harmonic positions numerically.
4. Keep Task 1 clearly labeled as source-spectrum validation, not radiation computation.

---

## Exit condition assessment

The Phase 1 exit condition was:

> the source spectrum behaves as physics predicts.

That condition is satisfied.

Task 1 can therefore be marked **complete**.

---

## Recommended next step

The strongest next move is:

### Task 1.5 — Analytic harmonic validation

Use the sinusoidal trajectory to compare the numerical spectrum against the expected harmonic/Bessel structure. This would make the benchmark more rigorous before moving to radiation extraction.

After that, proceed to the radiation-level quantities:

- propagating-manifold restriction,
- transverse projection,
- vacuum radiation observable.

---

## Summary

Task 1 successfully established the first computational foundation of the project:

- the Fourier-domain source current for 1D motion has been implemented,
- the constant-velocity benchmark behaves correctly,
- the finite-window scaling behaves correctly,
- the sinusoidal benchmark shows the expected harmonic structure,
- and the code has been improved to a memory-safe form suitable for the next stages.

This is a valid and necessary first-principles starting point for the full Haus-based radiation project.

