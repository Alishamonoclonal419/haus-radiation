# Task 1.5 Report — Analytic Harmonic Validation for Sinusoidal Motion

## Objective

Task 1.5 was designed to upgrade the sinusoidal benchmark from a qualitative numerical check to an analytic validation.

The quantity under study remains the **source spectrum**

$$
J_z(k_z,\omega)=q\int dt\,v(t)e^{-ik_z z(t)}e^{i\omega t},
$$

not yet the radiated field.

For the sinusoidal trajectory

$$
z(t)=d\sin(\omega_0 t),
\qquad
v(t)=d\omega_0\cos(\omega_0 t),
$$

the goal was to verify that the numerical harmonic structure follows the Bessel-function pattern implied by the Haus Fourier-domain formulation.

---

## Physics basis

For sinusoidal motion, the phase factor becomes

$$
e^{-ik_z d\sin(\omega_0 t)}.
$$

Define

$$
a = k_z d.
$$

Then the source spectrum is

$$
J_z(k_z,\omega)=q\int dt\,d\omega_0\cos(\omega_0 t)e^{-ia\sin(\omega_0 t)}e^{i\omega t}.
$$

Using the Jacobi–Anger expansion,

$$
e^{-ia\sin(\omega_0 t)}=\sum_{m=-\infty}^{\infty} J_m(a)e^{-im\omega_0 t},
$$

and

$$
\cos(\omega_0 t)=\frac{1}{2}\left(e^{i\omega_0 t}+e^{-i\omega_0 t}\right),
$$

the integrand becomes a sum of discrete harmonics. Collecting the coefficient at harmonic index $n$, one finds that the source-spectrum line strength is proportional to

$$
A_n(a)\propto J_{n+1}(a)+J_{n-1}(a).
$$

Using the Bessel recurrence relation,

$$
J_{n-1}(a)+J_{n+1}(a)=\frac{2n}{a}J_n(a),
$$

the same harmonic weight can also be written as

$$
A_n(a)\propto \frac{n}{a}J_n(a).
$$

So the analytic prediction is not only that lines occur at integer multiples of $\omega_0$, but also that their relative strengths follow a specific Bessel-governed pattern.

---

## Numerical setup

The benchmark used:

- sinusoidal trajectory,
- a Hann time window,
- long integration time to sharpen harmonics,
- fixed $(k_z=2.0)$,
- therefore $a=k_z d=2.0$,
- harmonic comparison over $n=-8,\ldots,8$.

The numerical spectrum was sampled at the discrete harmonic frequencies

$$
\omega = n\omega_0,
$$

and those sampled magnitudes were normalized and compared against the normalized analytic weights

$$
\left|J_{n+1}(a)+J_{n-1}(a)\right|.
$$

---

## Observed results

### 1. Harmonic locations

The full numerical spectrum showed sharp peaks exactly at integer values of

$$
\omega/\omega_0.
$$

This confirms that the source spectrum is discretized into harmonic lines, as expected for a periodic trajectory.

### 2. Relative harmonic strengths

The numerical harmonic heights matched the analytic Bessel prediction point by point over the sampled range.

The comparison table showed:

- exact agreement within displayed precision,
- correct symmetry between positive and negative harmonics,
- vanishing $n=0$ harmonic,
- dominant peaks at $n=\pm 2$ for the chosen parameter $a=2$.

### 3. Shape of the harmonic envelope

The harmonic-weight comparison curve had the expected non-monotonic Bessel-governed structure, rather than a naive exponential or Gaussian decay. This is important, because it shows that the code is not merely producing “some harmonics,” but the correct harmonic pattern implied by the analytic structure.

---

## Physics interpretation

This result is stronger than Task 1 alone.

Task 1 showed that the sinusoidal trajectory produces discrete harmonics.

Task 1.5 shows that those harmonics are not only located correctly, but have the correct **analytic weights** dictated by the Fourier-domain phase structure of the moving charge.

That means the oscillatory benchmark is now validated at two levels:

1. **kinematic level** — lines occur at integer multiples of $\omega_0$,
2. **amplitude level** — relative line strengths follow the Bessel combination
   $$
   J_{n+1}(a)+J_{n-1}(a).
   $$

This is exactly the kind of check a physicist should insist on before moving to the next layer.

---

## What Task 1.5 proves

Task 1.5 proves that the implementation of the sinusoidal source spectrum is analytically correct in its harmonic structure.

More precisely, it confirms that:

- the code correctly handles the oscillatory phase $e^{-ik_z d\sin(\omega_0 t)}$,
- the Fourier transform is resolving the right harmonic content,
- the relative weights are controlled by the expected Bessel structure,
- the normalization and sampling are consistent enough for a precise benchmark.

This is a much stronger result than “the spectrum looks plausible.”

---

## What Task 1.5 does *not* prove

Task 1.5 still does **not** compute radiation.

It validates the **source-spectrum object** for oscillatory motion.

The next physics layer still requires:

1. restricting the source to the propagating manifold,
2. applying the transverse projection,
3. constructing a far-field radiation observable.

So the right conclusion is:

> Task 1.5 closes the source-spectrum stage for the oscillatory benchmark, but does not yet address radiation extraction.

---

## Numerical verdict

Task 1.5 is a success.

The pass condition was:

1. harmonic locations occur at integer multiples of $\omega_0$,
2. relative peak heights follow the analytic Bessel-weight trend,
3. remaining differences are consistent with finite-window effects.

All three conditions are satisfied, and in fact the numerical/analytic agreement is stronger than merely acceptable.

---

## Consequence for project scope

With Task 1 and Task 1.5 both complete, the **source-spectrum stage is now fully validated** for the benchmark trajectories:

- constant velocity,
- sinusoidal motion.

This means the project is ready to move from source-spectrum validation to the next physical layer:

## Next step
### Task 2 — Radiation-level object in vacuum

This task will define and compute the first actual radiation observable by:

- restricting to the vacuum propagating manifold,
- applying the transverse projection,
- testing that constant velocity in vacuum does not radiate,
- testing that oscillatory motion does.

That is the correct next move.

---

## Summary

Task 1.5 completed the analytic validation of the sinusoidal source spectrum.

The major outcomes are:

- harmonic lines occur at the correct frequencies,
- relative harmonic heights follow the exact Bessel-based prediction,
- the oscillatory benchmark is now analytically grounded,
- and the project can move forward with confidence to radiation extraction.

This closes the source-spectrum phase on a solid first-principles foundation.
