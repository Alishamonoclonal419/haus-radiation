# Task 11B — Three-harmonic constrained directional optimization

## Goal

Test whether adding a third harmonic improves directional suppression in the chosen detector window under the same fixed RMS-velocity constraint used in Task 11A.

Task 11A showed that a constrained 2-harmonic trajectory can strongly suppress radiation in the narrow angular window \([0.55,0.80]\) rad, but does not achieve global suppression. Task 11B asks whether one more harmonic gives genuinely better directional control.

---

## Trajectory family

Use

$$
z(t)=a_1\sin(\omega t)+a_2\sin(2\omega t+\phi_2)+a_3\sin(3\omega t+\phi_3).
$$

Each candidate is rescaled so that

$$
\sqrt{\langle v^2\rangle}=v_{\rm rms,target},
$$

where the target is the RMS velocity of the baseline sinusoid.

This makes all comparisons kinematically fair.

---

## Primary objective

Minimize radiation in the narrow detector window

$$
\Theta_{\rm win}=[0.55,\ 0.80]\ \text{rad}.
$$

Define

$$
S_\Theta=\int_{\theta_1}^{\theta_2} d\theta \int_{|\omega|>\omega_{\rm cut}} I(\omega,\theta)\,d\omega.
$$

The corresponding suppression factor is

$$
R_\Theta=\frac{S_\Theta}{S_\Theta^{\rm base}}.
$$

---

## Secondary diagnostic

Also compute

$$
S_{\rm ff}=\int_0^\pi d\theta \int_{|\omega|>\omega_{\rm cut}} I(\omega,\theta)\,d\omega
$$

and

$$
R_{\rm ff}=\frac{S_{\rm ff}}{S_{\rm ff}^{\rm base}}.
$$

This determines whether the optimized result is only directional quieting through redistribution, or whether global radiative behavior also improves.

---

## What success means

Task 11B is successful if:

1. a 3-harmonic constrained candidate gives lower $R_\Theta$ than Task 11A,
2. the improvement is not paid for by a much worse $R_{\rm ff}$,
3. the result is robust under nearby parameter changes,
4. the best waveform remains physically interpretable rather than pathological.

---

## Why this matters

Task 11B tests whether the 2-harmonic optimum from Task 11A was already near the directional frontier, or whether an extra harmonic opens a new suppression mechanism.