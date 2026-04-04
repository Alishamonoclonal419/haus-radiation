# Task 8.2 — Threshold peak sharpness and strength scaling

## Goal

After validating the threshold-angle law in Task 8.1, the next question is how the angular enhancement develops as the system moves above threshold.

This task measures not only where the peak is, but also how strong and how sharp it becomes.

---

## Physical basis

For constant-velocity motion in a nondispersive medium, the phase mismatch is

$$
1 - n\beta\cos\theta.
$$

Above threshold, $n\beta > 1$, a real angle exists where this mismatch vanishes:

$$
\theta_C = \arccos\!\left(\frac{1}{n\beta}\right).
$$

Very close to threshold, the enhancement should be broad and weak. Farther above threshold, it should become more localized and more pronounced.

---

## Observable

Use the same integrated nonzero-frequency angular strength as in Task 8 and 8.1:

$$
S(\theta)=\int_{|\omega|>\omega_{\rm cut}} I(\omega,\theta)\, d\omega.
$$

From each $S(\theta)$, extract:

1. peak angle $\theta_{\rm peak}$
2. peak height $S_{\max}$
3. peak width (FWHM)

---

## What should be tested

Scan a set of above-threshold velocities, for example

$$
\beta = 0.68,\ 0.72,\ 0.76,\ 0.80,\ 0.85,\ 0.90
$$

for $n=1.5$.

For each case, compute $S(\theta)$, then measure:

- $\theta_{\rm peak}$
- $S_{\max}$
- FWHM

and plot them against

$$
n\beta - 1.
$$

---

## What success means

Task 8.2 is successful if:

1. the peak height increases in a sensible way as the system moves above threshold,
2. the peak width decreases in a sensible way away from threshold,
3. the extracted peak angle remains close to the law validated in Task 8.1.

This would complete the threshold story by showing not only the angle law, but also the onset of angular localization and peak growth.
