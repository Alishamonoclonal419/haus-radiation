# Haus Radiation Project Plan

## Project aim

Build a first-principles computational project starting from H. A. Haus, *On the radiation from point charges* (1986), then extend it into a serious scientific machine learning project only after the forward physics is clean and validated.

The project will proceed in layers:

1. Extract the exact minimum physics from Haus.
2. Build the forward Fourier-domain source-current solver.
3. Extract the radiating part in vacuum.
4. Validate against known physical behavior and reference formulations.
5. Extend to media.
6. Formulate an inverse problem.
7. Add SciML only where it solves a real bottleneck.

The central rule is simple:

**Physics first, code second, SciML last.**

---

# Planning philosophy

Every task must answer five questions:

1. What physical quantity are we computing?
2. Why is it the right quantity?
3. What equation justifies it?
4. How do we implement it numerically?
5. How do we know the output matches the physics?

We do not move to a later phase unless the current phase has a clear validation target.

---

# Full task ladder

## Phase 0 — Physics foundation from Haus

### Task 0.1

Extract the minimum equation chain from the paper:

* point-charge charge density,
* point-charge current density,
* spatial Fourier transform,
* space-time Fourier transform,
* propagating-wave condition,
* transverse projection,
* constant-velocity interpretation,
* oscillatory-motion example,
* medium modification.

### Task 0.2

Rewrite the physics in your own words as a compact derivation note.

### Task 0.3

Define the computational observables that will be implemented later.

**Exit condition:** the forward problem is defined precisely enough that the code has a clear target.

---

## Phase 1 — Basic forward solver

### Task 1

Implement the Fourier-domain source current for 1D motion:

[
J_z(k_z,\omega)=q\int dt,v(t),e^{-ik_z z(t)}e^{i\omega t}.
]

### Task 2

Implement two benchmark trajectories:

* constant velocity,
* sinusoidal motion.

### Task 3

Plot:

* heatmap of (|J_z(k_z,\omega)|) for constant velocity,
* spectrum for sinusoidal motion.

### Task 4

Check expected structure:

* constant velocity gives a ridge near (\omega=k_z v),
* sinusoidal motion gives harmonics.

**Exit condition:** the source spectrum behaves as physics predicts.

---

## Phase 2 — Radiation extraction in vacuum

### Task 5

Implement the vacuum propagating-mode condition:

[
\mathbf k = \frac{\omega}{c}\hat{\mathbf n}.
]

### Task 6

Implement the transverse projection:

[
\mathbf J_\perp = \hat{\mathbf n}\times(\hat{\mathbf n}\times \mathbf J).
]

### Task 7

Define the numerical radiation observable:

[
I(\omega,\hat{\mathbf n}) \propto |\mathbf J_\perp|^2.
]

### Task 8

Test vacuum behavior:

* constant velocity gives negligible radiation,
* sinusoidal motion radiates.

**Exit condition:** the solver cleanly distinguishes nonradiating and radiating motion in vacuum.

---

## Phase 3 — Validation against known physics

### Task 9

Construct a reference far-field or retarded-time formulation for simple trajectories.

### Task 10

Compare:

* angular patterns,
* frequency peaks,
* scaling behavior,
* numerical convergence.

### Task 11

Document numerical artifacts:

* finite-time broadening,
* windowing effects,
* resolution sensitivity.

**Exit condition:** the Fourier solver survives serious physics scrutiny.

---

## Phase 4 — Medium extension

### Task 12

Implement nondispersive medium:

[
k = n\omega/c.
]

### Task 13

Recover Cherenkov threshold behavior.

### Task 14

Extend to dispersive medium (n(\omega)).

**Exit condition:** one framework handles vacuum and medium cases.

---

## Phase 5 — Inverse problem setup

### Task 15

Choose a low-dimensional trajectory family.

### Task 16

Define sparse/noisy observation models.

### Task 17

Recover trajectory parameters using classical optimization first.

**Exit condition:** the project becomes an actual inverse problem rather than only forward simulation.

---

## Phase 6 — SciML layer

### Task 18

Choose either:

* differentiable solver,
* or physics-grounded surrogate.

### Task 19

Benchmark against classical inversion.

### Task 20

Add uncertainty and identifiability analysis.

**Exit condition:** SciML solves a real bottleneck and is not just decorative.

---

# Task 0.1 — Core equations from Haus

This section is the filled physics foundation extracted from the paper.

## 1. Point-charge charge density

For a point charge (q) moving on trajectory (\mathbf r_0(t)), the charge density is

[
\rho(\mathbf r,t)=q,\delta!\left(\mathbf r-\mathbf r_0(t)\right).
]

This says the entire charge is concentrated at the instantaneous particle position.

Here:

* (\mathbf r) is the field point,
* (t) is time,
* (\mathbf r_0(t)) is the source trajectory.

This is the starting source description in real space.

---

## 2. Point-charge current density

The current density for the same moving point charge is

[
\mathbf J(\mathbf r,t)=q,\mathbf v(t),\delta!\left(\mathbf r-\mathbf r_0(t)\right),
]

where

[
\mathbf v(t)=\dot{\mathbf r}_0(t).
]

This expresses the physically obvious fact that a moving localized charge creates a localized current directed along its instantaneous velocity.

This is the source that enters the wave equation.

---

## 3. Spatial Fourier transform

Take the spatial Fourier transform of the current density:

[
\mathbf J(\mathbf k,t)=\int d^3r,\mathbf J(\mathbf r,t)e^{-i\mathbf k\cdot\mathbf r}.
]

Substituting the point-charge form gives

[
\mathbf J(\mathbf k,t)=q,\mathbf v(t)e^{-i\mathbf k\cdot\mathbf r_0(t)}.
]

This is the first key object in the project.

Interpretation:

* the velocity provides the current amplitude,
* the phase (e^{-i\mathbf k\cdot\mathbf r_0(t)}) encodes the source motion in Fourier space.

This is where the trajectory enters the spectral formulation.

---

## 4. Space-time Fourier transform

Now take the time Fourier transform:

[
\mathbf J(\mathbf k,\omega)=\int dt,\mathbf J(\mathbf k,t)e^{i\omega t}.
]

Using the previous result,

[
\mathbf J(\mathbf k,\omega)=q\int dt,\mathbf v(t)e^{-i\mathbf k\cdot\mathbf r_0(t)}e^{i\omega t}.
]

This is the main source quantity for the project.

For 1D motion along (z),

[
\mathbf r_0(t)=z(t)\hat z,
\qquad
\mathbf v(t)=v(t)\hat z,
]

so the relevant scalar component becomes

[
J_z(k_z,\omega)=q\int dt,v(t)e^{-ik_z z(t)}e^{i\omega t}.
]

This is exactly what will be implemented in the first code stage.

---

## 5. Radiation / propagating-mode condition

The paper’s main physical interpretation is that radiation is associated with those Fourier components of the source that match propagating electromagnetic modes.

In vacuum, propagating electromagnetic waves satisfy

[
k=\omega/c.
]

So the radiating part of the source must be examined on the vacuum light cone in ((\mathbf k,\omega))-space.

This is the spectral-selection idea at the core of the paper:
not every source Fourier component becomes outgoing radiation.
Only those compatible with real propagating waves do.

---

## 6. Transverse projection

Far-field radiation depends on the transverse part of the source relative to the propagation direction.

If the observation direction is (\hat{\mathbf n}), then the transverse radiating component is

[
\mathbf J_\perp = \hat{\mathbf n}\times(\hat{\mathbf n}\times\mathbf J).
]

This removes the longitudinal component and keeps the part compatible with transverse electromagnetic radiation.

Physically:

* the longitudinal part does not represent outgoing transverse radiation,
* the far-field observable is built from the transverse projected current evaluated on the propagating manifold.

This is the quantity we will later use to build radiation intensity maps.

---

## 7. Constant-velocity interpretation

Consider uniform motion:

[
\mathbf r_0(t)=\mathbf v t.
]

Then

[
\mathbf J(\mathbf k,\omega)=q\int dt,\mathbf v,e^{-i\mathbf k\cdot\mathbf v t}e^{i\omega t}.
]

This has support where

[
\omega=\mathbf k\cdot\mathbf v.
]

So the source spectrum lies on that kinematic relation.

For subluminal uniform motion in vacuum, this does not meet the propagating-wave condition in the required way to produce free radiation. The source moves with the charge, but there is no synchronous coupling to freely propagating vacuum modes. That is why uniform motion in free space does not radiate in this framework.

This is one of the central physical messages of the paper.

---

## 8. Oscillatory-motion example

The paper considers an oscillatory trajectory of the form

[
z(t)=d\sin(\omega_0 t).
]

Then

[
v(t)=d\omega_0\cos(\omega_0 t).
]

Because the phase factor contains (e^{-ik_z d\sin(\omega_0 t)}), the resulting source spectrum contains harmonic structure. In the analytic treatment, this is naturally expressed through harmonic/Bessel expansions.

Physical meaning:

* periodic motion produces discrete spectral lines,
* higher harmonics appear depending on the oscillation amplitude and geometry,
* this is the natural benchmark case for a radiating trajectory.

This gives us the second benchmark for the forward solver.

---

## 9. Medium modification

In a medium, the propagating-wave condition changes because the phase velocity differs from (c).

In the simplest refractive-index description,

[
k = n(\omega),\omega/c.
]

This changes the spectral matching condition between the source and the allowed propagating modes.

As a result, constant-velocity motion can radiate if the phase-matching condition is satisfied. This is the spectral interpretation of Cherenkov-type radiation.

So the paper’s vacuum/media contrast is not an extra detail. It is part of the same Fourier-selection picture:

* vacuum: no radiating overlap for ordinary uniform motion,
* medium: overlap can occur.

---

## 10. Computational observables

The physics above defines the observables for later coding.

### Phase 1 observables

* (J_z(k_z,\omega)) for 1D motion,
* heatmap of (|J_z(k_z,\omega)|),
* line plots of (|J_z(\omega)|) at fixed (k_z).

### Phase 2 observables

* transverse projected current (\mathbf J_\perp),
* radiation intensity proxy

[
I(\omega,\hat{\mathbf n}) \propto |\mathbf J_\perp|^2,
]

* angular radiation maps,
* frequency-resolved radiation patterns.

### Phase 4 observables

* medium-modified radiation maps,
* threshold behavior,
* Cherenkov-angle recovery,
* dispersive angular-frequency structure.

### Final project observables

* sparse observation vectors,
* recovered trajectory parameters,
* uncertainty or identifiability metrics,
* speed/accuracy tradeoff for differentiable or surrogate-assisted inversion.

---

# Task 0.2 preview

The next task will be to turn the above equations into a shorter derivation note in plain language, with emphasis on:

* why uniform motion in vacuum does not radiate,
* why periodic motion does,
* why a medium changes the story,
* and exactly what will be coded first.

---

# Task 1 preview

The first code task will implement only the scalar 1D source-current transform:

[
J_z(k_z,\omega)=q\int dt,v(t)e^{-ik_z z(t)}e^{i\omega t}.
]

with two benchmark trajectories:

* constant velocity,
* sinusoidal motion.

The first numerical checks will be:

* ridge near (\omega=k_z v) for constant velocity,
* harmonic peaks for sinusoidal motion.

We will not move to radiation extraction until those are correct.

---

# Minimal milestone map

## Milestone A

Physics foundation complete.

## Milestone B

Source spectrum solver complete.

## Milestone C

Vacuum radiation extraction complete.

## Milestone D

Validation complete.

## Milestone E

Medium extension complete.

## Milestone F

Inverse problem complete.

## Milestone G

SciML layer complete.

## Milestone H

Paper-quality project complete.

---

# Final research target

The final serious version of the project is:

**A first-principles Fourier-domain radiation solver, grounded in Haus’s formulation, extended into a differentiable or surrogate-assisted inverse framework that recovers trajectory parameters from sparse radiation measurements.**

That is the version worth showing to a PhD professor or shaping into a publishable computational physics / SciML paper.
