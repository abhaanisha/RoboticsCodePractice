# Localization

**Localization** is a robot answering one question over and over: *where am I?*

It never gets a direct answer. It gets noisy hints — a GNSS fix that wanders by metres,
wheel encoders that drift, a laser scan that half-matches the map — and it has to fuse
them into a position estimate it can actually drive on.

Every algorithm in this folder is the **same two-step loop**, wearing different clothes:

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
   ┌─────────┐   "I moved"      ┌─────────┐        │
   │ PREDICT │ ───────────────► │ UPDATE  │ ───────┘
   │  blur   │                  │ sharpen │
   └─────────┘                  └─────────┘
   uncertainty grows            uncertainty shrinks
   (motion model)               (sensor model + Bayes' rule)
```

**Predict** folds in what the robot *did* and always makes it less sure.
**Update** folds in what the robot *saw* and always makes it more sure.
Localization is the race between the two.

What changes from day to day is only **how the belief is stored** — and that choice
drives everything else about cost, accuracy, and what can go wrong.

---

## Roadmap

| Day | Algorithm | How the belief is stored | Status |
|:---:|---|---|:---:|
| **1** | [**Histogram filter**](day01_histogram_filter_1d/) (discrete Bayes, 1D) | one probability per cell | ✅ done |
| 2 | Kalman filter, 1D | a mean and a variance (2 numbers) | planned |
| 3 | Kalman filter, multivariate | a mean vector and a covariance matrix | planned |
| 4 | Extended Kalman filter (EKF) | same, but linearised via Jacobians | planned |
| 5 | Unscented Kalman filter (UKF) | same, but propagated via sigma points | planned |
| 6 | Particle filter (Monte Carlo Localization) | a few thousand weighted guesses | planned |
| 7 | Adaptive MCL on an occupancy grid | particles + a real laser scan model | planned |
| 8 | GNSS + IMU fusion | an EKF over position, velocity and bias | planned |

The order is deliberate. Days 1–3 are exact and closed-form. Day 4–5 are the two standard
ways to cope with the fact that real robots don't move in straight lines. Days 6–7 throw
out the single-hump assumption entirely and get back the flexibility of Day 1 at a cost
you control. Day 8 is the one that looks most like a real vehicle stack.

---

## The trade-off, in one table

This is the thread running through the whole folder:

| | Shape of belief it can represent | Cost | Handles "I'm completely lost" |
|---|---|---|---|
| Histogram filter | **anything** | explodes with dimensions | yes |
| Kalman family | **one hump only** (Gaussian) | tiny and fast | no |
| Particle filter | **anything** | you pick it (particle count) | yes |

A Kalman filter cannot say "I'm either at the lift or the fire exit, and nowhere in
between" — its belief is a single hump, so it would answer "somewhere in the middle",
which is the one place the robot definitely is not. A histogram or particle filter can.
That is the single most important idea in this folder, and Day 1 shows it directly.

---

## Running anything here

```bash
pip install -r ../requirements.txt
cd day01_histogram_filter_1d && python run_demo.py
```

Every day is self-contained: an algorithm file with no plotting in it, a `run_demo.py`
that simulates and draws, and a README that explains it from scratch with the animation
embedded.
