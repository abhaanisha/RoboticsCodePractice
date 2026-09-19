"""
Day 1 - The Histogram Filter (a.k.a. the Discrete Bayes Filter) in 1D.

This file is *only* the algorithm. Nothing about drawing, nothing about the
simulated robot. Two functions carry the whole idea:

    sense()    <- fold a sensor reading into the belief   (Bayes' rule)
    predict()  <- fold a motion command into the belief   (total probability)

A "belief" here is just a probability distribution over where the robot might
be, stored as a numpy array with one entry per cell of the corridor. It always
sums to 1.0.

The corridor is treated as *cyclic* (cell N-1 is next to cell 0), which is what
lets us use np.roll and keeps the code to a few lines. Everything generalises
to a non-cyclic corridor by replacing np.roll with a clipped shift.
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_belief", "sense", "predict", "entropy", "MotionModel"]


def uniform_belief(n_cells: int) -> np.ndarray:
    """The state of total ignorance: every cell equally likely.

    This is where the robot starts on a "kidnapped robot" problem - it has been
    switched on somewhere in the corridor and has no idea where.
    """
    return np.full(n_cells, 1.0 / n_cells)


def sense(belief: np.ndarray, world: np.ndarray, z: bool,
          prob_hit: float, prob_miss: float) -> np.ndarray:
    """Update the belief with one sensor reading. This is Bayes' rule.

        posterior(x) proportional to  p(z | x) * prior(x)

    Args:
        belief:    prior belief over cells, shape (N,), sums to 1.
        world:     the map. world[i] is True if cell i contains a door.
        z:         what the sensor actually reported (True = "I see a door").
        prob_hit:  p(reading is correct). e.g. 0.9
        prob_miss: p(reading is wrong).   e.g. 0.1

    Returns:
        The posterior belief, renormalised to sum to 1.

    Intuition: cells that agree with what we just saw get their probability
    multiplied by 0.9; cells that disagree get multiplied by 0.1. Then we
    rescale so the numbers add to 1 again. Information *sharpens* the belief.
    """
    # p(z | x) evaluated at every cell at once.
    likelihood = np.where(world == z, prob_hit, prob_miss)

    posterior = belief * likelihood            # Bayes' numerator
    evidence = posterior.sum()                 # p(z), the normaliser
    if evidence == 0.0:
        raise ValueError("Impossible measurement: every cell has zero likelihood.")
    return posterior / evidence


def predict(belief: np.ndarray, u: int,
            kernel: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Update the belief after commanding the robot to move `u` cells.

        prior(x) = sum over x'  of  p(x | x', u) * belief(x')

    Args:
        belief:  belief before moving, shape (N,).
        u:       commanded displacement in cells (can be negative).
        kernel:  probabilities of the motion outcomes, e.g. [0.1, 0.8, 0.1].
        offsets: the displacement error each kernel entry refers to,
                 e.g. [-1, 0, +1] meaning "undershot / exact / overshot".

    Returns:
        The predicted ("prior") belief.

    Intuition: the robot's wheels slip. If we told it to go 2 cells, it
    probably went 2, but it might have gone 1 or 3. So we take the old belief,
    slide a copy of it by each possible distance, weight each copy by how
    likely that outcome is, and add them up. This is a convolution, and it
    always *blurs* the belief. Moving makes the robot less sure where it is.
    """
    if not np.isclose(kernel.sum(), 1.0):
        raise ValueError("Motion kernel must sum to 1.")

    prior = np.zeros_like(belief)
    for weight, offset in zip(kernel, offsets):
        # np.roll(belief, s)[j] == belief[j - s], i.e. mass at j came from j-s.
        prior += weight * np.roll(belief, u + offset)
    return prior


def entropy(belief: np.ndarray) -> float:
    """Shannon entropy in bits - a single number for "how lost am I?".

    log2(N) bits when the belief is uniform (maximally lost), 0 bits when the
    robot is certain. Handy for plotting convergence.
    """
    p = belief[belief > 0.0]
    return float(-np.sum(p * np.log2(p)))


class MotionModel:
    """Bundles the motion kernel so the demo code stays readable."""

    def __init__(self, p_exact: float = 0.8, p_under: float = 0.1, p_over: float = 0.1):
        total = p_under + p_exact + p_over
        if not np.isclose(total, 1.0):
            raise ValueError(f"Motion probabilities must sum to 1, got {total}.")
        self.kernel = np.array([p_under, p_exact, p_over])
        self.offsets = np.array([-1, 0, 1])

    def apply(self, belief: np.ndarray, u: int) -> np.ndarray:
        return predict(belief, u, self.kernel, self.offsets)

    def sample(self, u: int, rng: np.random.Generator) -> int:
        """Draw one *actual* displacement for the simulated robot."""
        return int(u + rng.choice(self.offsets, p=self.kernel))
