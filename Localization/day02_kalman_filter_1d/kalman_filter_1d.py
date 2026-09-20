"""
Day 2 - The Kalman filter in one dimension.

This file is only the algorithm. Nothing about drawing, nothing about the
simulated vehicle.

On day 1 the belief was one number for every cell of the world. Here it is two
numbers, a mean and a variance, which together describe a Gaussian. The same
two step loop applies, and each step turns out to be a single arithmetic
operation on Gaussians.

    predict  ->  ADD two Gaussians        (the belief and the motion)
    update   ->  MULTIPLY two Gaussians   (the belief and the measurement)

That really is the whole filter. The `Gaussian` class below overloads `+` and
`*` so the code can be written the way the sentence above reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["Gaussian", "predict", "update", "kalman_gain",
           "steady_state_variance", "steady_state_gain"]


@dataclass(frozen=True)
class Gaussian:
    """A one dimensional Gaussian, stored as a mean and a variance.

    Variance rather than standard deviation is stored because variance is what
    adds and combines cleanly. Standard deviation is offered as a property
    because it is what a person reads, being in metres rather than in metres
    squared.
    """

    mean: float
    var: float

    def __post_init__(self):
        if self.var <= 0.0:
            raise ValueError(f"variance must be positive, got {self.var}")

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def pdf(self, x):
        """Height of the bell curve at x. Used only for drawing."""
        return norm.pdf(x, self.mean, self.std)

    def interval(self, n_std: float = 2.0) -> tuple[float, float]:
        """The mean plus and minus n standard deviations."""
        half = n_std * self.std
        return self.mean - half, self.mean + half

    def __add__(self, other: "Gaussian") -> "Gaussian":
        """The PREDICT step. Adding two independent Gaussians adds both their
        means and their variances.

        The means adding is obvious. The variances adding is the important
        part, and it is why moving always makes the robot less certain. There
        is no way to add two positive variances and get a smaller one.
        """
        return Gaussian(self.mean + other.mean, self.var + other.var)

    def __mul__(self, other: "Gaussian") -> "Gaussian":
        """The UPDATE step. Multiplying two Gaussians and renormalising gives
        another Gaussian, which is the fact the whole filter rests on.

        Written with precisions, where precision is one over variance, the
        result is simple. Precisions add, and the new mean is the average of
        the two means weighted by their precisions.

        The new variance is smaller than either input, always. Two opinions
        combined are worth more than either one alone.
        """
        var = 1.0 / (1.0 / self.var + 1.0 / other.var)
        mean = var * (self.mean / self.var + other.mean / other.var)
        return Gaussian(mean, var)

    def __repr__(self) -> str:
        return f"N(mu={self.mean:.3f}, sigma={self.std:.3f})"


def predict(belief: Gaussian, motion: Gaussian) -> Gaussian:
    """Fold in a motion command. Uncertainty always grows."""
    return belief + motion


def kalman_gain(prior_var: float, meas_var: float) -> float:
    """How far to move the estimate toward the measurement, between 0 and 1.

    0 means ignore the measurement completely and keep the prediction.
    1 means throw the prediction away and believe the measurement.
    """
    return prior_var / (prior_var + meas_var)


def update(belief: Gaussian, measurement: Gaussian) -> tuple[Gaussian, float]:
    """Fold in a measurement. Uncertainty always shrinks.

    This is written in the classic Kalman gain form rather than as the product
    above, because the gain is the quantity worth watching. The two forms are
    algebraically identical, which `run_demo.py` checks numerically.

        K  = P / (P + R)
        mu = mu + K * (z - mu)        <- z - mu is called the innovation
        P  = (1 - K) * P
    """
    K = kalman_gain(belief.var, measurement.var)
    mean = belief.mean + K * (measurement.mean - belief.mean)
    var = (1.0 - K) * belief.var
    return Gaussian(mean, var), K


def steady_state_variance(q: float, r: float) -> float:
    """The variance the filter settles at, in closed form.

    After enough steps the growth from predict and the shrinkage from update
    cancel exactly, and the variance stops changing. Setting

        P = (P + q) r / (P + q + r)

    and rearranging gives the quadratic P^2 + P q - q r = 0, whose positive
    root is returned here.

    The striking part is what is absent. This value depends only on the process
    noise q and the measurement noise r. It does not depend on the initial
    guess, and it does not depend on any measurement the filter will ever
    receive. The accuracy of a Kalman filter can therefore be predicted before
    the vehicle is switched on.
    """
    return (-q + np.sqrt(q * q + 4.0 * q * r)) / 2.0


def steady_state_gain(q: float, r: float) -> float:
    """The Kalman gain the filter settles at."""
    p = steady_state_variance(q, r)
    return kalman_gain(p + q, r)
