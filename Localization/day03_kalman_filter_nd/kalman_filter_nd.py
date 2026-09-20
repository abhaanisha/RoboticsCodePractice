"""
Day 3 - The Kalman filter in several dimensions.

This file is only the algorithm. Nothing about drawing, nothing about the
simulated vehicle.

Day 2 stored the belief as a mean and a variance, two scalars. Here the mean
becomes a vector and the variance becomes a covariance matrix. The two steps
are the same two steps, written with matrices.

    predict   x = F x + B u            P = F P F' + Q
    update    x = x + K (z - H x)      P = (I - K H) P

The new thing is not the matrices. It is the off diagonal entries of P. A
scalar variance can only say how uncertain one quantity is. A covariance
matrix can also say how the errors in two quantities are related, and that is
what lets the filter work out a quantity it never measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.stats import chi2, multivariate_normal

__all__ = ["GaussianState", "predict", "update", "UpdateResult",
           "steady_state", "ellipse_coverage"]


@dataclass(frozen=True)
class GaussianState:
    """A Gaussian over several variables, as a mean vector and a covariance."""

    mean: np.ndarray            # shape (n,)
    cov: np.ndarray             # shape (n, n), symmetric positive definite

    def __post_init__(self):
        object.__setattr__(self, "mean", np.atleast_1d(np.asarray(self.mean, float)))
        object.__setattr__(self, "cov", np.atleast_2d(np.asarray(self.cov, float)))
        n = self.mean.size
        if self.cov.shape != (n, n):
            raise ValueError(f"cov must be {n} by {n}, got {self.cov.shape}")

    @property
    def std(self) -> np.ndarray:
        """One standard deviation for each variable on its own."""
        return np.sqrt(np.diag(self.cov))

    @property
    def correlation(self) -> np.ndarray:
        """The covariance rescaled so every entry lies between -1 and 1.

        The off diagonal entry is the interesting one. A value near zero means
        the errors in the two variables are unrelated. A value near one means
        that being wrong about the first in one direction almost guarantees
        being wrong about the second in a matching direction.
        """
        s = self.std
        return self.cov / np.outer(s, s)

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Height of the hill at the given points. Used only for drawing."""
        return multivariate_normal(self.mean, self.cov, allow_singular=True).pdf(points)

    def ellipse(self, n_std: float = 2.0, dims=(0, 1), n_points: int = 200):
        """Points tracing the n sigma ellipse of two of the variables.

        The ellipse is the set of points whose Mahalanobis distance from the
        mean equals n_std. Its axes are the eigenvectors of the covariance and
        its half widths are n_std times the square roots of the eigenvalues,
        so the tilt of the ellipse is a direct picture of the correlation.

        Note that an n sigma ellipse in two dimensions does not contain the
        same fraction of the probability as an n sigma interval does in one.
        Use `ellipse_coverage` to get the honest number.
        """
        i, j = dims
        sub = self.cov[np.ix_([i, j], [i, j])]
        vals, vecs = np.linalg.eigh(sub)
        vals = np.maximum(vals, 1e-15)
        t = np.linspace(0.0, 2.0 * np.pi, n_points)
        unit = np.stack([np.cos(t), np.sin(t)])
        pts = vecs @ (n_std * np.sqrt(vals)[:, None] * unit)
        return pts[0] + self.mean[i], pts[1] + self.mean[j]

    def __repr__(self) -> str:
        m = ", ".join(f"{v:.3f}" for v in self.mean)
        s = ", ".join(f"{v:.3f}" for v in self.std)
        return f"Gaussian(mean=[{m}], sigma=[{s}])"


def ellipse_coverage(n_std: float, dim: int = 2) -> float:
    """Fraction of the probability inside an n sigma ellipse.

    In one dimension two sigma holds 95.4 percent. In two dimensions the same
    two sigma ellipse holds only 86.5 percent, because probability can escape
    in more directions. Quoting the one dimensional number for a
    two dimensional ellipse is a common and quietly misleading mistake.
    """
    return float(chi2.cdf(n_std ** 2, dim))


def predict(state: GaussianState, F: np.ndarray, Q: np.ndarray,
            B: np.ndarray | None = None, u: np.ndarray | None = None
            ) -> GaussianState:
    """Push the belief forward through the motion model.

        x = F x + B u
        P = F P F' + Q

    F describes how the state evolves on its own. For a vehicle tracked by
    position and velocity, F says that the new position is the old position
    plus velocity times the time step, which is the line that couples the two
    variables together.

    That coupling is what makes `F P F'` interesting. Even if P starts
    perfectly diagonal, meaning the two errors are unrelated, the product
    F P F' generally is not diagonal. The predict step manufactures
    correlation out of nothing, and the update step then spends it.
    """
    mean = F @ state.mean
    if B is not None and u is not None:
        mean = mean + B @ np.atleast_1d(u)
    cov = F @ state.cov @ F.T + Q
    return GaussianState(mean, 0.5 * (cov + cov.T))      # keep it symmetric


@dataclass(frozen=True)
class UpdateResult:
    state: GaussianState
    gain: np.ndarray            # K, shape (n, m)
    innovation: np.ndarray      # z - H x, how much the reading surprised us
    innovation_cov: np.ndarray  # S, how surprised we expected to be


def update(state: GaussianState, z, H: np.ndarray, R: np.ndarray) -> UpdateResult:
    """Fold in a measurement.

        y = z - H x                 the innovation
        S = H P H' + R              how large an innovation was expected
        K = P H' inv(S)             the Kalman gain
        x = x + K y
        P = (I - K H) P

    H selects what the sensor actually observes. For a receiver that reports
    position only, H is the single row [1, 0], and the velocity is never
    touched directly by z.

    Velocity still improves. The gain K is P H' inv(S), and the H' picks out
    the column of P belonging to position. That column contains the
    position and velocity covariance, so a non zero correlation puts a non
    zero entry in the velocity row of K. The filter corrects a quantity it
    cannot see, by way of its relationship to one it can.
    """
    z = np.atleast_1d(np.asarray(z, float))
    y = z - H @ state.mean
    S = H @ state.cov @ H.T + R
    K = state.cov @ H.T @ np.linalg.inv(S)

    mean = state.mean + K @ y
    n = state.mean.size
    A = np.eye(n) - K @ H
    # Joseph form. Algebraically equal to (I - K H) P, but it stays symmetric
    # and positive definite after many steps of floating point arithmetic.
    cov = A @ state.cov @ A.T + K @ R @ K.T
    return UpdateResult(GaussianState(mean, 0.5 * (cov + cov.T)), K, y, S)


def steady_state(F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """The covariance the filter settles at, in closed form.

    Day 2 solved a quadratic for this. With matrices the same balance point is
    the discrete algebraic Riccati equation, and SciPy solves it directly, so
    there is no need to run the filter to find out how well it will do.

    Returns the settled prior covariance, the settled posterior covariance and
    the settled gain.
    """
    P_prior = solve_discrete_are(F.T, H.T, Q, R)
    S = H @ P_prior @ H.T + R
    K = P_prior @ H.T @ np.linalg.inv(S)
    P_post = P_prior - K @ H @ P_prior
    return P_prior, 0.5 * (P_post + P_post.T), K
