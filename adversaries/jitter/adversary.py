import numpy as np

from ..base import Adversary, _bbox_diagonal


class Jitter(Adversary):
    """Gaussian vertex noise with standard deviation ``sigma_frac`` times the
    bounding-box diagonal (scale-invariant, so it stays small for any base)."""

    def __init__(self, sigma_frac=1e-3):
        self.sigma_frac = sigma_frac

    def __call__(self, vertices, faces, rng):
        sigma = self.sigma_frac * _bbox_diagonal(vertices)
        return vertices + rng.normal(0.0, sigma, size=vertices.shape), faces
