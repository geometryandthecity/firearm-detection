from ..base import Adversary, _bbox_diagonal
from ..utils import random_rotation


class RigidTransform(Adversary):
    """Random rotation plus translation."""

    def __init__(self, max_translation_frac=0.1):
        self.max_translation_frac = max_translation_frac

    def __call__(self, vertices, faces, rng):
        q = random_rotation(rng)
        centroid = vertices.mean(axis=0)
        rotated = (vertices - centroid) @ q.T + centroid
        t = rng.uniform(-1.0, 1.0, size=3) * self.max_translation_frac * _bbox_diagonal(vertices)
        return rotated + t, faces
