"""Base class and shared helpers for procedural adversaries (see README.md).
"""

import numpy as np


def _bbox_diagonal(vertices):
    """Length of the mesh bounding-box diagonal (a scale-invariant length unit)."""
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


class Adversary:
    """A procedural modification of a mesh.
    """

    def __call__(self, vertices, faces, rng):
        """Returns modified vertices and faces (see jitter/adversary.py for an example)."""
        raise NotImplementedError
