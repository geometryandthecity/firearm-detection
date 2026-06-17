"""Shared helpers for procedural adversaries (see README.md).

Free functions reused by more than one adversary live here so the behavior is
defined once. (The :class:`~adversaries.base.Adversary` base class and the
``_bbox_diagonal`` length unit stay in ``base.py``.)
"""

import numpy as np


def random_rotation(rng):
    """Uniform-random proper rotation matrix (``3x3``, determinant ``+1``).

    QR of a Gaussian matrix gives a Haar-uniform orthogonal matrix; multiplying
    by the signs of ``R``'s diagonal makes that factorization unique, and
    flipping one column whenever the determinant is negative discards the
    reflections, leaving a uniform rotation. Drawing from ``rng`` keeps it
    reproducible. Shared by the ``rigid`` and ``disconnected`` adversaries.
    """
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q
