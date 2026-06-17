import os

import numpy as np
import gpytoolbox as gpy

from ..base import Adversary, _bbox_diagonal
from ..utils import random_rotation


def _unit_box():
    """Axis-aligned unit cube centered at the origin: ``(8, 3)`` V, ``(12, 3)`` F."""
    V = np.array([[x, y, z]
                  for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
                 float)
    F = np.array([
        [0, 1, 3], [0, 3, 2],   # x = -0.5
        [4, 7, 5], [4, 6, 7],   # x = +0.5
        [0, 5, 1], [0, 4, 5],   # y = -0.5
        [2, 3, 7], [2, 7, 6],   # y = +0.5
        [0, 2, 6], [0, 6, 4],   # z = -0.5
        [1, 5, 7], [1, 7, 3],   # z = +0.5
    ], np.int64)
    return V, F


class Disconnected(Adversary):
    """Hide the intact device among extra disconnected "junk" components.
    """

    def __init__(self, benign_dir=None, min_parts=1, max_parts=3,
                 size_lo=0.4, size_hi=1.5, spread=3.0, max_junk_faces=50000):
        # max_junk_faces matches the benign ABC facet cap (load_data._load_benign_abc),
        # so the half of benign_dir that is ABC CAD (median ~36k facets) qualifies as
        # junk rather than being rejected and skewing the junk back toward Thingi10K.
        self.benign_dir = benign_dir
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.size_lo = size_lo            # junk size as a fraction of device diag
        self.size_hi = size_hi
        self.spread = spread              # max offset, in device diagonals
        self.max_junk_faces = max_junk_faces
        self._pool = None                 # lazily-listed benign .obj paths

    def _benign_pool(self):
        if self._pool is None:
            if self.benign_dir and os.path.isdir(self.benign_dir):
                self._pool = [os.path.join(self.benign_dir, f)
                              for f in sorted(os.listdir(self.benign_dir))
                              if f.lower().endswith(".obj")]
            else:
                self._pool = []
        return self._pool

    def _junk_part(self, rng, diag):
        """A single junk component centered at the origin (benign shape or box).
        """
        pool = self._benign_pool()
        Vj = Fj = None
        if pool:
            for _ in range(4):            # a few tries in case a file is unreadable
                p = pool[int(rng.integers(len(pool)))]
                try:
                    v, f = gpy.read_mesh(p)
                    v = np.asarray(v, float)
                    f = np.asarray(f, np.int64)
                except Exception:
                    continue
                if (len(v) >= 3 and 0 < len(f) <= self.max_junk_faces
                        and np.all(np.isfinite(v))
                        and _bbox_diagonal(v) > 1e-9):
                    Vj, Fj = v, f
                    break
        if Vj is None:                    # fallback: a stretched random box
            Vj, Fj = _unit_box()
            Vj = Vj * rng.uniform(0.5, 2.0, size=3)
        # normalize to a random size relative to the device, then pose it
        d = _bbox_diagonal(Vj) or 1.0
        Vj = (Vj - Vj.mean(0)) * (rng.uniform(self.size_lo, self.size_hi) * diag / d)
        Vj = Vj @ random_rotation(rng).T
        return Vj, Fj

    def __call__(self, vertices, faces, rng):
        V = np.asarray(vertices, float).copy()
        F = np.asarray(faces, np.int64).copy()
        diag = _bbox_diagonal(V) or 1.0
        center = V.mean(0)

        Vs, Fs = [V], [F]
        nparts = int(rng.integers(self.min_parts, self.max_parts + 1))
        for _ in range(nparts):
            Vj, Fj = self._junk_part(rng, diag)
            offset = random_rotation(rng)[:, 0] * diag * rng.uniform(1.2, self.spread)
            Fs.append(Fj + sum(len(v) for v in Vs))   # reindex past everything so far
            Vs.append(Vj + center + offset)
        return np.vstack(Vs), np.vstack(Fs).astype(np.int64)
