import numpy as np

from ..base import Adversary, _bbox_diagonal


class Degenerate(Adversary):
    """This adversary keeps the vertices and faces exactly as-is and merely *appends* a random
    mix of garbage:
      * ``nonfinite``    -- a few NaN / +-inf vertices, plus faces that use them;
      * ``repeated``     -- zero-area faces with a repeated vertex index;
      * ``out_of_range`` -- faces indexing a vertex past the vertex count;
      * ``duplicate``    -- exact copies of existing device faces;
      * ``dangling``     -- unreferenced (used-by-nothing) vertices.
    """

    def __init__(self, p_each=0.6, max_extra_frac=0.5, max_dangling=200):
        self.p_each = p_each                  # prob. each trick is included
        self.max_extra_frac = max_extra_frac  # added bad faces, as frac of |F|
        self.max_dangling = max_dangling      # max added unreferenced vertices

    def __call__(self, vertices, faces, rng):
        V = np.asarray(vertices, float).copy()
        F = np.asarray(faces, np.int64).copy()
        nV0, nF0 = len(V), len(F)
        diag = _bbox_diagonal(V) or 1.0
        n_extra = max(1, int(self.max_extra_frac * max(nF0, 1)))

        tricks = ["nonfinite", "repeated", "out_of_range", "duplicate", "dangling"]
        chosen = [t for t in tricks if rng.random() < self.p_each]
        if not chosen:                        # never emit a clean device
            chosen = [tricks[int(rng.integers(len(tricks)))]]

        extra_faces = []

        if "nonfinite" in chosen:
            k = int(rng.integers(1, 6))
            bad = rng.choice(np.array([np.nan, np.inf, -np.inf]), size=(k, 3))
            base = len(V)
            V = np.vstack([V, bad])           # grow V; indices base..base+k-1
            tris = np.stack(
                [base + rng.integers(0, k, size=3) for _ in range(n_extra)]
            ).astype(np.int64)
            extra_faces.append(tris)

        if "repeated" in chosen:
            a = rng.integers(0, nV0, size=n_extra)
            b = rng.integers(0, nV0, size=n_extra)
            extra_faces.append(np.stack([a, a, b], axis=1).astype(np.int64))

        if "out_of_range" in chosen:
            lo = len(V)                        # past every real/garbage vertex
            extra_faces.append(
                rng.integers(lo, lo + 10 ** 6, size=(n_extra, 3)).astype(np.int64))

        if "duplicate" in chosen and nF0 > 0:
            idx = rng.integers(0, nF0, size=min(n_extra, nF0))
            extra_faces.append(F[idx].copy())

        if "dangling" in chosen:
            k = int(rng.integers(1, self.max_dangling + 1))
            pts = V[:nV0].mean(0) + rng.normal(0.0, diag, size=(k, 3))
            V = np.vstack([V, pts])            # referenced by nothing

        Fout = np.vstack([F] + extra_faces) if extra_faces else F
        return V, Fout.astype(np.int64)
