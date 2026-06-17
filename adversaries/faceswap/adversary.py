from ..base import Adversary


class FaceSwap(Adversary):
    """Reorder faces by swapping a random number of random row pairs.
    """

    def __init__(self, max_swaps=None):
        self.max_swaps = max_swaps

    def __call__(self, vertices, faces, rng):
        faces = faces.copy()
        m = faces.shape[0]
        if m >= 2:
            hi = m - 1 if self.max_swaps is None else min(self.max_swaps, m - 1)
            for _ in range(int(rng.integers(1, hi + 1))):
                i, j = rng.choice(m, size=2, replace=False)
                faces[[i, j]] = faces[[j, i]]
        return vertices, faces
