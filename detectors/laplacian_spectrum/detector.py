import os

import numpy as np
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
import gpytoolbox as gpy

from ..base import Detector
from ..utils import split_components


def _spectrum(V, F, k):
    """Area-normalized ``k`` smallest non-zero Laplace-Beltrami eigenvalues, or ``None``.

    The signature of a (sub)mesh is the ``k`` smallest non-zero eigenvalues of the
    generalized eigenproblem ``L v = lambda M v`` (cotangent Laplacian ``L``,
    Voronoi mass matrix ``M``), each multiplied by the surface area -- which makes
    the signature invariant to rigid motion *and* uniform scale (the
    rigid/jitter/tetwild variants of a shape map to nearly the same signature).
    Meshes too coarse to yield ``k`` eigenvalues are upsampled (midpoint
    subdivision preserves the geometry). Returns ``None`` for anything that can't
    yield a clean spectrum, so the caller can skip that component.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(V) < 3 or len(F) == 0:
        return None
    try:
        # Drop ~zero-area (degenerate) faces before building the operators: a
        # single one makes the cotangent Laplacian singular and the eigensolve
        # fail (DLASCL error), so a clean rigid variant could fail while its
        # jittered twin -- whose noise gives the sliver a finite area -- succeeded.
        # The spectrum is intrinsic, so removing degenerate faces restores the
        # exact rigid invariance that kept rigid/faceswap AUC below 1.0.
        tri = V[F]
        twice_area = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if twice_area.size:
            F = F[twice_area > 1e-9 * float(twice_area.max())]
            if len(F) == 0:
                return None
            used = np.unique(F)
            if len(used) < 3:
                return None
            remap = np.full(len(V), -1, np.int64)
            remap[used] = np.arange(len(used))
            V, F = V[used], remap[F]
        while V.shape[0] < k + 8:
            V, F = gpy.subdivide(V, F, method="upsample")
        L = gpy.cotangent_laplacian(V, F).astype(np.float64)
        M = gpy.massmatrix(V, F).astype(np.float64)
        area = float(gpy.doublearea(V, F).sum()) / 2.0
        n_request = min(k + 5, L.shape[0] - 1)
        try:
            vals = sla.eigsh(L, M=M, k=n_request, sigma=1e-8, which="LM",
                             return_eigenvectors=False)
        except Exception:
            vals = sla.eigsh(L, M=M, k=n_request, which="SM",
                             return_eigenvectors=False)
        vals = np.sort(np.abs(vals))
        vals = vals[vals > 1e-6][:k]
        if vals.size < k:
            return None
        return vals * area
    except Exception:
        return None


class LaplacianSpectrum(Detector):
    """Detect malign shapes via the low end of their Laplace-Beltrami spectrum,
    analyzed per connected component and hardened against degenerate input.

    The geometric signature of a (sub)mesh is the area-normalized ``k`` smallest
    non-zero eigenvalues of ``L v = lambda M v`` (see ``_spectrum``), invariant to
    rigid motion and uniform scale. Two adversary families break a naive
    whole-mesh version of this detector, and this class defends against both:

    * **degenerate input** (the ``degenerate`` adversary appends NaN/out-of-range/
      repeated-index faces). A whole-mesh featurizer raises on these and falls back
      to ``score 0.0`` (benign) -- an evasion. Every mesh is therefore **sanitized**
      first (``utils.sanitize``, inside ``split_components``), which strips the
      garbage and leaves the real geometry, so the device's spectrum still computes.

    * **disconnected junk** (the ``disconnected`` adversary hides the device among
      extra benign components). A single whole-mesh spectrum is dominated by the
      junk and no longer matches the device. The mesh is therefore **split into
      connected components** (``utils.split_components``) and each is scored
      independently; the mesh's score is the max over components, so a small
      device hidden in a big carrier is still caught.

    Scoring is a **two-class** nearest-signature ratio. ``train`` stores the
    per-component signatures of *both* the device and the benign training shapes;
    a query component is scored ``d_benign / (d_benign + d_device)`` -- ~1 when it
    sits on a known device signature, ~0 when it sits on a benign one. The
    leaderboard AUC is invariant to the exact ratio; it only needs to rank
    device-bearing meshes above benign ones.

    One subtlety makes or breaks this: the ``disconnected`` adversary assembles a
    malign mesh out of the device *plus* a pile of benign junk components, so
    naively pooling every component of every malign training mesh poisons the
    "malign" set with benign-junk signatures -- a benign query sharing such a junk
    component would then score ~1.0 (a false positive). ``train`` therefore selects
    device exemplars **per mesh** (``_collect_device``): it keeps only each malign
    mesh's most-anomalous component -- the one *least* like the benign training set,
    i.e. the device hidden in the carrier. The junk, being benign-derived, is less
    anomalous than the device, so this argmax excludes it with no tuned threshold.
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self._malign_sigs = None         # (Nd, k) per-component DEVICE signatures
        self._benign_sigs = None         # (Nb, k) per-component benign signatures

    def _component_signatures(self, obj_path):
        """Sanitize, split into all connected components, and return their
        per-component signatures (dropping any that can't yield one)."""
        try:
            V, F = gpy.read_mesh(obj_path)
        except Exception:
            return []
        if V is None or F is None:
            return []
        # Score every component (split_components default); the shared per-shape
        # eval timeout bounds the work. CAVEAT: component flooding can exhaust that
        # budget -- see the README to-do on a device-likeness pre-filter.
        comps = split_components(np.asarray(V, float), np.asarray(F, np.int64))
        sigs = []
        for Vc, Fc in comps:
            s = _spectrum(Vc, Fc, self.k)
            if s is not None:
                sigs.append(s)
        return sigs

    def _collect(self, paths, label, log_every=200):
        """Pool the per-component signatures of every mesh in ``paths``."""
        sigs, total = [], len(paths)
        for i, p in enumerate(paths, start=1):
            sigs.extend(self._component_signatures(p))
            if i % log_every == 0 or i == total:
                print(f"[laplacian_spectrum] {label}: {i}/{total} meshes, "
                      f"{len(sigs)} component signatures", flush=True)
        return sigs

    def train(self, train, test=None):
        print(f"[laplacian_spectrum] training on {len(train.malign)} malign + "
              f"{len(train.benign)} benign meshes (k={self.k})", flush=True)
        # Benign first: picking each malign mesh's device component means measuring
        # how anomalous each component is *relative to the benign set*.
        ben = self._collect(train.benign, "benign")
        if not ben:
            raise RuntimeError(
                "LaplacianSpectrum.train needs benign component signatures (got none).")
        self._benign_sigs = np.asarray(ben, dtype=float)
        # From each malign mesh keep only its most-anomalous component -- the
        # device hidden among the carrier's benign junk.
        self._malign_sigs = self._collect_device(train.malign, "malign")
        if self._malign_sigs.shape[0] == 0:
            raise RuntimeError(
                "LaplacianSpectrum.train needs at least one usable device component.")
        print(f"[laplacian_spectrum] done: {self._malign_sigs.shape[0]} device + "
              f"{self._benign_sigs.shape[0]} benign component signatures cached",
              flush=True)

    def _collect_device(self, paths, label, log_every=200):
        """Select per-mesh device exemplars from malign meshes.

        The ``disconnected`` adversary builds a malign mesh by adding the device to
        a pile of *benign* junk components, so pooling every component of every
        malign mesh would poison the malign set with benign-junk signatures -- and a
        benign query sharing such a junk component would then score ~1.0 (a false
        positive). Every malign training mesh contains the device by construction,
        so we pick exemplars **per mesh**: keep each mesh's most-anomalous
        component -- the one *least* like anything benign (max nearest-benign
        distance), i.e. the device. The junk is benign-derived and therefore less
        anomalous, so this argmax excludes it with no tuned threshold."""
        btree = cKDTree(self._benign_sigs)
        dev, total = [], len(paths)
        for i, p in enumerate(paths, start=1):
            S = self._component_signatures(p)
            if S:
                S = np.asarray(S, dtype=float)
                d_nb, _ = btree.query(S, k=1)
                keep = d_nb >= d_nb.max()   # most-anomalous component(s) = the device
                dev.extend(S[keep])
            if i % log_every == 0 or i == total:
                print(f"[laplacian_spectrum] {label}: {i}/{total} meshes, "
                      f"{len(dev)} device signatures", flush=True)
        return (np.asarray(dev, dtype=float) if dev
                else np.empty((0, self.k), dtype=float))

    def _score_component(self, s):
        """Two-class nearest-signature ratio for one component signature ``s``:
        ``d_benign / (d_benign + d_device)`` -- ~1 on a device signature, ~0 on a
        benign one."""
        d_mal = float(np.linalg.norm(self._malign_sigs - s, axis=1).min())
        d_ben = float(np.linalg.norm(self._benign_sigs - s, axis=1).min())
        return d_ben / (d_ben + d_mal + 1e-12)

    def save(self) -> None:
        """Cache the per-component malign/benign signatures to ``trained_dir``."""
        if self._malign_sigs is None:
            raise RuntimeError("LaplacianSpectrum.save called before train().")
        np.savez(
            os.path.join(self.trained_dir, "signatures.npz"),
            malign_sigs=self._malign_sigs, benign_sigs=self._benign_sigs, k=self.k,
        )

    def load(self) -> None:
        """Restore the signatures saved by ``save``."""
        path = os.path.join(self.trained_dir, "signatures.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached signatures at {path}; run `python evaluate.py --train` first.")
        data = np.load(path)
        self._malign_sigs = data["malign_sigs"]
        self._benign_sigs = data["benign_sigs"]
        self.k = int(data["k"])

    def eval(self, obj_path) -> dict:
        if self._malign_sigs is None:
            return {"score": 0.0}
        sigs = self._component_signatures(obj_path)
        if not sigs:  # nothing survived sanitizing -> no device geometry -> benign
            return {"score": 0.0}
        return {"score": float(max(self._score_component(s) for s in sigs))}
