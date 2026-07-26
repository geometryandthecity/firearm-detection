"""D2 shape-distribution detector (Pham, Lee, Kwon & Kwon, "Anti-3D Weapon Model
Detection for Safe 3D Printing Based on Convolutional Neural Networks and D2 Shape
Distribution", *Symmetry* 2018, 10, 90).

The D2 shape distribution (Osada et al. 2002) is the probability distribution of
the Euclidean distance between two random points on a shape's surface. Sampling
``N`` such pairs, binning the distances into ``B`` bins over the shape's own
``[d_min, d_max]`` range and normalizing by ``N`` yields a length-``B`` *D2
vector* -- a global shape signature. The paper feeds that vector to a small 1-D
CNN (one convolution of kernel length ``m=5`` followed by two ``tanh`` hidden
layers) to classify weapon models; here the same feature + network is repurposed
as a benign/malign detector for the benchmark.

Why D2 fits this benchmark especially well: the distance distribution is
*intrinsically* invariant to rigid motion, and -- because every shape is binned
over its own ``[d_min, d_max]`` -- to uniform scale as well. So the ``rigid``,
``jitter``, ``faceswap`` and ``tetwild`` adversaries map a device to (very nearly)
the same D2 vector with **no data augmentation** at all, unlike the PointNet
baseline which must learn rotation invariance from augmented poses. The remaining
two adversary families are handled exactly as the other geometry detectors handle
them (``utils``): the mesh is **sanitized** (so the ``degenerate`` adversary's
NaN/out-of-range/duplicate faces don't make sampling raise and silently return
benign) and **split into connected components** scored independently with a
**max** pool (so the ``disconnected`` adversary's device-hidden-among-benign-junk
is still caught).

Faithfulness notes / deliberate adaptations for the benchmark:

* **Per-component MIL, not whole-model.** The paper scores one D2 vector per model
  (each model is a single weapon). Because the ``disconnected`` adversary bolts the
  device onto a pile of benign junk components, a single whole-mesh D2 would be
  dominated by the junk. We therefore compute one D2 vector per connected
  component and train the network on the **max** over component logits (max-pool
  multiple-instance learning, as in the PointNet baseline): the mesh is malign iff
  *any* component looks like the device, so no per-component device label is needed.
* **One malign logit, not two softmax outputs.** The paper's output layer has two
  neurons (firearm/knife); a binary detector needs a single malign score, so the
  head emits one logit scored through a sigmoid.
* **Cross-entropy, not MSE.** The paper trains with MSE; we use the numerically
  stabler ``BCEWithLogitsLoss``, standard for binary classification (the
  leaderboard AUC is invariant to this choice -- it only ranks shapes).
* **Sampling density.** The paper uses ``N = 1024 x 1024`` pairs and ``B = 1024``
  bins. We keep ``B = 1024`` and default ``n_pairs = 2**18``; at that density the
  D2 histogram is visually indistinguishable from the ``2**20`` one (the
  cross-seed L1 noise is ~0.07 over 1024 bins) at a quarter of the cost. Both are
  tunable via the constructor.

Runs on CPU or Apple-Silicon GPU (MPS) out of the box (``device="auto"`` prefers
cuda > mps > cpu); pass ``device="cuda"`` for an NVIDIA GPU. All randomness
(surface sampling, weight init, shuffling) is seeded from ``seed`` (ensemble model
``i`` uses ``seed + i``) so a run is reproducible.
"""
import os

# Let any op MPS hasn't implemented fall back to CPU, so the exact same code runs
# on Apple-Silicon GPU, plain CPU, or CUDA without special-casing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import gpytoolbox as gpy
import torch
from sklearn.metrics import roc_auc_score

from ..base import Detector
from ..utils import split_components


def _resolve_device(device):
    """Map ``"auto"`` to the best available backend (cuda > mps > cpu); anything
    else is taken as-is, so the same class runs on a MacBook today and a CUDA GPU
    later just by passing ``device="cuda"`` (or leaving ``"auto"``)."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _d2_vector(V, F, n_pairs, n_bins, rng):
    """The D2 shape distribution of one (sub)mesh as a length-``n_bins`` vector, or ``None``.

    Samples ``2 * n_pairs`` area-weighted random surface points, pairs them, takes
    the ``n_pairs`` Euclidean distances, and histograms them into ``n_bins`` bins
    over this mesh's own ``[d_min, d_max]`` (Eqs. 3-6 of the paper), normalized by
    ``n_pairs`` so the vector sums to 1 (Eqs. 9-10). Binning over the per-mesh
    range is what makes the vector invariant to uniform scale; using distances
    makes it invariant to rigid motion. Returns ``None`` for anything too small or
    degenerate to sample (the caller then skips that component).
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(V) < 3 or len(F) == 0:
        return None
    try:
        pts = np.asarray(gpy.random_points_on_mesh(V, F, 2 * n_pairs, rng=rng), float)
    except Exception:
        return None
    if pts.shape != (2 * n_pairs, 3):
        return None
    d = np.linalg.norm(pts[:n_pairs] - pts[n_pairs:], axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    dmin, dmax = float(d.min()), float(d.max())
    hist = np.zeros(n_bins, np.float64)
    if dmax <= dmin:
        # All distances coincide (e.g. a single degenerate point): one full bin.
        hist[0] = 1.0
        return hist.astype(np.float32)
    counts, _ = np.histogram(d, bins=n_bins, range=(dmin, dmax))
    hist = counts.astype(np.float64) / float(d.size)   # D2 vector: sums to 1
    return hist.astype(np.float32)


class _D2CNN(torch.nn.Module):
    """The paper's 1-D CNN over a D2 vector, emitting one malign logit.

    A single convolution of kernel length ``kernel`` (``m=5`` in the paper) slides
    over the length-``n_bins`` histogram, producing ``conv_channels x (n_bins -
    kernel + 1)`` "input neurons"; two ``tanh`` hidden layers (the paper's
    ``1020`` then ``15`` neurons) and a final linear unit map those to one logit.
    The paper's two softmax outputs (firearm/knife) collapse to this single
    malign logit for binary detection. ``hidden1`` defaults to ``n_bins - kernel + 1``,
    matching the paper where the first hidden layer equals ``N_input`` (for ``conv_channels=1``).

    A fixed per-bin **input standardization** ``(x - feat_mean) / feat_std``
    precedes the convolution. The raw D2 vector sums to 1 over ``n_bins`` bins, so
    every value is ~``1e-3``; fed directly into the ``tanh`` network that lands in a
    near-constant regime and the loss does not move (verified empirically: raw-D2
    train AUC ~0.84 with a near-constant output, versus 1.0 once standardized,
    matching a logistic-regression upper bound). ``feat_mean``/``feat_std`` are
    non-learnable buffers fit once on the training components (``set_norm``) and so
    persist in the checkpoint and apply identically at train and eval time.
    """

    def __init__(self, n_bins=1024, conv_channels=1, kernel=5,
                 hidden1=None, hidden2=15):
        super().__init__()
        conv_out = conv_channels * (n_bins - kernel + 1)
        if hidden1 is None:
            hidden1 = n_bins - kernel + 1     # paper: first hidden layer == N_input
        self.register_buffer("feat_mean", torch.zeros(n_bins))
        self.register_buffer("feat_std", torch.ones(n_bins))
        self.conv = torch.nn.Conv1d(1, conv_channels, kernel)   # valid conv, no pad
        self.mlp = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Flatten(),
            torch.nn.Linear(conv_out, hidden1), torch.nn.Tanh(),
            torch.nn.Linear(hidden1, hidden2), torch.nn.Tanh(),
            torch.nn.Linear(hidden2, 1),
        )

    def set_norm(self, mean, std):
        """Install the per-bin standardization fit on the training components.
        Bins with ~zero variance keep std=1 (centered, not blown up)."""
        std = np.asarray(std, np.float32).copy()
        std[std < 1e-8] = 1.0
        self.feat_mean.copy_(torch.as_tensor(np.asarray(mean, np.float32)))
        self.feat_std.copy_(torch.as_tensor(std))

    def forward(self, x):                      # x: (B, n_bins)
        x = (x - self.feat_mean) / self.feat_std
        x = self.conv(x.unsqueeze(1))          # (B, conv_channels, n_bins-kernel+1)
        return self.mlp(x).squeeze(1)          # (B,): one logit per D2 vector


class ShapeDistribution(Detector):
    """Detect malign shapes from their D2 shape distribution with the paper's CNN,
    analyzed per connected component (max-pool MIL) and hardened against degenerate
    input. See the module docstring for the method and the benchmark adaptations.

    Like the other geometry detectors, the mesh is sanitized and split into
    components (``utils.split_components``, ``maxc`` cap defeats component
    flooding); each component's D2 vector is scored by the CNN and the mesh score
    is the **max** over components. An ``n_models`` seed ensemble (the D2 features
    are sampled once and shared across ensemble members, so only the cheap CNN
    optimization repeats) averages out the idiosyncratic false positive a single
    seed can keep. No pose/scale augmentation is used: the D2 vector is already
    rigid- and scale-invariant.
    """

    def __init__(self, n_pairs=2**18, n_bins=1024, conv_channels=1, kernel=5,
                 hidden2=15, epochs=60, batch_size=16, lr=1e-3, device="auto",
                 seed=0, maxc=16, n_models=3, cosine=True):
        super().__init__()
        self.n_pairs = n_pairs           # distance pairs sampled per component
        self.n_bins = n_bins             # D2-vector length (histogram bins)
        self.conv_channels = conv_channels
        self.kernel = kernel
        self.hidden2 = hidden2
        self.epochs = epochs
        self.batch_size = batch_size     # meshes (bags) per optimizer step
        self.lr = lr
        self.device = _resolve_device(device)
        self.seed = seed
        self.maxc = maxc
        self.n_models = n_models         # seed-ensemble size (sigmoid scores averaged)
        self.cosine = cosine             # cosine-anneal the learning rate to 0
        self._models = []

    # ---- feature extraction -------------------------------------------------
    def _bag(self, obj_path):
        """The list of per-component D2 vectors for ``obj_path``, or ``None``.

        Sanitizes and splits the mesh (``split_components``) then computes one D2
        vector per component. ``None`` means nothing usable survived (no geometry
        to score -> benign). A fresh ``seed``-derived RNG per mesh keeps sampling
        reproducible and gives a transformed copy of a device the same draw."""
        try:
            V, F = gpy.read_mesh(obj_path)
        except Exception:
            return None
        if V is None or F is None:
            return None
        comps = split_components(np.asarray(V, float), np.asarray(F, np.int64),
                                 maxc=self.maxc)
        rng = np.random.default_rng(self.seed)
        vecs = []
        for Vc, Fc in comps:
            v = _d2_vector(Vc, Fc, self.n_pairs, self.n_bins, rng)
            if v is not None:
                vecs.append(v)
        return vecs or None

    def _load_bags(self, split, log_every=400):
        """Compute every usable mesh in ``split`` into a (bag, label) list, where a
        bag is a ``(n_comp, n_bins)`` array of its component D2 vectors."""
        bags = []
        for label, paths in ((1.0, split.malign), (0.0, split.benign)):
            total = len(paths)
            for i, p in enumerate(paths, start=1):
                vs = self._bag(p)
                if vs:
                    bags.append((np.stack(vs), label))
                if i % log_every == 0 or i == total:
                    print(f"[shape_distribution] D2 {'malign' if label else 'benign'}: "
                          f"{i}/{total} meshes, {len(bags)} bags", flush=True)
        return bags

    # ---- training / scoring -------------------------------------------------
    def _bag_logits(self, model, bags_arr):
        """Per-mesh logits = max over each bag's component logits.

        All components across the batch are concatenated into one forward pass,
        then scattered back and max-pooled per bag. No augmentation: the D2 vector
        is rigid- and scale-invariant, so the canonical feature is the only one."""
        sizes = [a.shape[0] for a in bags_arr]
        X = torch.from_numpy(np.concatenate(bags_arr, axis=0)).to(self.device)
        logits = model(X)                                  # (sum sizes,)
        out, off = [], 0
        for s in sizes:
            out.append(logits[off:off + s].max())
            off += s
        return torch.stack(out)

    def train(self, train, test=None):
        print(f"[shape_distribution] training {self.n_models} model(s) on "
              f"{len(train.malign)} malign + {len(train.benign)} benign meshes "
              f"(n_pairs={self.n_pairs}, n_bins={self.n_bins}, maxc={self.maxc}, "
              f"epochs={self.epochs})", flush=True)
        bags = self._load_bags(train)
        if not bags:
            raise RuntimeError("ShapeDistribution.train found no usable training meshes.")
        labels = np.array([lab for _, lab in bags], np.float32)
        n = len(bags)

        # Fit the per-bin input standardization once on every training component's
        # D2 vector (both classes pooled); shared by every ensemble member. Without
        # it the ~1e-3 D2 values keep the tanh network in a near-constant regime.
        allc = np.concatenate([b for b, _ in bags], axis=0)
        feat_mean, feat_std = allc.mean(0), allc.std(0)

        # Train an ensemble of identical networks from consecutive seeds; their
        # scores are averaged at eval. The expensive D2 sampling is done once above
        # and shared, so only the (cheap) optimization repeats per model.
        models = []
        for mi in range(self.n_models):
            seed = self.seed + mi
            _seed_everything(seed)
            model = _D2CNN(self.n_bins, self.conv_channels, self.kernel,
                           hidden2=self.hidden2).to(self.device)
            model.set_norm(feat_mean, feat_std)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
                     if self.cosine else None)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            rng = np.random.default_rng(seed)

            for epoch in range(self.epochs):
                model.train()
                order = rng.permutation(n)
                running, nb = 0.0, 0
                for s in range(0, n, self.batch_size):
                    idx = order[s:s + self.batch_size]
                    bags_arr = [bags[k][0] for k in idx]
                    yb = torch.from_numpy(labels[idx]).to(self.device)
                    logits = self._bag_logits(model, bags_arr)
                    loss = loss_fn(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running += float(loss.detach())
                    nb += 1
                if sched is not None:
                    sched.step()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"[shape_distribution] model {mi + 1}/{self.n_models} "
                          f"epoch {epoch + 1}/{self.epochs} "
                          f"loss={running / max(nb, 1):.4f}", flush=True)
            model.eval()
            models.append(model)
        self._models = models

        # Informational only: validation AUC on the test split (not used to tune).
        if test is not None:
            tb = self._load_bags(test)
            if tb and len({lab for _, lab in tb}) == 2:
                scores = self._scores([a for a, _ in tb])
                yte = np.array([lab for _, lab in tb])
                print(f"[shape_distribution] validation AUC="
                      f"{roc_auc_score(yte, scores):.4f}", flush=True)

    @torch.no_grad()
    def _scores(self, bag_arrays):
        """Ensemble malign scores for a list of bag arrays: each model's sigmoid of
        the max-pooled bag logit, averaged over the ``n_models`` models."""
        if not self._models or not bag_arrays:
            return np.empty((0,), np.float32)
        per_model = []
        for model in self._models:
            model.eval()
            out = []
            for s in range(0, len(bag_arrays), self.batch_size):
                chunk = bag_arrays[s:s + self.batch_size]
                out.append(torch.sigmoid(self._bag_logits(model, chunk)).cpu())
            per_model.append(torch.cat(out).numpy())
        return np.mean(np.stack(per_model, axis=0), axis=0)

    # ---- persistence --------------------------------------------------------
    def save(self) -> None:
        """Cache the trained ensemble weights plus the geometry/feature
        hyperparameters ``eval`` needs to rebuild the feature and network."""
        if not self._models:
            raise RuntimeError("ShapeDistribution.save called before train().")
        torch.save(
            {"state_dicts": [m.state_dict() for m in self._models],
             "n_pairs": self.n_pairs, "n_bins": self.n_bins,
             "conv_channels": self.conv_channels, "kernel": self.kernel,
             # hidden1 is derived (n_bins-kernel+1) today, but persist it so a
             # future hidden1 knob can't desync the saved weight shapes from load().
             "hidden1": self.n_bins - self.kernel + 1,
             "hidden2": self.hidden2, "maxc": self.maxc,
             # seed drives the D2 surface sampling; persist it so the no-arg
             # cls()+load() eval path samples features with the trained RNG (matters
             # only for non-default seeds -- D2 is near seed-invariant otherwise).
             "seed": self.seed,
             "n_models": len(self._models)},
            os.path.join(self.trained_dir, "model.pt"),
        )

    def load(self) -> None:
        """Rebuild the ensemble (and feature hyperparameters) from the checkpoint."""
        path = os.path.join(self.trained_dir, "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached model at {path}; run `python evaluate.py --train` first.")
        ckpt = torch.load(path, map_location=self.device)
        self.n_pairs = int(ckpt["n_pairs"])
        self.n_bins = int(ckpt["n_bins"])
        self.conv_channels = int(ckpt.get("conv_channels", self.conv_channels))
        self.kernel = int(ckpt.get("kernel", self.kernel))
        self.hidden2 = int(ckpt.get("hidden2", self.hidden2))
        self.maxc = int(ckpt.get("maxc", self.maxc))
        self.seed = int(ckpt.get("seed", self.seed))
        hidden1 = int(ckpt.get("hidden1", self.n_bins - self.kernel + 1))
        models = []
        for sd in ckpt["state_dicts"]:
            model = _D2CNN(self.n_bins, self.conv_channels, self.kernel,
                           hidden1=hidden1, hidden2=self.hidden2).to(self.device)
            model.load_state_dict(sd)
            model.eval()
            models.append(model)
        self._models = models
        self.n_models = len(models)

    def eval(self, obj_path) -> dict:
        if not self._models:
            return {"score": 0.0}
        vs = self._bag(obj_path)
        if vs is None:  # nothing survived sanitizing -> no device geometry -> benign
            return {"score": 0.0}
        with torch.no_grad():
            X = torch.from_numpy(np.stack(vs)).to(self.device)
            scores = [torch.sigmoid(m(X).max()).item() for m in self._models]
        return {"score": float(np.mean(scores))}
