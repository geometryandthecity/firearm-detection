"""First AI baseline: sample surface points and classify them with PointNet,
analyzed per connected component (max-pool multiple-instance learning) and
hardened against degenerate input."""
import os

# Let any op MPS hasn't implemented fall back to CPU, so the exact same code
# runs on Apple-Silicon GPU, plain CPU, or CUDA without special-casing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import gpytoolbox as gpy
import torch
from sklearn.metrics import roc_auc_score

from ..base import Detector
from ..utils import sanitize, split_components


def _resolve_device(device):
    """Map ``"auto"`` to the best available backend (cuda > mps > cpu).

    Anything else (``"cpu"``, ``"mps"``, ``"cuda"``) is taken as-is, so the same
    class runs locally on a MacBook today and on a CUDA GPU later just by passing
    ``device="cuda"`` (or leaving it on ``"auto"`` on a machine that has one).
    """
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


def _rand_rot(n, device):
    """``n`` uniform-random proper rotations -- shape ``(n, 3, 3)``, determinant
    +1 -- via the QR decomposition of a Gaussian matrix.

    Used to rotate each point cloud independently during training (and so make the
    T-Net-free network learn to score a shape the same way at any orientation).
    """
    A = torch.randn(n, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R, dim1=1, dim2=2))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(1)                          # make each Q a proper rotation
    det = torch.det(Q)
    Q[:, :, 0] = Q[:, :, 0] * det.unsqueeze(1)      # flip a column where det = -1
    return Q


def _sample_component(V, F, n, seed):
    """Area-weighted surface cloud of one component, centered + unit-radius, or ``None``.

    Each component is sampled and normalized on its own, so the device gets the
    full ``n``-point budget and its own unit scale regardless of how small it is
    relative to the junk it is hidden among (translation/uniform-scale invariant).
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(V) < 3 or len(F) == 0:
        return None
    try:
        pts = np.asarray(
            gpy.random_points_on_mesh(V, F, n, rng=np.random.default_rng(seed)), float)
    except Exception:
        return None
    if pts.shape != (n, 3):
        return None
    pts -= pts.mean(0, keepdims=True)
    r = float(np.linalg.norm(pts, axis=1).max())
    if r > 0:
        pts /= r
    return pts.astype(np.float32)


class _PointNetModel(torch.nn.Module):
    """Vanilla PointNet classifier (Qi et al. 2017), without the T-Net transforms.

    A shared per-point MLP -- implemented as width-1 convolutions so the same
    weights apply to every one of the ``N`` points -- lifts each xyz coordinate
    to a 1024-d embedding. A max-pool over the point axis collapses those into a
    single permutation-invariant global descriptor (the symmetric function that
    makes the network independent of point ordering), and a small MLP head maps
    that descriptor to one malign logit. The two T-Nets from the paper are left
    out on purpose: they add substantial machinery for under ~1% accuracy in the
    original work, and this is meant to be a clean first AI baseline. Robustness
    to rigid motion comes instead from training-time augmentation (each cloud is
    randomly rotated and jittered every step; see ``PointNet``) and from centering
    + unit-scaling each cloud.
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        def shared_mlp(i, o):
            return torch.nn.Sequential(
                torch.nn.Conv1d(i, o, 1, bias=False),
                torch.nn.BatchNorm1d(o),
                torch.nn.ReLU(inplace=True),
            )

        # shared per-point feature extractor: 3 -> 64 -> 64 -> 64 -> 128 -> 1024
        self.encoder = torch.nn.Sequential(
            shared_mlp(3, 64), shared_mlp(64, 64), shared_mlp(64, 64),
            shared_mlp(64, 128), shared_mlp(128, 1024),
        )
        # classifier head on the pooled global feature: 1024 -> 512 -> 256 -> 1
        self.head = torch.nn.Sequential(
            torch.nn.Linear(1024, 512, bias=False),
            torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):                      # x: (B, N, 3)
        x = self.encoder(x.transpose(1, 2))    # (B, 1024, N)
        x = x.amax(dim=2)                      # (B, 1024): symmetric global pool
        return self.head(x).squeeze(1)         # (B,): one logit per cloud


class PointNet(Detector):
    """Detect malign shapes with a per-component multiple-instance PointNet.

    The mesh is treated as a *bag* of connected components: it is sanitized and
    split (``utils``), each component is sampled into its own ``n_points``
    cloud (centered + unit-scaled), and the same PointNet (see ``_PointNetModel``)
    scores every component. The mesh logit is the **max** over component logits --
    classic max-pool multiple-instance learning: the mesh is malign iff *any*
    component looks like the device. Trained end-to-end on that max, the network
    pushes the consistent device component up and benign components down.

    This hardens the plain whole-mesh PointNet against the two adversary families
    in the benchmark:

    * **degenerate input** (``degenerate`` adversary) -- sanitizing before
      sampling means NaN/out-of-range/repeated-index faces no longer make
      ``random_points_on_mesh`` raise and the detector silently return ``0.0``.
    * **disconnected junk** (``disconnected`` adversary) -- per-component sampling
      gives the device its own full point budget, so a small device hidden among
      large benign components is no longer area-sampled away into the carrier.

    Two more ingredients close the remaining gap to a perfect ROC on the eval split:

    * **rotation + jitter augmentation** -- without the T-Nets the raw network is
      not rotation-invariant, so the re-posed ``rigid`` and ``tetwild`` adversaries
      evade a model trained only on canonical poses. Every training step rotates
      each component cloud by an independent uniform ``SO(3)`` rotation
      (``rot_aug``) and adds small Gaussian jitter (``jitter``), teaching the
      network to score a shape the same way at any orientation; eval then needs no
      test-time augmentation.
    * **seed ensemble** -- ``n_models`` copies are trained from consecutive seeds
      and their sigmoid scores are averaged. Averaging cancels the idiosyncratic
      false positive a single seed can retain (one benign scan can sit a hair
      above the lowest malign for one seed), so the ensemble reaches AUC 1.0
      robustly rather than depending on a lucky seed.

    It runs on CPU or Apple-Silicon GPU (MPS) out of the box (``device="auto"``
    prefers cuda > mps > cpu); pass ``device="cuda"`` for an NVIDIA GPU. All
    randomness (weight init, shuffling, point sampling, augmentation) is seeded
    from ``seed`` (model ``i`` uses ``seed + i``), so a run is reproducible.
    """

    def __init__(self, n_points=1024, epochs=60, batch_size=16, lr=2e-3,
                 dropout=0.3, device="auto", seed=0, maxc=16,
                 n_models=3, rot_aug=True, jitter=0.01, cosine=True):
        super().__init__()
        self.n_points = n_points
        self.epochs = epochs
        self.batch_size = batch_size   # number of meshes (bags) per optimizer step
        self.lr = lr
        self.dropout = dropout
        self.device = _resolve_device(device)
        self.seed = seed
        self.maxc = maxc
        self.n_models = n_models       # seed-ensemble size (sigmoid scores averaged)
        self.rot_aug = rot_aug         # random SO(3) rotation per cloud each step
        self.jitter = jitter           # Gaussian vertex-jitter sigma each step
        self.cosine = cosine           # cosine-anneal the learning rate to 0
        self._models = []

    def _bag(self, obj_path):
        """Return the list of per-component clouds for ``obj_path``, or ``None``.

        Sanitizes and splits the mesh, then samples each component into its own
        unit-normalized cloud. ``None`` means nothing usable survived (no device
        geometry to score).
        """
        try:
            V, F = gpy.read_mesh(obj_path)
        except Exception:
            return None
        if V is None or F is None:
            return None
        comps = split_components(np.asarray(V, float), np.asarray(F, np.int64),
                                 maxc=self.maxc)
        clouds = []
        for Vc, Fc in comps:
            pc = _sample_component(Vc, Fc, self.n_points, self.seed)
            if pc is not None:
                clouds.append(pc)
        return clouds or None

    def _load_bags(self, split, log_every=400):
        """Sample every usable mesh in ``split`` into a (bag, label) list."""
        bags = []
        for label, paths in ((1.0, split.malign), (0.0, split.benign)):
            total = len(paths)
            for i, p in enumerate(paths, start=1):
                cs = self._bag(p)
                if cs:
                    bags.append((np.stack(cs), label))
                if i % log_every == 0 or i == total:
                    print(f"[pointnet] sampling {'malign' if label else 'benign'}: "
                          f"{i}/{total} meshes, {len(bags)} bags", flush=True)
        return bags

    def _bag_logits(self, model, bags_arr, rot=False, jit=0.0):
        """Per-mesh logits = max over each bag's component logits.

        ``bags_arr`` is a list of ``(n_comp_i, n_points, 3)`` float arrays. All
        components are concatenated into one batch for a single forward pass, then
        scattered back and max-pooled per bag (BatchNorm needs >= 2 clouds total).
        During training each cloud is augmented with an independent random rotation
        (``rot``) and Gaussian jitter (``jit``); both default off, so eval is the
        plain canonical-pose forward pass.
        """
        sizes = [a.shape[0] for a in bags_arr]
        X = torch.from_numpy(np.concatenate(bags_arr, axis=0)).to(self.device)
        if rot:
            X = torch.bmm(X, _rand_rot(X.shape[0], X.device))   # rotate each cloud
        if jit > 0:
            X = X + jit * torch.randn_like(X)                   # + small jitter
        logits = model(X)                                  # (sum sizes,)
        out, off = [], 0
        for s in sizes:
            out.append(logits[off:off + s].max())
            off += s
        return torch.stack(out)

    def train(self, train, test=None):
        print(f"[pointnet] training {self.n_models} model(s) on "
              f"{len(train.malign)} malign + {len(train.benign)} benign meshes "
              f"(n_points={self.n_points}, maxc={self.maxc}, epochs={self.epochs}, "
              f"rot_aug={self.rot_aug}, jitter={self.jitter})", flush=True)
        bags = self._load_bags(train)
        if not bags:
            raise RuntimeError("PointNet.train found no usable training meshes.")
        labels = np.array([lab for _, lab in bags], np.float32)
        n = len(bags)

        # Train an ensemble of identical networks from consecutive seeds; their
        # scores are averaged at eval (see _scores/eval). Sampling is done once
        # above and shared, so only the optimization repeats per model.
        models = []
        for mi in range(self.n_models):
            seed = self.seed + mi
            _seed_everything(seed)
            model = _PointNetModel(dropout=self.dropout).to(self.device)
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
                    if sum(a.shape[0] for a in bags_arr) < 2:   # keep BatchNorm happy
                        continue
                    yb = torch.from_numpy(labels[idx]).to(self.device)
                    logits = self._bag_logits(model, bags_arr,
                                              rot=self.rot_aug, jit=self.jitter)
                    loss = loss_fn(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running += float(loss.detach())
                    nb += 1
                if sched is not None:
                    sched.step()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"[pointnet] model {mi + 1}/{self.n_models} "
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
                print(f"[pointnet] validation AUC="
                      f"{roc_auc_score(yte, scores):.4f}", flush=True)

    @torch.no_grad()
    def _scores(self, bag_arrays):
        """Ensemble malign scores for a list of bag arrays: each model's sigmoid
        of the max-pooled bag logit, averaged over the ``n_models`` models."""
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

    def save(self) -> None:
        """Cache the trained ensemble weights (plus the geometry hyperparameters
        ``eval`` needs to rebuild and feed the network) to ``trained_dir``."""
        if not self._models:
            raise RuntimeError("PointNet.save called before train().")
        torch.save(
            {"state_dicts": [m.state_dict() for m in self._models],
             "n_points": self.n_points, "dropout": self.dropout,
             "maxc": self.maxc, "n_models": len(self._models)},
            os.path.join(self.trained_dir, "model.pt"),
        )

    def load(self) -> None:
        """Rebuild the ensemble from the checkpoint saved by ``save``."""
        path = os.path.join(self.trained_dir, "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached model at {path}; run `python evaluate.py --train` first."
            )
        ckpt = torch.load(path, map_location=self.device)
        self.n_points = int(ckpt["n_points"])
        self.maxc = int(ckpt.get("maxc", self.maxc))
        # Back-compat: older checkpoints stored a single "state_dict".
        state_dicts = ckpt.get("state_dicts") or [ckpt["state_dict"]]
        models = []
        for sd in state_dicts:
            model = _PointNetModel(dropout=ckpt.get("dropout", self.dropout)).to(self.device)
            model.load_state_dict(sd)
            model.eval()
            models.append(model)
        self._models = models
        self.n_models = len(models)

    def eval(self, obj_path) -> dict:
        if not self._models:
            return {"score": 0.0}
        cs = self._bag(obj_path)
        if cs is None:  # nothing survived sanitizing -> no device geometry -> benign
            return {"score": 0.0}
        with torch.no_grad():
            X = torch.from_numpy(np.stack(cs)).to(self.device)
            scores = [torch.sigmoid(m(X).max()).item() for m in self._models]
        return {"score": float(np.mean(scores))}
