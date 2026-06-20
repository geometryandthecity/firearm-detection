"""Tri-plane projection CNN detector (C3PO, Ma et al., ASPDAC 2020) for meshes.

Implements the "G-code interpretation" scenario of C3PO -- project the object
onto the xy/xz/yz planes, stack the three silhouettes into one RGB image, and
classify it with a small CNN that uses Log-Sum-Exp (LSE) pooling -- adapted to
score triangle meshes in this benchmark. The three-channel image is produced by
``render.py``; this module is the classifier, the training loop, and the mesh
robustness wrapper.

Faithful to the paper's network: three convolution layers with 128 (5x5),
256 (3x3) and 512 (1x1) filters, LSE pooling (the paper's best pooling, r=10,
93.6%) after the first two, a global pool, then a 1x1-conv head. The original
10-class softmax head is replaced by a single malign logit trained with BCE, to
match every detector's [0, 1] score contract.

Adaptations for this benchmark (see ``render.py`` for the rendering side):

* **Per-component multiple-instance learning.** A mesh is a *bag* of connected
  components; the CNN scores each component's three-plane image and the mesh
  logit is the **max** over components -- the mesh is malign iff *any* component
  looks like the device. This is the same max-pool MIL the ``pointnet`` detector
  uses, and it is what makes the ``disconnected`` adversary (device hidden among
  junk) and ``degenerate`` adversary (sanitized away before rendering) tractable.
* **Sign-flip augmentation / TTA.** PCA canonicalization leaves a residual
  axis-sign ambiguity (4 proper-rotation sign combos). Training randomizes over
  them per sample and evaluation averages over all 4, so a near-symmetric shape
  whose skewness sign convention is unstable is still scored consistently.

Runs on CPU, Apple-Silicon GPU (MPS) or CUDA (``device="auto"`` prefers
cuda > mps > cpu). All randomness is seeded from ``seed`` for reproducibility.
"""
import json
import hashlib
import os

# Let any op MPS hasn't implemented fall back to CPU, so the same code runs on
# Apple-Silicon GPU, plain CPU, or CUDA without special-casing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..base import Detector
from .render import render_triplane, RENDER_STYLE

# The four axis-sign combinations that keep a proper rotation (product = +1).
# Each is a residual ambiguity of the PCA frame; the network is trained to be
# invariant to them (augmentation) and they are averaged over at eval (TTA).
_SIGN_COMBOS = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))


def _resolve_device(device):
    """Map ``"auto"`` to the best available backend (cuda > mps > cpu)."""
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


def _signflip(x, sx, sy, sz):
    """Apply one axis-sign combo to a tri-plane batch ``x`` (N, 3, H, W).

    Flipping world axis x mirrors the xy (R) and xz (G) views along their column
    axis; flipping y mirrors xy (R) rows and yz (B) columns; flipping z mirrors
    xz (G) and yz (B) rows. (Convention: per plane, the first dropped-from axis
    is the image column, the second is the row -- matching ``render._component_image``.)
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    if sx < 0:
        r = torch.flip(r, [-1]); g = torch.flip(g, [-1])
    if sy < 0:
        r = torch.flip(r, [-2]); b = torch.flip(b, [-1])
    if sz < 0:
        g = torch.flip(g, [-2]); b = torch.flip(b, [-2])
    return torch.cat([r, g, b], dim=1)


def _lse_pool(x, kernel, stride, padding, r):
    """Log-Sum-Exp spatial pooling (paper Eq. 1): ``(1/r) log mean_window exp(r x)``.

    Interpolates between average (r->0) and max (r->inf) pooling. Computed in a
    numerically stable, shift-covariant form by subtracting the per-channel
    global max before exponentiating; ``count_include_pad=False`` so padded
    border cells do not bias the window mean.
    """
    m = x.amax(dim=(2, 3), keepdim=True)
    z = torch.exp(r * (x - m))
    a = F.avg_pool2d(z, kernel, stride, padding, count_include_pad=False)
    return m + (1.0 / r) * torch.log(a.clamp_min(1e-12))


def _global_lse(x, r):
    """Global Log-Sum-Exp pool over the spatial dims -> ``(B, C, 1, 1)``."""
    m = x.amax(dim=(2, 3), keepdim=True)
    a = torch.exp(r * (x - m)).mean(dim=(2, 3), keepdim=True)
    return m + (1.0 / r) * torch.log(a.clamp_min(1e-12))


class _C3PONet(torch.nn.Module):
    """The paper's three-conv LSE-pooling CNN, with a single malign logit.

    ``conv(128, 5x5) -> LSE-pool -> conv(256, 3x3) -> LSE-pool -> conv(512, 1x1)
    -> global LSE pool -> 1x1-conv head``. GroupNorm+ReLU follow each conv to make
    the from-scratch network train stably (the paper trains in TensorFlow with L2
    regularization; here weight decay plays that role -- see ``TriPlaneCNN``).
    GroupNorm rather than BatchNorm because the multiple-instance batches mix a
    variable number of component images per mesh and are forwarded in chunks to
    bound memory; a normalizer that is independent of the (chunk) batch size keeps
    training and eval identical regardless of how the components are grouped.
    """

    def __init__(self, r=10.0, dropout=0.3, groups=32):
        super().__init__()
        self.r = r
        self.conv1 = torch.nn.Conv2d(3, 128, 5, padding=2, bias=False)
        self.gn1 = torch.nn.GroupNorm(groups, 128)
        self.conv2 = torch.nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(groups, 256)
        self.conv3 = torch.nn.Conv2d(256, 512, 1, bias=False)
        self.gn3 = torch.nn.GroupNorm(groups, 512)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Conv2d(512, 1, 1)     # 1x1 conv == linear on the pooled feature

    def forward(self, x):                          # x: (B, 3, H, W)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = _lse_pool(x, 3, 2, 1, self.r)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = _lse_pool(x, 3, 2, 1, self.r)
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        x = _global_lse(x, self.r)                 # (B, 512, 1, 1)
        x = self.head(self.drop(x))                # (B, 1, 1, 1)
        return x.flatten(1).squeeze(1)             # (B,): one logit per image


class TriPlaneCNN(Detector):
    """Detect malign meshes by classifying their tri-plane RGB projection (C3PO).

    See the module docstring. Renders are cached on disk (keyed by file identity +
    render params + ``RENDER_STYLE``) so the slow first pass is paid once.
    """

    def __init__(self, img_size=128, supersample=2, maxc=16, n_pca=4096,
                 r=10.0, epochs=40, batch_size=16, lr=1e-3, weight_decay=1e-4,
                 dropout=0.3, device="auto", seed=0, n_models=3, cosine=True,
                 tta=True, max_per_class=None, max_fwd=64):
        super().__init__()
        self.img_size = img_size
        self.supersample = supersample
        self.maxc = maxc
        self.n_pca = n_pca
        self.r = r
        self.epochs = epochs
        self.batch_size = batch_size       # number of meshes (bags) per optimizer step
        self.lr = lr
        self.weight_decay = weight_decay   # the paper's L2 regularization
        self.dropout = dropout
        self.device = _resolve_device(device)
        self.seed = seed
        self.n_models = n_models           # seed-ensemble size (sigmoid scores averaged)
        self.cosine = cosine
        self.tta = tta                     # average over the 4 sign combos at eval
        self.max_per_class = max_per_class
        self.max_fwd = max_fwd             # max component images per CNN forward (memory cap)
        self._models = []

    # ------------------------------------------------------------------ rendering
    def _cache_dir(self, obj_path):
        """Per-shape render-cache folder, keyed by file identity + render params.

        Folds in the file's size and mtime so regenerating the data (which
        rewrites the .obj) invalidates the cache, plus the render params and
        ``RENDER_STYLE`` so changing the projection appearance does too.
        """
        ap = os.path.abspath(obj_path)
        try:
            st = os.stat(ap)
            key = (f"{ap}|{st.st_size}|{int(st.st_mtime)}|{self.img_size}"
                   f"|{self.supersample}|{self.maxc}|{self.n_pca}|{RENDER_STYLE}")
        except OSError:
            key = (f"{ap}|{self.img_size}|{self.supersample}|{self.maxc}"
                   f"|{self.n_pca}|{RENDER_STYLE}")
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self.trained_dir, "render_cache", h)

    def _bag_paths(self, obj_path):
        """Component PNG paths for ``obj_path`` (rendering on a cache miss), or ``None``.

        ``None`` means no usable component survived -- no device geometry to score.
        """
        cache = self._cache_dir(obj_path)
        meta_path = os.path.join(cache, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    n = int(json.load(f)["n"])
                paths = [os.path.join(cache, f"comp{i}.png") for i in range(n)]
                if all(os.path.exists(p) for p in paths):
                    return paths or None
            paths = render_triplane(obj_path, cache, size=self.img_size,
                                    supersample=self.supersample, maxc=self.maxc,
                                    n_pca=self.n_pca)
            return paths or None
        except Exception:
            return None

    def _load_bag(self, paths):
        """Load a component-PNG list into one ``(n_comp, 3, H, W)`` uint8 tensor."""
        from PIL import Image
        imgs = []
        for p in paths:
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # writable copy
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        return torch.stack(imgs)                    # (n_comp, 3, H, W)

    def _load_bags(self, split, log_every=400):
        """Render (or reuse) and load every usable mesh in ``split`` into RAM.

        Returns a list of ``((n_comp, 3, H, W) uint8 tensor, label)``. The first
        pass over a split renders (the slow part) and is logged like the other
        detectors; later passes hit the cache.
        """
        pairs = [(p, 1.0) for p in split.malign] + [(p, 0.0) for p in split.benign]
        if self.max_per_class is not None:
            pairs = ([(p, 1.0) for p in split.malign[:self.max_per_class]] +
                     [(p, 0.0) for p in split.benign[:self.max_per_class]])
        bags, total = [], len(pairs)
        for i, (p, label) in enumerate(pairs, start=1):
            paths = self._bag_paths(p)
            if paths:
                bags.append((self._load_bag(paths), label))
            if i % log_every == 0 or i == total:
                print(f"[triplane_cnn] rendered {i}/{total}, {len(bags)} usable",
                      flush=True)
        return bags

    # ------------------------------------------------------------------ batching
    def _normalize(self, x_uint8):
        """uint8 ``(N, 3, H, W)`` on CPU -> float in [0, 1] on the device."""
        return x_uint8.to(self.device).float().div_(255.0)

    def _augment_signflip(self, x, rng):
        """Randomly assign each image one of the 4 sign combos (train augmentation)."""
        idx = rng.integers(0, len(_SIGN_COMBOS), size=x.shape[0])
        out = x.clone()
        for c, (sx, sy, sz) in enumerate(_SIGN_COMBOS):
            sel = np.nonzero(idx == c)[0]
            if len(sel):
                sel_t = torch.from_numpy(sel).to(x.device)
                out[sel_t] = _signflip(x[sel_t], sx, sy, sz)
        return out

    def _forward_imgs(self, model, x):
        """Run the CNN over ``x`` (M, 3, H, W), chunked at ``max_fwd`` images.

        Conv1 keeps 128 feature maps at the full input resolution (~33 MB of
        activation per image), so a multiple-instance batch that concatenates the
        components of many meshes -- especially ``disconnected`` bags of up to
        ``maxc`` parts -- can exceed memory. Forwarding in fixed-size chunks and
        concatenating the logits bounds peak memory; gradients flow through the
        concat, and GroupNorm makes each chunk's result independent of chunk size.
        """
        M = x.shape[0]
        if M <= self.max_fwd:
            return model(x)
        return torch.cat([model(x[i:i + self.max_fwd])
                          for i in range(0, M, self.max_fwd)])

    def _bag_logits(self, model, bags, rng=None):
        """Per-mesh logits = max over each bag's component logits.

        All components in the batch are concatenated, forwarded (chunked) in one
        pass, then scattered back and max-pooled per bag. When ``rng`` is given,
        each component image is sign-flip augmented first.
        """
        sizes = [b.shape[0] for b in bags]
        x = self._normalize(torch.cat(bags, dim=0))
        if rng is not None:
            x = self._augment_signflip(x, rng)
        logits = self._forward_imgs(model, x)       # (sum sizes,)
        out, off = [], 0
        for s in sizes:
            out.append(logits[off:off + s].max())
            off += s
        return torch.stack(out)

    # ----------------------------------------------------------------- lifecycle
    def train(self, train, test=None):
        print(f"[triplane_cnn] training {self.n_models} model(s) on "
              f"{len(train.malign)} malign + {len(train.benign)} benign meshes "
              f"(img_size={self.img_size}, maxc={self.maxc}, r={self.r}, "
              f"epochs={self.epochs})", flush=True)
        bags = self._load_bags(train)
        if not bags:
            raise RuntimeError("TriPlaneCNN.train found no renderable training meshes.")
        labels = np.array([lab for _, lab in bags], np.float32)
        n = len(bags)

        models = []
        for mi in range(self.n_models):
            seed = self.seed + mi
            _seed_everything(seed)
            model = _C3PONet(r=self.r, dropout=self.dropout).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
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
                    batch = [bags[k][0] for k in idx]
                    yb = torch.from_numpy(labels[idx]).to(self.device)
                    logits = self._bag_logits(model, batch, rng=rng)
                    loss = loss_fn(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running += float(loss.detach())
                    nb += 1
                if sched is not None:
                    sched.step()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"[triplane_cnn] model {mi + 1}/{self.n_models} "
                          f"epoch {epoch + 1}/{self.epochs} "
                          f"loss={running / max(nb, 1):.4f}", flush=True)
            model.eval()
            models.append(model)
        self._models = models

        # Informational only: validation AUC on the test split (not used to tune).
        if test is not None:
            tb = self._load_bags(test)
            if tb and len({lab for _, lab in tb}) == 2:
                scores = np.array([self._score_bag(b) for b, _ in tb])
                yte = np.array([lab for _, lab in tb])
                print(f"[triplane_cnn] validation AUC="
                      f"{roc_auc_score(yte, scores):.4f}", flush=True)

    @torch.no_grad()
    def _score_bag(self, bag):
        """Ensemble malign score for one bag ``(n_comp, 3, H, W)`` uint8 tensor.

        Each model scores every component; with TTA the component probability is
        the mean sigmoid over the 4 sign combos. The mesh probability is the max
        over components, and the final score is the mean over the ensemble.
        """
        if not self._models:
            return 0.0
        x = self._normalize(bag)                    # (n_comp, 3, H, W) float on device
        variants = ([_signflip(x, *c) for c in _SIGN_COMBOS] if self.tta else [x])
        per_model = []
        for model in self._models:
            model.eval()
            probs = torch.stack([torch.sigmoid(self._forward_imgs(model, v))
                                 for v in variants])    # (T, n_comp)
            comp_prob = probs.mean(dim=0)           # (n_comp,) TTA-averaged
            per_model.append(float(comp_prob.max()))
        return float(np.mean(per_model))

    def save(self):
        """Cache the trained ensemble plus the hyperparameters ``eval`` needs."""
        if not self._models:
            raise RuntimeError("TriPlaneCNN.save called before train().")
        torch.save(
            {"state_dicts": [m.state_dict() for m in self._models],
             "img_size": self.img_size, "supersample": self.supersample,
             "maxc": self.maxc, "n_pca": self.n_pca, "r": self.r,
             "dropout": self.dropout, "tta": self.tta},
            os.path.join(self.trained_dir, "model.pt"),
        )

    def load(self):
        """Rebuild the ensemble from the checkpoint saved by ``save``."""
        path = os.path.join(self.trained_dir, "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached model at {path}; run `python evaluate.py --train` first."
            )
        ckpt = torch.load(path, map_location=self.device)
        self.img_size = int(ckpt["img_size"])
        self.supersample = int(ckpt["supersample"])
        self.maxc = int(ckpt["maxc"])
        self.n_pca = int(ckpt["n_pca"])
        self.r = float(ckpt["r"])
        self.tta = bool(ckpt.get("tta", self.tta))
        models = []
        for sd in ckpt["state_dicts"]:
            model = _C3PONet(r=self.r, dropout=ckpt.get("dropout", self.dropout))
            model.load_state_dict(sd)
            model.to(self.device).eval()
            models.append(model)
        self._models = models
        self.n_models = len(models)

    def eval(self, obj_path):
        if not self._models:
            return {"score": 0.0}
        paths = self._bag_paths(obj_path)
        if not paths:                               # nothing renderable -> benign
            return {"score": 0.0}
        return {"score": self._score_bag(self._load_bag(paths))}
