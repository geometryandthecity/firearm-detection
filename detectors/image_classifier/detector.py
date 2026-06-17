"""Multi-view image classifier: render each shape, then classify with an MVCNN.
"""
import hashlib
import os

# Let any op MPS hasn't implemented fall back to CPU, so the exact same code
# runs on Apple-Silicon GPU, plain CPU, or CUDA without special-casing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..base import Detector
from .render import render_views, RENDER_STYLE

# ImageNet normalization the pretrained torchvision backbones expect.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


class _MVCNN(torch.nn.Module):
    """View-pooled CNN: per-view backbone -> max over views -> one malign logit.

    The backbone runs on every view independently (the views are folded into the
    batch dimension), and the resulting per-view feature vectors are collapsed by
    a max over the view axis -- the symmetric pool that makes the descriptor
    independent of which viewpoint saw what, mirroring PointNet's pooling over
    points. A dropout + linear head turns the pooled feature into the score logit.
    """

    def __init__(self, backbone, feat_dim, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feat_dim, 1),
        )

    def forward(self, x):                       # x: (B, V, 3, H, W)
        b, v = x.shape[0], x.shape[1]
        x = x.flatten(0, 1)                     # (B*V, 3, H, W)
        f = self.backbone(x)                    # (B*V, feat_dim)
        f = f.view(b, v, -1).amax(dim=1)        # (B, feat_dim): symmetric view-pool
        return self.head(f).squeeze(1)          # (B,): one logit per shape


class ImageClassifier(Detector):
    """Detect malign shapes by rendering them and classifying the images (MVCNN).
    """

    def __init__(self, n_views=8, img_size=128, backbone="resnet18",
                 pretrained=True, epochs=20, batch_size=16, lr=1e-4, dropout=0.3,
                 device="auto", seed=0, max_per_class=None):
        super().__init__()
        self.n_views = n_views
        self.img_size = img_size
        self.backbone = backbone
        self.pretrained = pretrained
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.device = _resolve_device(device)
        self.seed = seed
        self.max_per_class = max_per_class
        self._model = None
        self._feat_dim = None
        self._mean = None
        self._std = None

    # ------------------------------------------------------------------ helpers
    def _prep_hub(self):
        """Cache torchvision pretrained weights under ``installed_dir``.

        Pointing torch.hub at our per-method install folder keeps the backbone
        download out of the user's global cache and lets ``install`` fetch it once
        -- changing the dataset and retraining then never re-downloads weights.
        """
        torch.hub.set_dir(self.installed_dir)

    def _build_backbone(self, pretrained=None):
        """Build the 2D feature extractor, returning ``(module, feat_dim)``.

        ``module`` maps ``(B, 3, H, W) -> (B, feat_dim)`` (the classification head
        is stripped to an identity). ``pretrained`` defaults to ``self.pretrained``;
        ``load`` passes ``False`` because the trained weights are restored from the
        checkpoint and there is no reason to download ImageNet weights at eval.
        Any ResNet in the standard family (18/34/50) is accepted; ``feat_dim`` is
        read off the original ``fc`` (512 for resnet18/34, 2048 for resnet50).
        """
        import torchvision
        if pretrained is None:
            pretrained = self.pretrained
        builders = {
            "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights),
            "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights),
            "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights),
        }
        if self.backbone not in builders:
            raise ValueError(f"unknown backbone {self.backbone!r} "
                             f"(supported: {', '.join(sorted(builders))})")
        ctor, weights_enum = builders[self.backbone]
        net = ctor(weights=weights_enum.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = torch.nn.Identity()
        return net, feat_dim

    def _cache_dir(self, obj_path):
        """Per-shape render-cache folder, keyed by file identity + render params.

        The key folds in the file's size and mtime so that regenerating the data
        (which rewrites the .obj) invalidates the cache, while unchanged shapes
        keep reusing their renders across runs. ``RENDER_STYLE`` is included so
        that changing the rendered *appearance* (e.g. the render engine or
        lighting) also invalidates the cache instead of reusing stale PNGs.
        """
        ap = os.path.abspath(obj_path)
        try:
            st = os.stat(ap)
            key = (f"{ap}|{st.st_size}|{int(st.st_mtime)}"
                   f"|{self.n_views}|{self.img_size}|{RENDER_STYLE}")
        except OSError:
            key = f"{ap}|{self.n_views}|{self.img_size}|{RENDER_STYLE}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self.trained_dir, "render_cache", h)

    def _views_for(self, obj_path):
        """Return the ``n_views`` cached PNG paths for ``obj_path``, rendering if
        needed; ``None`` if the shape can't be rendered (degenerate input)."""
        try:
            cache = self._cache_dir(obj_path)
            outs = [os.path.join(cache, f"view{i}.png") for i in range(self.n_views)]
            if all(os.path.exists(o) for o in outs):
                return outs
            return render_views(obj_path, cache, n_views=self.n_views,
                                img_size=self.img_size)
        except Exception:
            return None

    def _prepare_items(self, split, split_name):
        """Render (or reuse cached) views for every usable shape in ``split``.

        Returns a list of ``(view_paths, label)`` for shapes that rendered. The
        first pass over a split is the slow part (Blender), so progress is logged
        like the other methods; later passes hit the cache and fly.
        """
        pairs = [(p, 1.0) for p in split.malign] + [(p, 0.0) for p in split.benign]
        if self.max_per_class is not None:
            pairs = ([(p, 1.0) for p in split.malign[:self.max_per_class]] +
                     [(p, 0.0) for p in split.benign[:self.max_per_class]])
        items, total = [], len(pairs)
        for i, (p, label) in enumerate(pairs, start=1):
            vp = self._views_for(p)
            if vp is not None:
                items.append((vp, label))
            if i % 50 == 0 or i == total:
                print(f"[image_classifier] {split_name}: rendered {i}/{total}, "
                      f"{len(items)} usable", flush=True)
        return items

    def _load_view(self, path):
        """Load one PNG as a ``(3, H, W)`` uint8 tensor at ``img_size``."""
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.uint8)              # (H, W, 3); copy -> writable
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_views_tensor(self, items, split_name):
        """Load all cached views for ``items`` into one uint8 ``(N, V, 3, H, W)``
        tensor (+ float labels), logging progress.

        Keeping the whole split resident as uint8 (a few GB at most) means the
        training loop never re-reads or re-decodes a PNG between epochs -- the
        same in-RAM strategy ``PointNet`` uses for its point clouds.
        """
        n = len(items)
        X = torch.empty((n, self.n_views, 3, self.img_size, self.img_size),
                        dtype=torch.uint8)
        y = torch.empty((n,), dtype=torch.float32)
        for i, (view_paths, label) in enumerate(items):
            for j, p in enumerate(view_paths):
                X[i, j] = self._load_view(p)
            y[i] = label
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"[image_classifier] {split_name}: loaded {i + 1}/{n} into RAM",
                      flush=True)
        return X, y

    def _normalize(self, x_uint8):
        """uint8 ``(B, V, 3, H, W)`` on CPU -> normalized float batch on device."""
        if self._mean is None:
            self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 1, 3, 1, 1)
            self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 1, 3, 1, 1)
        x = x_uint8.to(self.device).float().div_(255.0)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def _scores(self, model, X):
        """Batched sigmoid malign scores for a uint8 view tensor ``X`` (on CPU)."""
        model.eval()
        out = []
        for i in range(0, X.shape[0], self.batch_size):
            xb = self._normalize(X[i:i + self.batch_size])
            out.append(torch.sigmoid(model(xb)).cpu())
        return torch.cat(out).numpy() if out else np.empty((0,), np.float32)

    # ----------------------------------------------------------------- lifecycle
    def install(self) -> None:
        """Pre-fetch the pretrained backbone weights into ``installed_dir``."""
        if not self.pretrained:
            print("[image_classifier] install: pretrained=False, nothing to fetch.",
                  flush=True)
            return
        self._prep_hub()
        print(f"[image_classifier] install: fetching backbone weights "
              f"({self.backbone})...", flush=True)
        self._build_backbone(pretrained=True)
        print("[image_classifier] install: done.", flush=True)

    def train(self, train, test=None):
        _seed_everything(self.seed)
        self._prep_hub()
        print(f"[image_classifier] training on {len(train.malign)} malign + "
              f"{len(train.benign)} benign shapes "
              f"(n_views={self.n_views}, img_size={self.img_size}, "
              f"backbone={self.backbone})", flush=True)

        items = self._prepare_items(train, "train")
        if not items:
            raise RuntimeError("ImageClassifier.train found no renderable training meshes.")
        X, y = self._load_views_tensor(items, "train")
        n = X.shape[0]

        backbone, feat_dim = self._build_backbone()
        self._feat_dim = feat_dim
        model = _MVCNN(backbone, feat_dim, dropout=self.dropout).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n)
            running, nb = 0.0, 0
            # Drop the final partial batch so BatchNorm always sees >1 sample.
            for i in range(0, n - self.batch_size + 1, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = self._normalize(X[idx])
                yb = y[idx].to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                running += float(loss.detach())
                nb += 1
            print(f"[image_classifier] epoch {epoch + 1}/{self.epochs} "
                  f"loss={running / max(nb, 1):.4f}", flush=True)

        # Informational only: validation AUC on the test split (not used to tune).
        if test is not None:
            test_items = self._prepare_items(test, "val")
            if test_items:
                Xte, yte = self._load_views_tensor(test_items, "val")
                if len(np.unique(yte.numpy())) == 2:
                    auc = roc_auc_score(yte.numpy(), self._scores(model, Xte))
                    print(f"[image_classifier] validation AUC={auc:.4f}", flush=True)

        model.eval()
        self._model = model

    def save(self) -> None:
        """Cache the trained weights plus the render/arch hyperparameters
        ``eval`` needs to rebuild the network, into ``trained_dir``."""
        if self._model is None:
            raise RuntimeError("ImageClassifier.save called before train().")
        torch.save(
            {"state_dict": self._model.state_dict(),
             "backbone": self.backbone, "feat_dim": self._feat_dim,
             "n_views": self.n_views, "img_size": self.img_size,
             "dropout": self.dropout},
            os.path.join(self.trained_dir, "model.pt"),
        )

    def load(self) -> None:
        """Rebuild the network from the checkpoint saved by ``save``."""
        path = os.path.join(self.trained_dir, "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached model at {path}; run `python evaluate.py --train` first."
            )
        ckpt = torch.load(path, map_location=self.device)
        self.backbone = ckpt["backbone"]
        self.n_views = int(ckpt["n_views"])
        self.img_size = int(ckpt["img_size"])
        backbone, feat_dim = self._build_backbone(pretrained=False)
        self._feat_dim = feat_dim
        model = _MVCNN(backbone, feat_dim, dropout=ckpt.get("dropout", self.dropout))
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        self._model = model

    def eval(self, obj_path) -> dict:
        if self._model is None:
            return {"score": 0.0}
        vp = self._views_for(obj_path)
        if vp is None:  # unrenderable mesh -> treat as benign
            return {"score": 0.0}
        views = torch.stack([self._load_view(p) for p in vp])     # (V, 3, H, W) uint8
        score = self._scores(self._model, views.unsqueeze(0))[0]  # (1, V, 3, H, W)
        return {"score": float(score)}
