"""Tri-plane orthographic projection renderer for the C3PO detector.

This turns one ``.obj`` mesh into the input representation of the C3PO method
(Ma et al., "Database and Benchmark for Early-stage Malicious Activity Detection
in 3D Printing", ASPDAC 2020): the shape's silhouette is projected onto the
``xy``, ``xz`` and ``yz`` planes and the three single-channel silhouettes are
stacked into one RGB image (R=xy, G=xz, B=yz) -- the paper's Fig. 5/6. A 2D CNN
then classifies that three-channel image (see ``detector.py``).

Mesh adaptations of the original G-code method, all motivated by *this*
benchmark (arbitrarily-posed meshes + the adversaries) and consistent with how
the other detectors are hardened (``utils.sanitize`` / ``split_components``):

* **Pure-CPU rasterization.** The silhouettes are rasterized on the CPU with
  Pillow (filled-triangle scan conversion), so a render is bit-identical on
  macOS and the Linux GPU server with no EGL/Blender dependency -- unlike the
  ``image_classifier``'s shaded renderer, which needs a GL/Blender backend.
  A projected silhouette is just the union of the triangles' 2D footprints, so
  it needs no z-buffer, lighting, or camera rig.
* **PCA pose-canonicalization.** The paper projects along the known print-bed
  axes; here meshes arrive in arbitrary poses (and the ``rigid``/``tetwild``
  adversaries re-pose them on purpose). Each component is rotated into its
  principal-axis frame -- computed from an *area-weighted point sample*, so the
  frame is invariant to tessellation (defeating ``faceswap``/``tetwild``/
  ``jitter``) -- before projecting. Axis signs are fixed by a deterministic
  skewness convention; the residual sign ambiguity is mopped up by the
  detector's train-time sign-flip augmentation and eval-time TTA.
* **Per-component, sanitized.** ``sanitize`` strips loader-exploit garbage
  (defeats ``degenerate``) and the mesh is split into connected components; each
  component is canonicalized + projected on its own (the detector max-pools over
  components, so a device hidden among junk -- ``disconnected`` -- still gets its
  own clean three-plane image).
"""
import json
import os

import numpy as np
import gpytoolbox as gpy
from PIL import Image, ImageDraw

from ..utils import sanitize, split_components

# Bump this tag whenever the rendered *appearance* changes (projection geometry,
# canonicalization, supersampling) so the detector's on-disk render cache
# invalidates instead of silently reusing visually-stale PNGs.
RENDER_STYLE = "triplane-pca-v1"

# Fixed seed for the area-weighted point sample used to estimate the PCA frame.
# Kept constant (not tied to the model seed) so renders are reproducible and
# shareable across model seeds / ensemble members and the cache stays valid.
_PCA_SEED = 12345


def _pca_frame(V, F, n_samples, seed=_PCA_SEED):
    """Return ``(center, R)`` mapping ``V`` into its canonical principal-axis frame.

    ``center`` is the area-weighted centroid and ``R`` is a proper rotation
    (``det = +1``) whose columns are the principal axes ordered by descending
    variance, so ``(V - center) @ R`` places the largest-spread axis on x, the
    next on y, the smallest on z. The frame is estimated from an area-weighted
    surface point sample (not the raw vertices) so it does not move when the mesh
    is re-tessellated. Axis signs are fixed deterministically by making each
    axis's third moment (skewness) non-negative; if that leaves a reflection,
    the least-skewed (most ambiguous) axis is flipped to restore ``det = +1``.
    Falls back to vertex statistics if surface sampling is unavailable.
    """
    V = np.asarray(V, float)
    try:
        pts = np.asarray(
            gpy.random_points_on_mesh(V, np.asarray(F, np.int64), n_samples,
                                      rng=np.random.default_rng(seed)), float)
        if pts.shape != (n_samples, 3):
            pts = V
    except Exception:
        pts = V

    # macOS's Accelerate BLAS spuriously raises FP-flag RuntimeWarnings ("divide
    # by zero"/"invalid value") for some small matmuls even though the inputs and
    # results are finite; silence them locally (the result is validated below).
    with np.errstate(all="ignore"):
        center = pts.mean(axis=0)
        P = pts - center
        # Principal axes = eigenvectors of the covariance, ordered by descending var.
        cov = (P.T @ P) / max(len(P), 1)
        w, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
        R = vecs[:, np.argsort(w)[::-1]]       # columns: axes, descending variance

        proj = P @ R
        skew = (proj ** 3).sum(axis=0)         # per-axis third moment
        for i in range(3):
            if skew[i] < 0:
                R[:, i] *= -1.0
        if np.linalg.det(R) < 0:               # restore a proper rotation
            R[:, int(np.argmin(np.abs(skew)))] *= -1.0

    if not np.isfinite(R).all():               # degenerate cov -> no rotation
        center = np.asarray(V, float).mean(axis=0)
        R = np.eye(3)
    return center, R


def _rasterize(verts_px, F, S, size):
    """Rasterize one plane's filled-triangle silhouette to a ``size``x``size`` uint8.

    ``verts_px`` is the ``(V, 2)`` integer pixel coordinates of the projected
    vertices at the supersampled resolution ``S``; every triangle is drawn filled
    white on black and the result is box-filtered down to ``size`` for cheap
    anti-aliasing (so edges become grey, matching the paper's "grayscale
    projection"). The silhouette is the union of the triangle footprints, which
    is exactly what filling every face produces.
    """
    img = Image.new("L", (S, S), 0)
    draw = ImageDraw.Draw(img)
    # Bulk-convert to a list of flat [x0,y0,x1,y1,x2,y2] polygons once; the only
    # per-triangle work left is the C-level fill call.
    polys = verts_px[F].reshape(len(F), 6).tolist()
    for p in polys:
        draw.polygon(p, fill=255)
    if S != size:
        img = img.resize((size, size), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def _component_image(V, F, size, supersample, n_pca, border=0.05):
    """Canonicalize one component and render its three-plane RGB image, or ``None``.

    Returns an ``(size, size, 3)`` uint8 array with channel 0 = xy, 1 = xz,
    2 = yz silhouette, or ``None`` if the component is too small / degenerate to
    project. A single isotropic scale is used for all three planes so the
    projections stay mutually consistent (a vertex's x is the same column in the
    xy and xz views), and the shape is centered with a small border.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(V) < 3 or len(F) == 0:
        return None

    center, R = _pca_frame(V, F, n_pca)
    with np.errstate(all="ignore"):             # silence Accelerate's spurious matmul FP flags
        Vr = (V - center) @ R                   # canonical principal-axis frame

    half = float(np.abs(Vr).max())
    if not np.isfinite(half) or half <= 0:
        return None

    S = size * supersample
    scale = (S * (1.0 - 2.0 * border) / 2.0) / half
    px = np.rint(Vr * scale + S / 2.0).astype(np.int32)   # (V, 3) pixel coord per axis
    px = np.clip(px, 0, S - 1)

    # Each plane drops one axis; (col, row) = (first axis, second axis).
    chans = [
        _rasterize(px[:, [0, 1]], F, S, size),  # xy -> R
        _rasterize(px[:, [0, 2]], F, S, size),  # xz -> G
        _rasterize(px[:, [1, 2]], F, S, size),  # yz -> B
    ]
    return np.stack(chans, axis=-1)


def render_triplane(obj_path, out_dir, size=128, supersample=2, maxc=16,
                    n_pca=4096):
    """Render every component of ``obj_path`` to a tri-plane RGB PNG in ``out_dir``.

    Sanitizes the mesh, splits it into (at most ``maxc``) connected components,
    and writes one ``comp{i}.png`` per component plus a ``meta.json`` recording
    the count (so the cache can be validated without re-rendering). Returns the
    list of PNG paths in order; an empty list means nothing usable survived
    (the caller scores such a mesh benign, like the other detectors). The PNGs
    are 3-channel ``size``x``size`` images: R=xy, G=xz, B=yz silhouette.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        V, F = gpy.read_mesh(obj_path)
    except Exception:
        V, F = None, None

    paths = []
    if V is not None and F is not None:
        V = np.asarray(V, float)
        F = np.asarray(F, np.int64)
        if F.ndim == 2 and F.shape[1] == 3 and len(F) > 0:
            V, F = sanitize(V, F)
            for Vc, Fc in split_components(V, F, maxc=maxc, do_sanitize=False):
                img = _component_image(Vc, Fc, size, supersample, n_pca)
                if img is None:
                    continue
                p = os.path.join(out_dir, f"comp{len(paths)}.png")
                Image.fromarray(img, "RGB").save(p)
                paths.append(p)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"n": len(paths), "style": RENDER_STYLE}, f)
    return paths
