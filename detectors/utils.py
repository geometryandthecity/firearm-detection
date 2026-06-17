"""helpers for detectors.
"""
import numpy as np
import gpytoolbox as gpy


def mesh_area(V, F):
    """Total surface area of a triangle mesh; ``0.0`` for an empty/invalid mesh."""
    if F is None or len(F) == 0:
        return 0.0
    try:
        return float(gpy.doublearea(np.asarray(V, float),
                                    np.asarray(F, np.int64)).sum()) / 2.0
    except Exception:
        return 0.0


def sanitize(V, F):
    """Return ``(V, F)`` with degenerate faces and unreferenced vertices removed.

    Drops faces that (a) index a vertex outside ``[0, len(V))``, (b) touch a
    non-finite (NaN/inf) vertex, or (c) repeat a vertex index (zero area by
    construction), then removes vertices no surviving face references. ``F`` may
    come back empty (shape ``(0, 3)``) when nothing valid remains. Deliberately
    conservative: it only removes invalid data, never relocating or adding any.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if V.size == 0 or F is None or len(F) == 0:
        return V, np.zeros((0, 3), np.int64)
    nV = len(V)
    F = F[np.all((F >= 0) & (F < nV), axis=1)]            # in-range indices only
    if len(F) == 0:
        return V, np.zeros((0, 3), np.int64)
    finite_v = np.all(np.isfinite(V), axis=1)
    F = F[finite_v[F].all(axis=1)]                        # no non-finite vertices
    if len(F) == 0:
        return V, np.zeros((0, 3), np.int64)
    nondegen = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[nondegen]                                       # no repeated-index faces
    if len(F) == 0:
        return V, np.zeros((0, 3), np.int64)
    # Drop coincident (duplicate) faces -- the same three vertices in any winding.
    # These are non-manifold and double a face's contribution to area/Laplacian,
    # which would corrupt the very signatures the detectors compare; a clean mesh
    # has none, so this only ever removes garbage (e.g. the degenerate adversary's
    # duplicated faces). First occurrence is kept, preserving the original order.
    _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    F = F[np.sort(keep)]
    if len(F) == 0:
        return V, np.zeros((0, 3), np.int64)
    try:
        V2, F2 = gpy.remove_unreferenced(V, F)
        return np.asarray(V2, float), np.asarray(F2, np.int64)
    except Exception:
        return V, F


def split_components(V, F, maxc=None, do_sanitize=True):
    """List the connected components of ``(V, F)`` as ``(Vc, Fc)`` pairs.

    Components are found by shared-vertex face adjacency
    (``gpytoolbox.connected_components``); ``do_sanitize`` runs ``sanitize`` first
    (the default) so degenerate faces never poison the split. A single-component
    mesh returns one item (whole-mesh analysis).

    Per-component analysis (a spectrum / point sample) is the cost, so how many
    components are returned is controlled by ``maxc``:

    * ``maxc=None`` (default): return **all** components, sorted ascending by face
      count (cheapest first), letting a caller-side per-shape *timeout* bound the
      work. ``laplacian_spectrum`` uses this. CAVEAT: defeatable by *component
      flooding* -- padding a mesh with a huge number of trivial components can
      exhaust the timeout before the device's component is scored (it then defaults
      to benign). See the README to-do on a device-likeness pre-filter.
    * ``maxc`` an int: keep at most ``maxc`` components -- the ``maxc//2`` smallest
      + ``maxc//2`` largest by area (used by ``pointnet``).
    """
    if do_sanitize:
        V, F = sanitize(V, F)
    if F is None or len(F) == 0:
        return []
    try:
        _, CF = gpy.connected_components(F, return_face_indices=True)
    except Exception:
        return [(np.asarray(V, float), np.asarray(F, np.int64))]
    comps = []
    for lab in np.unique(CF):
        Fc = F[CF == lab]
        try:
            Vc, Fc2 = gpy.remove_unreferenced(V, Fc)
        except Exception:
            continue
        if len(Fc2) > 0 and len(Vc) >= 3:
            comps.append((np.asarray(Vc, float), np.asarray(Fc2, np.int64)))
    if maxc is None:
        comps.sort(key=lambda c: len(c[1]))   # ascending by size; the timeout bounds work
        return comps
    if len(comps) > maxc:
        areas = np.array([mesh_area(Vc, Fc) for Vc, Fc in comps])
        order = np.argsort(areas)
        nsmall = maxc // 2
        keep = sorted(set(order[:nsmall].tolist())
                      | set(order[-(maxc - nsmall):].tolist()))
        comps = [comps[i] for i in keep]
    return comps
