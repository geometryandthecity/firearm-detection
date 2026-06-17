import os
import sys
import shutil
import tempfile
import contextlib

import numpy as np
import gpytoolbox as gpy
import wildmeshing as wm

from ..base import Adversary
from ..jitter import Jitter
from ..rigid import RigidTransform


@contextlib.contextmanager
def _quiet_isolated_run():
    """Run fTetWild quietly and without littering the working directory.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="ftetwild_")
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.chdir(tmp)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_out)
        os.close(old_err)
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


class TetWild(Adversary):
    """Re-mesh the surface using tetwild (aka the *same way* the benign Thingi10K shapes were made)

    Reproducibility note: fTetWild carries internal randomness with no seed
    parameter, so the exact tessellation is
    not bit-identical across runs (vertex/face counts wobble by a few percent).
    """

    def __init__(self, edge_frac_range=(0.04, 0.06), epsilon=1e-3, jitter_sigma_frac=1e-3):
        self.edge_frac_range = edge_frac_range
        self.epsilon = epsilon
        # Post-remesh perturbations (cheap, deterministic from rng): a random
        # isometry and small vertex jitter, so tetwild shapes vary in pose and
        # exact coordinates instead of all sharing the base pose.
        self._rigid = RigidTransform()
        self._jitter = Jitter(sigma_frac=jitter_sigma_frac)

    def __call__(self, vertices, faces, rng):
        lo, hi = self.edge_frac_range
        edge_length_r = float(rng.uniform(lo, hi))
        with _quiet_isolated_run():
            tet = wm.Tetrahedralizer(
                epsilon=self.epsilon,
                edge_length_r=edge_length_r,
                max_threads=1,
            )
            tet.set_mesh(vertices.astype(np.float64), faces.astype(np.int32))
            tet.tetrahedralize()
            V_tet, T_tet, _ = tet.get_tet_mesh()
        boundary = gpy.boundary_faces(np.asarray(T_tet, dtype=np.int32))
        V_out, F_out = gpy.remove_unreferenced(np.asarray(V_tet, dtype=np.float64), boundary)[:2]
        # Perturb the remeshed surface with a random isometry, then small jitter.
        V_out, F_out = self._rigid(np.asarray(V_out), np.asarray(F_out), rng)
        V_out, F_out = self._jitter(V_out, F_out, rng)
        return np.asarray(V_out), np.asarray(F_out)
