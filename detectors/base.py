import os
import csv
import json
import time
import inspect
import multiprocessing as mp
from collections import namedtuple
from typing import final

# Per-shape evaluation budget shared by *every* detector: if a detector cannot
# score one shape within this many seconds it is stopped and that shape defaults
# to benign (score 0.0). This mirrors the per-shape timeout enforced during data
# generation (``load_data.py``) and makes "too slow to be practical" count
# against a detector (a timed-out malign shape becomes a miss). Override with the
# ``DETECTOR_EVAL_TIMEOUT`` env var (seconds), mainly for tests.
EVAL_TIMEOUT_SECONDS = float(os.environ.get("DETECTOR_EVAL_TIMEOUT", "60"))

# A dataset split: lists of .obj file paths for each class.
Split = namedtuple("Split", ["benign", "malign"])


def make_entry(name, method_output, true_label):
    return {
        "name": name,
        "score": method_output["score"],
        "label": true_label
    }


def _list_objs(directory):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if f.lower().endswith(".obj")]


def load_split(data_root, split):
    """Return a ``Split`` of .obj paths for ``data_root/<split>/{benign,malign}``."""
    return Split(
        benign=_list_objs(os.path.join(data_root, split, "benign")),
        malign=_list_objs(os.path.join(data_root, split, "malign")),
    )


# ---------------------------------------------------------------------------
# Per-shape eval timeout. Each shape is scored in a reusable ``spawn`` worker so
# the parent can enforce a hard wall-clock limit even on a detector stuck in a
# native call (e.g. a huge eigensolve) -- a Python-level signal/thread could not.
# The worker reconstructs the detector from its class with ``cls()`` + ``load()``
# (the documented save/load contract every detector already honors so plain
# ``evaluate.py`` can score without retraining), so it owns the trained model. On
# overrun the worker is killed and respawned; the shape defaults to benign. A
# ``spawn`` context is used (never ``fork``) for the same native-load safety the
# data pipeline relies on. Mirrors ``load_data._TimeoutPool``.
# ---------------------------------------------------------------------------
def _eval_worker(conn, cls):
    """Reconstruct ``cls`` (``cls()`` + ``load()``) once, then score the obj paths
    sent over ``conn`` until told to stop."""
    try:
        inst = cls()
        inst.load()
    except Exception as exc:                       # model cache missing/broken
        try:
            conn.send(("fatal", repr(exc)))
        except Exception:
            pass
        conn.close()
        return
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        try:
            out = inst.eval(msg)
            conn.send(("ok", float(out["score"])))
        except Exception as exc:                   # one bad shape: report, keep going
            conn.send(("err", repr(exc)))
    conn.close()


class _EvalTimeoutPool:
    """A single reusable spawn worker that scores one shape at a time under a
    wall-clock timeout, killing and reloading itself if a shape overruns or the
    worker dies."""

    READY_TIMEOUT = 600.0  # generous budget for the worker's one-time model load

    def __init__(self, cls, timeout):
        self.cls = cls
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self._spawn()

    def _spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_eval_worker, args=(child, self.cls),
                                     daemon=True)
        self.proc.start()
        child.close()  # parent keeps only its end of the pipe
        if not self.conn.poll(self.READY_TIMEOUT):
            self._kill()
            raise RuntimeError(
                f"{self.cls.__name__}: eval worker not ready within "
                f"{self.READY_TIMEOUT:.0f}s (model load stuck).")
        status, payload = self.conn.recv()
        if status != "ready":
            self._kill()
            raise RuntimeError(f"{self.cls.__name__}: eval worker load failed: {payload}")

    def _kill(self):
        try:
            self.proc.kill()
            self.proc.join()
        except Exception:
            pass

    def _restart(self):
        self._kill()
        self._spawn()

    def score(self, path):
        """Score one shape. Returns ``("ok", score)``, ``("timeout", None)``, or
        ``("error", message)``. The worker survives an ``"error"`` (a single bad
        shape) but is restarted after a timeout or a crash."""
        try:
            self.conn.send(path)
        except (BrokenPipeError, OSError) as exc:
            self._restart()
            return "error", f"send failed: {exc!r}"
        if not self.conn.poll(self.timeout):
            self._restart()                        # worker stuck mid-eval: kill it
            return "timeout", None
        try:
            status, payload = self.conn.recv()
        except (EOFError, OSError) as exc:
            self._restart()                        # worker died (e.g. native crash)
            return "error", f"recv failed: {exc!r}"
        return ("ok", payload) if status == "ok" else ("error", payload)

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self._kill()


class Detector():

    def __init__(self):
        pass

    @property
    def _pkg_dir(self) -> str:
        """Directory of the subclass's ``method.py`` (its package folder)."""
        return os.path.dirname(inspect.getfile(type(self)))

    @property
    def data_dir(self) -> str:
        """Tracked per-method folder for *pre-shipped* assets: ``detectors/<pkg>/data/``.

        Use this for files that ship with the method and belong in git -- custom
        reference shapes, lookup tables, and the like. It is **not** git-ignored
        and **not** auto-created (commit the folder alongside your assets).
        Anything a method *learns* during training goes in ``trained_dir`` instead.
        """
        return os.path.join(self._pkg_dir, "data")

    @property
    def trained_dir(self) -> str:
        """Git-ignored per-method folder for training output: ``detectors/<pkg>/trained/``.

        Created on first access. ``train``/``save`` write the learned state here
        (checkpoints, signatures, ...) and ``load`` reads it back, so ``eval`` can
        run later without retraining. It is git-ignored because these artifacts
        are regenerated by ``evaluate.py --train``.
        """
        d = os.path.join(self._pkg_dir, "trained")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def installed_dir(self) -> str:
        """Git-ignored per-method folder for ``install`` output: ``detectors/<pkg>/installed/``.

        Created on first access. Use it for one-time build artifacts produced by
        ``install`` (compiled binaries, fetched model weights, ...) that are
        independent of the dataset, so changing the data and re-running
        ``train`` does not require re-installing. Git-ignored like ``trained_dir``.
        """
        d = os.path.join(self._pkg_dir, "installed")
        os.makedirs(d, exist_ok=True)
        return d

    def install(self) -> None:
        """One-time setup independent of the dataset (default: no-op).

        Runs only when ``evaluate.py --install`` is passed, *before* ``train``.
        Override it for methods that need to build native code (cmake/make),
        fetch large model weights, or otherwise prepare prerequisites once;
        write any artifacts into ``self.installed_dir`` so a later run can reuse
        them without re-installing. Most pure-Python detectors need nothing here.
        """
        pass

    def train(self, train: Split, test: Split = None) -> None:
        """Optionally fit the method before evaluation (default: no-op).

        Called once, before any ``eval`` call, and *only* when training is
        requested (``evaluate.py --train``). ``train`` is the training split and
        ``test`` is a validation split for hyperparameter tuning; both are
        ``Split`` objects whose ``.benign``/``.malign`` are lists of .obj paths.
        Store anything learned on ``self`` for ``eval`` to use, and persist it in
        ``save`` so a later run can ``load`` it instead of retraining.
        """
        pass

    def save(self) -> None:
        """Optionally persist what ``train`` learned into ``self.trained_dir`` (default: no-op).

        Called once right after ``train`` (so only under ``--train``). Override it
        in methods that learn state, writing every artifact ``eval`` needs into
        ``self.trained_dir``.
        """
        pass

    def load(self) -> None:
        """Optionally restore what ``save`` wrote from ``self.trained_dir`` (default: no-op).

        Called instead of ``train`` whenever evaluation runs *without* ``--train``
        (the default), so ``eval`` never retrains. Override it to mirror ``save``,
        and raise a clear error if the cached artifact is missing so the user
        knows to run ``evaluate.py --train`` once first.
        """
        pass

    # Potentially have another wrapper function that collects extra metrics
    # e.g runtime and have that return the result dict.  Then have this purely
    # return method outputs?
    def eval(self, obj_path) -> dict:
        """
        Should return a dict containing the following keys
        {
        'score' : Confidence score of object being malign. [0, 1].
        }
        """
        raise NotImplementedError("Must be Implemented by Child Method")

    @final
    def eval_method(self, output_path: str, data_root: str = "data",
                    train: bool = False, install: bool = False,
                    timeout: float = EVAL_TIMEOUT_SECONDS):
        """Prepare the method, then score the held-out eval split.

        When ``install`` is true, ``install()`` runs first (one-time, dataset-
        independent setup). When ``train`` is true the method is fit from scratch:
        ``train()`` runs on the train/validation splits and ``save()`` then caches
        the result into ``self.trained_dir``. When ``train`` is false (the
        default) nothing is retrained -- ``load()`` restores the cached state
        instead, so evaluation stays cheap even for methods that take a long time
        to fit. Either way the method is then run over the held-out eval split,
        writing one row per shape to ``<output_path>/outputs.csv`` (``name,
        score, label``; label 0 = benign, 1 = malign). Methods never see the eval
        labels; reporting (ROC/AUC, plots, leaderboard) is handled by
        ``generate_report.py``.

        Wall-clock timings (install/train/load, and mean/total per-shape eval)
        are written to ``<output_path>/timing.json`` so the report can include
        them. Returns the number of shapes scored.
        """
        os.makedirs(output_path, exist_ok=True)
        timing = {}

        if install:
            t0 = time.perf_counter()
            self.install()
            timing["install_seconds"] = time.perf_counter() - t0

        if train:
            t0 = time.perf_counter()
            self.train(load_split(data_root, "train"), load_split(data_root, "test"))
            self.save()
            timing["train_seconds"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            self.load()
            timing["load_seconds"] = time.perf_counter() - t0

        eval_split = load_split(data_root, "eval")

        # Score every shape in a reusable worker under a shared per-shape timeout.
        # A shape the detector can't decide within ``timeout`` (or that crashes the
        # worker) defaults to benign (0.0), so a too-slow malign shape counts as a
        # miss and no single shape can hang the whole run.
        cls_name = type(self).__name__
        method_results = []
        eval_times = []
        n_timeout = n_error = 0
        pool = _EvalTimeoutPool(type(self), timeout)
        try:
            for label, paths in ((0, eval_split.benign), (1, eval_split.malign)):
                for path in paths:
                    t0 = time.perf_counter()
                    status, value = pool.score(path)
                    eval_times.append(time.perf_counter() - t0)
                    if status == "ok":
                        score = value
                    else:
                        score = 0.0  # timed-out / crashed -> default to benign
                        if status == "timeout":
                            n_timeout += 1
                            print(f"[{cls_name}] {os.path.basename(path)} TIMEOUT after "
                                  f"{timeout:.0f}s -> scored benign (0.0)", flush=True)
                        else:
                            n_error += 1
                            print(f"[{cls_name}] {os.path.basename(path)} ERROR {value} "
                                  f"-> scored benign (0.0)", flush=True)
                    method_results.append(
                        make_entry(os.path.basename(path), {"score": score}, label))
        finally:
            pool.close()
        if len(method_results) == 0:
            print("No objects evaluated")
            return 0
        if n_timeout or n_error:
            print(f"[{cls_name}] {n_timeout} timed out, {n_error} errored out of "
                  f"{len(method_results)} shapes (all scored benign).", flush=True)

        timing["n_eval"] = len(eval_times)
        timing["eval_seconds_total"] = sum(eval_times)
        timing["eval_seconds_mean"] = sum(eval_times) / len(eval_times)
        timing["eval_timeout_seconds"] = timeout
        timing["n_timeout"] = n_timeout
        timing["n_error"] = n_error

        headers = method_results[0].keys()
        with open(os.path.join(output_path, "outputs.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(method_results)
        with open(os.path.join(output_path, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

        return len(method_results)
