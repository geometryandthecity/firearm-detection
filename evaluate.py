import argparse
import concurrent.futures
import functools
import json
import os

from detectors.base import Split, load_split
from naming import adversary_of
from detectors.always_benign import AlwaysBenign
from detectors.always_malign import AlwaysMalign
from detectors.always_random import Random
from detectors.file_hash import FileHash
from detectors.laplacian_spectrum import LaplacianSpectrum
from detectors.pointnet import PointNet
from detectors.image_classifier import ImageClassifier
# Deferred: the PSO solver builds and runs on macOS, but it runs a full
# 1500-particle x 200-iteration swarm per mesh (~2-3 min/shape, ~9h for the
# eval split), so it's left out of routine runs for now.
# from detectors.pso_fit import ParticleSwarm

DETECTORS = [
    ("always_benign", AlwaysBenign),
    ("always_malign", AlwaysMalign),
    ("always_random", Random),
    ("file_hash", FileHash),
    ("laplacian_spectrum", LaplacianSpectrum),
    ("pointnet", PointNet),
    ("image_classifier", ImageClassifier),
    # ("pso", ParticleSwarm),  # deferred: too slow for routine runs (see import above)
]

def _run(args, train=False, install=False):
    name, cls = args
    print(f"[{name}] {'Training + scoring' if train else 'Loading + scoring'}...", flush=True)
    try:
        instance = cls()
        n = instance.eval_method(os.path.join("output", "by_detector", name),
                                 train=train, install=install)
        print(f"[{name}] Done.", flush=True)
        return name, True, n
    except Exception as e:
        print(f"[{name}] Failed: {e}", flush=True)
        return name, False, str(e)


def _read_timing(name):
    """Return the timing dict written by ``eval_method`` for ``name``, or {}."""
    try:
        with open(os.path.join("output", "by_detector", name, "timing.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Leave-one-out (LOO) regime (``--loo``)
#
# Where the standard run trains every detector on *all* adversaries, LOO asks the
# harder question -- how well does a detector catch a disguise it never saw in
# training? For each adversary A in turn it retrains every detector on all the
# *other* adversaries (A dropped from both the train and the validation/test
# split, so nothing tunes on it) and scores that model only on A's malign shapes,
# against all benign shapes. Looping over every adversary yields one fold per
# adversary, so the cost is roughly ``#adversaries`` full training runs.
#
# Raw scores land in the same per-detector layout the standard run uses, one tree
# per held-out adversary:
#     output/loo/<held-out adversary>/by_detector/<detector>/{outputs.csv,timing.json}
# Because each fold's eval split is *all* benign + only the held-out adversary's
# malign, a detector's overall AUC there is exactly its AUC at catching the unseen
# adversary -- directly comparable to the standard (detector, adversary) cell.
# ``generate_report.py`` turns these into output/loo/leaderboard_loo.md (heatmap +
# standard-vs-LOO generalization gap). Each fold trains a separate model, isolated
# under trained/loo/<adversary>/ (via ``trained_subdir``), so a later plain
# ``evaluate.py`` still finds the full-data model and nothing is clobbered.
# ---------------------------------------------------------------------------
def _adversaries_in(split):
    """Sorted adversary names present in a split's malign filenames."""
    return sorted({a for p in split.malign
                   if (a := adversary_of(os.path.basename(p))) is not None})


def _filter_malign(paths, *, keep=None, drop=None):
    """Subset of malign ``paths`` by their adversary: keep only ``keep``, or drop
    ``drop`` (exactly one is given). Paths whose adversary can't be parsed are
    excluded either way, so an unlabelled shape never leaks across the split."""
    out = []
    for p in paths:
        a = adversary_of(os.path.basename(p))
        if a is None:
            continue
        if keep is not None and a != keep:
            continue
        if drop is not None and a == drop:
            continue
        out.append(p)
    return out


def _run_loo(task):
    """Pool worker: train a detector on the fold's filtered train split and score
    the held-out adversary's eval split. ``task`` is fully picklable so it crosses
    the process boundary cleanly."""
    name, cls, adv, output_path, train_split, test_split, eval_split, install = task
    print(f"[{adv}/{name}] training on all-but-{adv}, scoring held-out {adv}...",
          flush=True)
    try:
        n = cls().eval_method(
            output_path, train=True, install=install,
            train_split=train_split, test_split=test_split, eval_split=eval_split,
            trained_subdir=os.path.join("loo", adv))
        print(f"[{adv}/{name}] done ({n} shapes).", flush=True)
        return adv, name, True, n
    except Exception as e:  # one detector failing a fold must not sink the rest
        print(f"[{adv}/{name}] FAILED: {e}", flush=True)
        return adv, name, False, str(e)


def run_loo(selected, args):
    """Run the leave-one-out sweep over every (selected) adversary fold."""
    train_full = load_split(args.data_root, "train")
    test_full = load_split(args.data_root, "test")
    eval_full = load_split(args.data_root, "eval")

    present = _adversaries_in(train_full)
    if not present:
        raise SystemExit(f"No adversaries found in {args.data_root}/train/malign; "
                         f"run load_data.py first.")
    if args.adversaries:
        unknown = [a for a in args.adversaries if a not in present]
        if unknown:
            raise SystemExit(f"Unknown adversary(ies): {unknown}. "
                             f"Present in train/malign: {present}")
        held = list(args.adversaries)
    else:
        held = present
    if len(present) < 2:
        print(f"[warn] only {len(present)} adversary present ({present}); a "
              f"leave-one-out fold then trains on benign shapes alone.", flush=True)

    loo_root = os.path.join("output", "loo")
    print(f"Leave-one-out over {len(held)} adversary fold(s): {held}\n"
          f"Detectors: {[n for n, _ in selected]}\n"
          f"Each fold retrains from scratch -- this is ~{len(held)}x a normal run.",
          flush=True)

    install_pending = args.install
    all_results = []
    for adv in held:
        tr = Split(train_full.benign, _filter_malign(train_full.malign, drop=adv))
        te = Split(test_full.benign, _filter_malign(test_full.malign, drop=adv))
        ev = Split(eval_full.benign, _filter_malign(eval_full.malign, keep=adv))
        print(f"\n=== Held-out adversary: {adv}  "
              f"(train malign {len(tr.malign)}, held-out eval malign {len(ev.malign)}, "
              f"benign {len(ev.benign)}) ===", flush=True)
        if not ev.malign:
            print(f"[{adv}] no malign shapes for this adversary in the eval split "
                  f"-- skipping fold.", flush=True)
            continue
        install_now = install_pending
        install_pending = False  # install is dataset-independent: once is enough
        tasks = [(name, cls, adv,
                  os.path.join(loo_root, adv, "by_detector", name),
                  tr, te, ev, install_now)
                 for name, cls in selected]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            fold_results = list(ex.map(_run_loo, tasks))
        all_results.extend(fold_results)
        ok = sum(1 for _, _, s, _ in fold_results if s)
        print(f"--- {adv}: {ok}/{len(fold_results)} detectors scored ---", flush=True)

    print("\n=== Leave-one-out summary ===")
    for adv in held:
        scored = [(n, info) for a, n, s, info in all_results if a == adv and s]
        if scored:
            print(f"  {adv:14s} " + ", ".join(f"{n}={info}" for n, info in scored))
    failures = [(a, n, info) for a, n, s, info in all_results if not s]
    if failures:
        print("\nFailures:")
        for adv, name, info in failures:
            print(f"  {adv}/{name}: {info}")
    print(f"\nRaw scores under {loo_root}/<adversary>/by_detector/<detector>/.")
    print("Run `python generate_report.py` to build the LOO leaderboard and heatmap.")


def main():
    parser = argparse.ArgumentParser(
        description="Score detectors in DETECTORS on the held-out eval split."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Retrain (selected) detectors from scratch and cache the result to each "
             "detector's trained/ folder. Without this flag (the default) detectors "
             "load their cached artifacts and nothing is retrained -- run with "
             "--train once before the first plain `evaluate.py`. (Ignored with --loo, "
             "which always retrains per fold.)",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Run each detector's one-time install() (build native code, fetch "
             "weights, ...) before training. Off by default, so installation is "
             "not repeated on every run; pass it once after adding/changing a detector.",
    )
    parser.add_argument(
        "--detectors", nargs="+", metavar="NAME", default=None,
        help="Only run these detectors (by name); default: all of "
             f"{[n for n, _ in DETECTORS]}. Lets you install/train/score a single "
             "detector without touching the others (e.g. --detectors image_classifier).",
    )
    parser.add_argument(
        "--loo", action="store_true",
        help="Leave-one-out regime: instead of the standard run, loop over the "
             "adversaries and, for each, retrain every detector on all the *others* "
             "and score it only on the held-out adversary. Writes to "
             "output/loo/<adversary>/by_detector/; rebuild the LOO leaderboard with "
             "generate_report.py. Always retrains per fold (~#adversaries x a normal "
             "run), so it leaves the standard trained/ model untouched.",
    )
    parser.add_argument(
        "--adversaries", nargs="+", metavar="NAME", default=None,
        help="With --loo: only hold out these adversaries (one fold each); default: "
             "every adversary found in data/train/malign.",
    )
    parser.add_argument("--data-root", default="data",
                        help="Dataset root with train/test/eval splits (default: data).")
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Detectors scored concurrently (default: os.cpu_count()). Lower it "
             "(e.g. 2) if the GPU-backed detectors exhaust memory in parallel.")
    args = parser.parse_args()

    selected = DETECTORS
    if args.detectors:
        by_name = dict(DETECTORS)
        unknown = [m for m in args.detectors if m not in by_name]
        if unknown:
            raise SystemExit(f"Unknown detector(s): {unknown}. "
                             f"Available: {[n for n, _ in DETECTORS]}")
        selected = [(n, by_name[n]) for n in args.detectors]

    if args.loo:
        run_loo(selected, args)
        return

    os.makedirs(os.path.join("output", "by_detector"), exist_ok=True)

    run = functools.partial(_run, train=args.train, install=args.install)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(run, selected))

    print("\n=== Evaluation Summary ===")
    for name, success, info in results:
        if not success:
            print(f"  {name:20s}  FAILED — {info}")
            continue
        t = _read_timing(name)
        bits = []
        if "install_seconds" in t:
            bits.append(f"install {t['install_seconds']:.1f}s")
        if "train_seconds" in t:
            bits.append(f"train {t['train_seconds']:.1f}s")
        if "eval_seconds_mean" in t:
            bits.append(f"eval {t['eval_seconds_mean'] * 1e3:.1f}ms/shape")
        extra = ("  [" + ", ".join(bits) + "]") if bits else ""
        print(f"  {name:20s}  scored {info} shapes{extra}")
    if not args.train and any(not s for _, s, _ in results):
        print("\nSome detectors failed to load a cache. Run `python evaluate.py --train` "
              "once to train and cache them first.")
    print("\nRun `python generate_report.py` to build the ROC comparison and leaderboard.")

if __name__ == "__main__":
    main()
