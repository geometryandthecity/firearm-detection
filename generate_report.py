#!/usr/bin/env python3
"""Aggregate the raw detector scores under ``output/`` into plots and a leaderboard.

This is deliberately separate from ``evaluate.py``. Evaluation scores each
detector once and writes raw per-shape results to
``output/by_detector/<detector>/outputs.csv`` (the expensive step). This script
only *reads* those CSVs to (re)generate the ROC curves and leaderboard (the
cheap step you iterate on), so you can tune the report without re-scoring.

Detectors are discovered, not hardcoded: every sub-folder of ``output/by_detector/``
that contains an ``outputs.csv`` is included. The report is written into a fixed
layout::

    output/
      by_detector/<detector>/    roc.png, metrics.txt  (beside evaluate.py's outputs.csv)
      by_adversary/<adversary>/ roc.png, metrics.csv, metrics.txt
      roc_all.png               every detector's overall ROC, overlaid
      heatmap_all.png           detector x adversary AUC grid
      leaderboard.md            symmetric detector/adversary leaderboard

    python generate_report.py [--output-dir output]
"""

import argparse
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from naming import adversary_of

OUTPUT_DIR = "output"


def read_rows(csv_path):
    """Read ``(names, labels, scores)`` from a detector's ``outputs.csv``.

    ``names`` (the per-shape filenames) are kept so malign rows can be grouped
    by the adversary that produced them (see ``adversary_of``).
    """
    names, labels, scores = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return names, np.asarray(labels), np.asarray(scores)


def read_timing(detector_dir):
    """Return the timing dict ``evaluate.py`` wrote for a detector, or ``{}``.

    ``evaluate.py`` records wall-clock timings (one-time ``install``, then
    ``train`` *or* ``load``, plus mean/total per-shape ``eval``) to
    ``<detector_dir>/timing.json``. Surfacing them in the leaderboard lets cost and
    accuracy be read together. Older runs without the file simply show ``—``.
    """
    try:
        with open(os.path.join(detector_dir, "timing.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _adv_roc(result, adversary):
    """ROC of *all* benign vs one adversary's malign, for a single detector.

    Returns a dict with ``fpr``/``tpr``/``auc`` and the ``n_benign``/``n_malign``
    that went into it, or ``None`` when either side is empty (curve undefined).
    """
    benign = result["scores"][result["labels"] == 0]
    mal = np.asarray(
        [sc for nm, lab, sc in zip(result["names"], result["labels"], result["scores"])
         if lab == 1 and adversary_of(nm) == adversary],
        dtype=float,
    )
    if benign.size == 0 or mal.size == 0:
        return None
    y_true = np.concatenate([np.zeros(benign.size), np.ones(mal.size)])
    y_score = np.concatenate([benign, mal])
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "auc": metrics.auc(fpr, tpr),
            "n_benign": int(benign.size), "n_malign": int(mal.size)}


def collect(by_detector_dir):
    """Discover every detector under ``by_detector_dir`` and compute its ROC/AUC."""
    results = []
    if not os.path.isdir(by_detector_dir):
        return results
    for name in sorted(os.listdir(by_detector_dir)):
        csv_path = os.path.join(by_detector_dir, name, "outputs.csv")
        if not os.path.isfile(csv_path):
            continue
        names, labels, scores = read_rows(csv_path)
        if labels.size == 0 or np.unique(labels).size < 2:
            print(f"[skip] {name}: needs both benign and malign scores")
            continue
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        results.append({
            "name": name,
            "names": names,
            "labels": labels,
            "scores": scores,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": metrics.auc(fpr, tpr),
            "n_benign": int((labels == 0).sum()),
            "n_malign": int((labels == 1).sum()),
            "timing": read_timing(os.path.join(by_detector_dir, name)),
        })
    return results


def write_per_detector(result, by_detector_dir, adversaries):
    """Write one detector's ``metrics.txt`` and ``roc.png`` (split by adversary).

    Lands beside the ``outputs.csv`` that ``evaluate.py`` wrote in
    ``by_detector/<detector>/``. ``metrics.txt`` reports the overall AUC plus a
    per-adversary AUC breakdown, and ``roc.png`` overlays the overall ROC with
    one curve per adversary, so you can see where the detector is strong or weak.
    """
    d = os.path.join(by_detector_dir, result["name"])
    os.makedirs(d, exist_ok=True)

    # per-adversary ROC for this detector (skip adversaries it has no malign for)
    per_adv = {a: r for a in adversaries if (r := _adv_roc(result, a)) is not None}
    weakest_first = sorted(per_adv, key=lambda a: per_adv[a]["auc"])

    with open(os.path.join(d, "metrics.txt"), "w") as f:
        f.write(f"Detector: {result['name']}\n")
        f.write(f"Area Under ROC (overall): {result['auc']}\n")
        f.write(f"# benign: {result['n_benign']}    # malign: {result['n_malign']}\n")
        if weakest_first:
            f.write("\nAUC by adversary (all benign vs that adversary's malign, "
                    "weakest detection first):\n")
            for a in weakest_first:
                r = per_adv[a]
                f.write(f"  {a:32s} AUC={r['auc']:.4f}  (# malign: {r['n_malign']})\n")
        f.write("\nOverall ROC curve:\n")
        f.write("False Positive Rates " + ", ".join(map(str, result["fpr"])) + "\n")
        f.write("True Positive Rates " + ", ".join(map(str, result["tpr"])) + "\n")
        f.write("Thresholds " + ", ".join(map(str, result["thresholds"])) + "\n")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(result["fpr"], result["tpr"], marker=".", linewidth=2.5, color="black",
            label=f"overall (AUC = {result['auc']:.3f})")
    for a in weakest_first:
        r = per_adv[a]
        ax.plot(r["fpr"], r["tpr"], marker=".", linewidth=1, alpha=0.8,
                label=f"{a} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- {result['name']}")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(d, "roc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_per_adversary(adversary, results, by_adversary_dir):
    """Write one adversary's ``roc.png`` + ``metrics.csv``/``metrics.txt``.

    The mirror of ``write_per_detector``: every detector is scored on the same task
    -- separate *all* benign shapes from just the malign shapes this adversary
    produced -- so ``roc.png`` overlays one curve per detector and the metrics
    rank detectors by how well they catch this adversary.
    """
    scored = [(r["name"], roc) for r in results
              if (roc := _adv_roc(r, adversary)) is not None]
    if not scored:
        return None
    scored.sort(key=lambda kv: kv[1]["auc"], reverse=True)  # best detector first

    d = os.path.join(by_adversary_dir, adversary)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "auc", "n_benign", "n_malign"])
        for name, roc in scored:
            w.writerow([name, f"{roc['auc']:.6f}", roc["n_benign"], roc["n_malign"]])

    best_name, best = scored[0]
    with open(os.path.join(d, "metrics.txt"), "w") as f:
        f.write(f"Adversary: {adversary}\n")
        f.write(f"# malign shapes: {best['n_malign']}    "
                f"# benign (shared negatives): {best['n_benign']}\n")
        f.write(f"Best detector: {best_name} (AUC = {best['auc']:.4f})\n")
        f.write(f"Mean AUC across detectors: "
                f"{_mean([roc['auc'] for _, roc in scored]):.4f}\n")
        f.write("\nDetector ranking (higher AUC = better detection):\n")
        for name, roc in scored:
            f.write(f"  {name:32s} AUC={roc['auc']:.4f}\n")

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, roc in scored:
        ax.plot(roc["fpr"], roc["tpr"], marker=".",
                label=f"{name} (AUC = {roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- adversary: {adversary}")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(d, "roc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return d


def write_composite(results, output_dir):
    """Write one ROC plot overlaying every detector (ranked by AUC)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in sorted(results, key=lambda r: r["auc"], reverse=True):
        ax.plot(r["fpr"], r["tpr"], marker=".",
                label=f"{r['name']} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC comparison")
    ax.legend(loc="lower right")
    path = os.path.join(output_dir, "roc_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean(values):
    """Mean of the non-``None`` entries (``nan`` if there are none)."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def challenge_matrix(results):
    """Per-(detector, adversary) detection AUC -- the symmetric challenge grid.

    For each detector and each adversary, computes the ROC AUC separating
    *all* benign shapes (the shared negatives) from just the malign shapes that
    adversary produced. Reading the grid one way answers "which detector is best
    for this adversary?"; the other way answers "which adversary best evades this
    detector?". Returns ``(adversaries, matrix, counts)`` where ``adversaries``
    is the adversary list ordered by evasion (strongest evader first),
    ``matrix[(detector, adv)]`` is the AUC (or ``None`` when a detector has no malign
    shapes for that adversary), and ``counts[adv]`` is the number of malign shapes
    for that adversary.
    """
    adversaries = sorted({
        a for r in results for nm, lab in zip(r["names"], r["labels"])
        if lab == 1 and (a := adversary_of(nm)) is not None
    })
    matrix, counts = {}, {}
    for r in results:
        benign = r["scores"][r["labels"] == 0]
        by_adv = {}
        for nm, lab, sc in zip(r["names"], r["labels"], r["scores"]):
            if lab == 1 and (a := adversary_of(nm)) is not None:
                by_adv.setdefault(a, []).append(sc)
        for a in adversaries:
            mal = np.asarray(by_adv.get(a, []), dtype=float)
            if mal.size == 0 or benign.size == 0:
                matrix[(r["name"], a)] = None
                continue
            y_true = np.concatenate([np.zeros(benign.size), np.ones(mal.size)])
            y_score = np.concatenate([benign, mal])
            matrix[(r["name"], a)] = float(metrics.roc_auc_score(y_true, y_score))
            counts[a] = mal.size
    # Order adversaries by evasion (mean AUC across detectors, strongest first)
    # so every artifact -- tables and heatmap -- agrees on the ranking.
    adv_mean = {a: _mean([matrix[(r["name"], a)] for r in results]) for a in adversaries}
    adversaries.sort(key=lambda a: adv_mean[a])
    return adversaries, matrix, counts


def write_matrix_heatmap(results, adversaries, matrix, output_dir,
                         path=None, title=None):
    """Heatmap of the detector x adversary AUC grid (lower = better evasion).

    ``path``/``title`` override the defaults so the same renderer draws both the
    standard grid and the leave-one-out grid; ``results`` only supplies the
    detector column order via ``r["name"]``.
    """
    detectors = [r["name"] for r in results]
    M = np.array([[matrix[(m, a)] if matrix[(m, a)] is not None else np.nan
                   for m in detectors] for a in adversaries])
    fig, ax = plt.subplots(figsize=(1.1 * len(detectors) + 3, 0.5 * len(adversaries) + 2))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels(detectors, rotation=45, ha="right")
    ax.set_yticks(range(len(adversaries)))
    ax.set_yticklabels(adversaries)
    for i in range(len(adversaries)):
        for j in range(len(detectors)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.75 else "black", fontsize=8)
    ax.set_title(title or "Detector × adversary AUC (lower = adversary evades better)")
    fig.colorbar(im, ax=ax, label="AUC")
    path = path or os.path.join(output_dir, "heatmap_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_leaderboard(results, adversaries, matrix, counts, output_dir, loo_link=None):
    """Write (and return) the symmetric challenge report as ``leaderboard.md``.

    Sections: detectors ranked by overall AUC (each tagged with the adversary
    that evades it best), a per-detector timing table (install/train/eval cost
    from the most recent run), adversaries ranked by evasion -- mean AUC across
    detectors, lowest first -- (each tagged with the detector that catches it
    best), and the full detector x adversary AUC matrix linking the two. When
    ``loo_link`` is given, a closing section points to the leave-one-out
    generalization report at that (relative) path.
    """
    detectors = [r["name"] for r in results]
    lines = ["# Leaderboard", ""]

    # Detectors: which detector is best overall, and its single toughest adversary.
    lines += [
        "## Detectors — ranked by overall AUC",
        "",
        "| Rank | Detector | AUC | Toughest adversary (AUC) | # benign | # malign |",
        "| ---: | :--- | ---: | :--- | ---: | ---: |",
    ]
    for i, r in enumerate(sorted(results, key=lambda r: r["auc"], reverse=True), 1):
        row = {a: matrix[(r["name"], a)] for a in adversaries
               if matrix[(r["name"], a)] is not None}
        tough = min(row, key=row.get) if row else None
        tough_str = f"{tough} ({row[tough]:.3f})" if tough is not None else "—"
        lines.append(
            f"| {i} | {r['name']} | {r['auc']:.4f} | {tough_str} | "
            f"{r['n_benign']} | {r['n_malign']} |"
        )

    # Timing/cost for each detector (from the most recent run's timing.json).
    # ``train`` and ``load`` are mutually exclusive per run, so one is ``—``.
    def _t(timing, key, scale=1.0, fmt="{:.1f}"):
        v = timing.get(key)
        return fmt.format(v * scale) if isinstance(v, (int, float)) else "—"

    lines += [
        "",
        "## Detectors — timing (most recent run)",
        "",
        "| Detector | Install (s) | Train (s) | Load (s) | Eval (ms/shape) | # eval |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda r: r["auc"], reverse=True):
        t = r.get("timing", {})
        n_eval = t.get("n_eval")
        n_eval_str = str(n_eval) if isinstance(n_eval, (int, float)) else "—"
        lines.append(
            f"| {r['name']} | {_t(t, 'install_seconds')} | {_t(t, 'train_seconds')} | "
            f"{_t(t, 'load_seconds')} | {_t(t, 'eval_seconds_mean', 1e3)} | {n_eval_str} |"
        )

    if adversaries:
        adv_mean = {a: _mean([matrix[(m, a)] for m in detectors]) for a in adversaries}
        ranked_adv = sorted(adversaries, key=lambda a: adv_mean[a])

        # Adversaries: which adversary evades best, and its toughest detector.
        lines += [
            "",
            "## Adversaries — ranked by evasion (mean AUC across detectors, lowest first)",
            "",
            "| Rank | Adversary | Mean AUC | Best detector (AUC) | # malign |",
            "| ---: | :--- | ---: | :--- | ---: |",
        ]
        for i, a in enumerate(ranked_adv, 1):
            col = {m: matrix[(m, a)] for m in detectors if matrix[(m, a)] is not None}
            best = max(col, key=col.get) if col else None
            best_str = f"{best} ({col[best]:.3f})" if best is not None else "—"
            lines.append(
                f"| {i} | {a} | {adv_mean[a]:.4f} | {best_str} | {counts.get(a, 0)} |"
            )

        # The grid linking both directions (rows ordered like the adversary table).
        lines += [
            "",
            "## Detector × adversary AUC matrix",
            "",
            "| Adversary \\ Detector | " + " | ".join(detectors) + " | Mean |",
            "| :--- | " + " | ".join(["---:"] * (len(detectors) + 1)) + " |",
        ]
        for a in ranked_adv:
            cells = [matrix[(m, a)] for m in detectors]
            cell_strs = [f"{v:.3f}" if v is not None else "—" for v in cells]
            lines.append(f"| {a} | " + " | ".join(cell_strs) + f" | {_mean(cells):.3f} |")
        detector_means = [_mean([matrix[(m, a)] for a in adversaries]) for m in detectors]
        lines.append("| **Mean** | " + " | ".join(f"{v:.3f}" for v in detector_means)
                     + f" | {_mean(detector_means):.3f} |")

    if loo_link:
        lines += [
            "",
            "## Leave-one-out generalization",
            "",
            f"How well do these detectors catch an adversary they were **not** "
            f"trained on? See [`{loo_link}`]({loo_link}): for each adversary, every "
            f"detector is retrained on all the *other* adversaries and scored only "
            f"on the held-out one. The gap from the matrix above measures how much "
            f"each detector relies on having seen an adversary during training.",
        ]

    text = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "leaderboard.md")
    with open(path, "w") as f:
        f.write(text)
    return text, path


# ---------------------------------------------------------------------------
# Leave-one-out (LOO) generalization report.
#
# ``evaluate.py --loo`` writes, per held-out adversary A, a normal by_detector tree
# under ``<output_dir>/loo/<A>/by_detector/<detector>/`` whose eval split is *all*
# benign + only A's malign (the detector having been trained on every adversary
# but A). So each detector's overall AUC in that fold IS its AUC at catching the
# unseen adversary A -- exactly comparable to the standard matrix cell (detector,
# A), where the detector had seen A during training. The difference is the
# generalization gap.
# ---------------------------------------------------------------------------
def collect_loo(loo_dir):
    """``{held-out adversary: [detector result dicts]}`` for each fold under ``loo_dir``.

    Each fold is read with the same ``collect`` used for the standard run, so a
    detector that errored a whole fold (no ``outputs.csv``) is simply absent.
    """
    out = {}
    if not os.path.isdir(loo_dir):
        return out
    for adv in sorted(os.listdir(loo_dir)):
        by_detector = os.path.join(loo_dir, adv, "by_detector")
        if not os.path.isdir(by_detector):
            continue
        res = collect(by_detector)
        if res:
            out[adv] = res
    return out


def write_loo_leaderboard(detectors, held_out, loo_matrix, counts, std_matrix,
                          loo_dir):
    """Write ``loo_dir/leaderboard_loo.md`` and return its ``(text, path)``.

    ``detectors`` is ordered best-generalizer-first and ``held_out``
    strongest-evader-first (set by the caller). ``loo_matrix[(detector, adv)]`` is
    the held-out AUC (or ``None``); ``std_matrix`` is the standard run's grid (or
    ``None``) used only for the generalization-gap section.
    """
    det_mean = {m: _mean([loo_matrix.get((m, a)) for a in held_out]) for m in detectors}
    adv_mean = {a: _mean([loo_matrix.get((m, a)) for m in detectors]) for a in held_out}

    lines = [
        "# Leave-one-out leaderboard",
        "",
        "Each detector is trained on every adversary **but one** and scored only on "
        "the held-out adversary (against all benign shapes), so this measures "
        "generalization to an *unseen* disguise rather than recognition of a "
        "familiar one. Higher AUC = better detection of the unseen adversary.",
        "",
        "## Detectors — ranked by mean held-out AUC (generalization)",
        "",
        "| Rank | Detector | Mean held-out AUC | Worst held-out adversary (AUC) |",
        "| ---: | :--- | ---: | :--- |",
    ]
    for i, m in enumerate(sorted(detectors, key=lambda m: det_mean[m], reverse=True), 1):
        row = {a: loo_matrix[(m, a)] for a in held_out if loo_matrix.get((m, a)) is not None}
        worst = min(row, key=row.get) if row else None
        worst_str = f"{worst} ({row[worst]:.3f})" if worst is not None else "—"
        lines.append(f"| {i} | {m} | {det_mean[m]:.4f} | {worst_str} |")

    lines += [
        "",
        "## Held-out adversaries — ranked by evasion under LOO (mean AUC, lowest first)",
        "",
        "| Rank | Held-out adversary | Mean AUC | Best detector (AUC) | # malign |",
        "| ---: | :--- | ---: | :--- | ---: |",
    ]
    for i, a in enumerate(sorted(held_out, key=lambda a: adv_mean[a]), 1):
        col = {m: loo_matrix[(m, a)] for m in detectors if loo_matrix.get((m, a)) is not None}
        best = max(col, key=col.get) if col else None
        best_str = f"{best} ({col[best]:.3f})" if best is not None else "—"
        lines.append(f"| {i} | {a} | {adv_mean[a]:.4f} | {best_str} | {counts.get(a, 0)} |")

    lines += [
        "",
        "## Held-out adversary × detector AUC matrix",
        "",
        "| Held-out adversary \\ Detector | " + " | ".join(detectors) + " | Mean |",
        "| :--- | " + " | ".join(["---:"] * (len(detectors) + 1)) + " |",
    ]
    for a in held_out:
        cells = [loo_matrix.get((m, a)) for m in detectors]
        cell_strs = [f"{v:.3f}" if v is not None else "—" for v in cells]
        lines.append(f"| {a} | " + " | ".join(cell_strs) + f" | {_mean(cells):.3f} |")
    det_means = [_mean([loo_matrix.get((m, a)) for a in held_out]) for m in detectors]
    lines.append("| **Mean** | " + " | ".join(f"{v:.3f}" for v in det_means)
                 + f" | {_mean(det_means):.3f} |")

    # Generalization gap: standard AUC (adversary seen in training) minus LOO AUC
    # (held out). Positive => the detector leaned on having seen this adversary.
    if std_matrix is not None:
        gap = {(m, a): (std_matrix.get((m, a)) - loo_matrix.get((m, a)))
               for m in detectors for a in held_out
               if std_matrix.get((m, a)) is not None and loo_matrix.get((m, a)) is not None}
        if gap:
            lines += [
                "",
                "## Generalization gap (standard AUC − held-out AUC)",
                "",
                "Drop in detection when the adversary is held out of training. "
                "Larger (more positive) = the detector relied more on having seen "
                "that adversary; ~0 = it generalizes; negative = it did *better* "
                "unseen (usually noise on an already-easy adversary).",
                "",
                "| Held-out adversary \\ Detector | " + " | ".join(detectors) + " | Mean |",
                "| :--- | " + " | ".join(["---:"] * (len(detectors) + 1)) + " |",
            ]
            for a in held_out:
                cells = [gap.get((m, a)) for m in detectors]
                cell_strs = [f"{v:+.3f}" if v is not None else "—" for v in cells]
                lines.append(f"| {a} | " + " | ".join(cell_strs) + f" | {_mean(cells):+.3f} |")
            det_gap_means = [_mean([gap.get((m, a)) for a in held_out]) for m in detectors]
            lines.append("| **Mean** | " + " | ".join(f"{v:+.3f}" for v in det_gap_means)
                         + f" | {_mean(det_gap_means):+.3f} |")

    text = "\n".join(lines) + "\n"
    path = os.path.join(loo_dir, "leaderboard_loo.md")
    with open(path, "w") as f:
        f.write(text)
    return text, path


def build_loo_report(loo_dir, std_matrix, output_dir):
    """Build every LOO artifact under ``loo_dir`` (leaderboard, heatmap, per-fold
    ROC overlays), reusing the standard renderers. Returns the leaderboard path,
    or ``None`` when there are no LOO outputs yet.
    """
    loo_by_adv = collect_loo(loo_dir)
    if not loo_by_adv:
        return None
    held_out = sorted(loo_by_adv)
    detectors = sorted({r["name"] for res in loo_by_adv.values() for r in res})

    # Each fold's eval set is all-benign + only-the-held-out-adversary malign, so a
    # detector's overall AUC there is its held-out-adversary AUC.
    loo_matrix, counts = {}, {}
    for adv, res in loo_by_adv.items():
        for r in res:
            loo_matrix[(r["name"], adv)] = r["auc"]
            counts[adv] = max(counts.get(adv, 0), r["n_malign"])
    for m in detectors:
        for adv in held_out:
            loo_matrix.setdefault((m, adv), None)

    # Per-fold ROC overlay (one curve per detector) -- write_per_adversary filters
    # a fold's malign by the held-out adversary, which is all of it. Rebuilt from
    # scratch so stale folds never linger.
    by_adversary_dir = os.path.join(loo_dir, "by_adversary")
    shutil.rmtree(by_adversary_dir, ignore_errors=True)
    for adv, res in loo_by_adv.items():
        write_per_adversary(adv, res, by_adversary_dir)

    # Heatmap: detectors best-generalizer-first, adversaries strongest-evader-first.
    det_mean = {m: _mean([loo_matrix.get((m, a)) for a in held_out]) for m in detectors}
    adv_mean = {a: _mean([loo_matrix.get((m, a)) for m in detectors]) for a in held_out}
    det_order = sorted(detectors, key=lambda m: det_mean[m], reverse=True)
    adv_order = sorted(held_out, key=lambda a: adv_mean[a])
    write_matrix_heatmap(
        [{"name": m} for m in det_order], adv_order, loo_matrix, output_dir,
        path=os.path.join(loo_dir, "heatmap_loo.png"),
        title="LOO: detector × held-out adversary AUC (trained without that adversary)")

    _, path = write_loo_leaderboard(det_order, adv_order, loo_matrix, counts,
                                    std_matrix, loo_dir)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Top-level output directory (default: output)")
    args = parser.parse_args()

    by_detector_dir = os.path.join(args.output_dir, "by_detector")
    by_adversary_dir = os.path.join(args.output_dir, "by_adversary")

    results = collect(by_detector_dir)
    if not results:
        print(f"No detector outputs found in {by_detector_dir}/. Run evaluate.py first.")
        return

    adversaries, matrix, counts = challenge_matrix(results)

    # Per-detector view (lands beside each detector's outputs.csv).
    for r in results:
        write_per_detector(r, by_detector_dir, adversaries)

    written = [write_composite(results, args.output_dir)]  # roc_all.png

    # Per-adversary view. It is fully derived from the scores, so rebuild it from
    # scratch -- otherwise folders for adversaries that no longer exist linger.
    if adversaries:
        shutil.rmtree(by_adversary_dir, ignore_errors=True)
        for a in adversaries:
            write_per_adversary(a, results, by_adversary_dir)
        written.append(write_matrix_heatmap(results, adversaries, matrix, args.output_dir))
    else:
        print("[note] no adversary tags parsed from malign filenames; "
              "by_adversary/ and heatmap skipped.")

    # Leave-one-out generalization report (only if `evaluate.py --loo` has run).
    loo_dir = os.path.join(args.output_dir, "loo")
    loo_path = build_loo_report(loo_dir, matrix if adversaries else None, args.output_dir)
    loo_link = os.path.relpath(loo_path, args.output_dir) if loo_path else None
    if loo_path:
        written += [os.path.join(loo_dir, "heatmap_loo.png"), loo_path]

    text, leaderboard_path = write_leaderboard(
        results, adversaries, matrix, counts, args.output_dir, loo_link=loo_link)
    written.append(leaderboard_path)

    print(text)
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    print(f"  {by_detector_dir}/<detector>/  (roc.png, metrics.txt)")
    if adversaries:
        print(f"  {by_adversary_dir}/<adversary>/  (roc.png, metrics.csv, metrics.txt)")
    if loo_path:
        print(f"  {loo_dir}/by_adversary/<adversary>/  (roc.png, metrics.csv, metrics.txt)")


if __name__ == "__main__":
    main()
