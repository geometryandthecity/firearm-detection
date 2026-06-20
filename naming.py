"""The malign-shape filename convention, shared by the evaluator and the report.

``load_data.py`` names every malign variant ``<stem>_<adversary>_<idx>.obj`` (e.g.
``device_03_tetwild_0007.obj``): the base-shape stem, the single adversary that
produced it, then a zero-padded counter. This module is the one place that turns
such a filename back into its adversary, so the leave-one-out
evaluator (``evaluate.py --loo``) and the report (``generate_report.py``) agree on
which adversary owns a shape. It is deliberately dependency-free (no matplotlib / torch)
so it can be imported anywhere, including inside spawned training/eval workers.
"""


def adversary_of(filename):
    """Return the adversary name from a malign filename, or ``None``.

    ``<stem>_<adversary>_<idx>.obj`` -> ``<adversary>``: the adversary is the
    ``_``-delimited field just before the trailing numeric index, so it is found
    regardless of how many underscores the base stem itself contains. Benign files
    (``t10k_<id>.obj`` / ``abc_<stem>.obj``) and anything not matching this shape
    return ``None``, so they stay shared negatives rather than becoming adversaries.
    """
    stem = filename[:-4] if filename.lower().endswith(".obj") else filename
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return parts[-2]
    return None
