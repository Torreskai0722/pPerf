"""One P1–P99 sample policy for plots and statistical analysis."""

import numpy as np


POLICY = "p1-p99-linear-per-execution-model-v1"


def p1_p99(values):
    """Return an inclusive mask and the original, unfiltered sample cutoffs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("P1–P99 filtering requires a nonempty finite sample")
    lower, upper = map(float, np.percentile(values, [1, 99], method="linear"))
    mask = (values >= lower) & (values <= upper)
    if not mask.any():
        raise ValueError("insufficient evidence: no observations inside P1–P99")
    return mask, {"policy": POLICY, "original_count": len(values),
                  "retained_count": int(mask.sum()), "cutoff_P1_ms": lower,
                  "cutoff_P99_ms": upper, "excluded_below_P1": int((values < lower).sum()),
                  "excluded_above_P99": int((values > upper).sum())}


def filter_frames(frames):
    """Filter one execution/model once; keep paired components on these frames."""
    mask, audit = p1_p99([r["latency_ms"] for r in frames])
    selected = [r for r, keep in zip(frames, mask) if keep]
    audit["retained_unique_source_count"] = len({r["source_frame_id"] for r in selected})
    audit["excluded_frames"] = [
        {k: r[k] for k in ("input_id", "source_frame_id", "latency_ms") if k in r}
        for r, keep in zip(frames, mask) if not keep]
    return selected, audit
