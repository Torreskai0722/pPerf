"""Boundary inclusion, source pairing, and insufficient retained evidence."""

import numpy as np
import pytest

from closeloop_analyzer.sample_filter import filter_frames, p1_p99


def test_boundaries_are_inclusive_and_paired_components_keep_frame_identity():
    values = np.arange(101, dtype=float)
    frames = [dict(input_id=i, source_frame_id=str(i), latency_ms=v, component=100-v)
              for i, v in enumerate(values)]
    retained, audit = filter_frames(frames)
    assert [r["input_id"] for r in retained] == list(range(1, 100))
    assert all(r["latency_ms"] + r["component"] == 100 for r in retained)
    assert [r["input_id"] for r in audit["excluded_frames"]] == [0, 100]
    assert audit["cutoff_P1_ms"] == 1 and audit["cutoff_P99_ms"] == 99
    assert audit["retained_unique_source_count"] == 99
    assert p1_p99([7, 7, 7])[0].all()
    with pytest.raises(ValueError, match="insufficient evidence"):
        p1_p99([1, 2])
    with pytest.raises(ValueError, match="finite"):
        p1_p99([1, np.nan])
