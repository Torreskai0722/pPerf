"""Check additive variance and tail accounting on differently ordered components."""

import numpy as np

from closeloop_analyzer.input_data.dino_diagnosis import component_stats, kernel_gaps


def test_component_accounting_with_opposing_variation():
    latency = np.array([12., 7., 15., 9., 11.])
    kernel = np.array([1., 6., 2., 3., 4.])
    left = component_stats(kernel, latency)
    right = component_stats(latency-kernel, latency)
    assert left["covariance_share"] < 0
    assert np.isclose(left["covariance_share"] + right["covariance_share"], 1.)
    gap = np.diff(np.percentile(latency, [50, 99], method="linear"))[0]
    assert np.isclose(left["total_latency_endpoint_gap_ms"] +
                      right["total_latency_endpoint_gap_ms"], gap)


def test_gpu_gap_partition_merges_overlapping_activity():
    assert kernel_gaps(0, 20, [(2, 5), (4, 8), (10, 15)]) == [(0, 2), (8, 10), (15, 20)]
    assert kernel_gaps(0, 20, []) == [(0, 20)]
