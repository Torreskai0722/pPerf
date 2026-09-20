"""Check additive attribution at the total inference latency percentiles."""

import numpy as np

from closeloop_analyzer.input_data.centerpoint_diagnosis import (
    correlation, endpoint_gap,
)
from closeloop_profiler.centerpoint_input_geometry import scan_work


def test_endpoint_contributions_follow_latency_order_and_sum():
    latency = np.array([12., 7., 15., 9., 11.])
    module = np.array([1., 6., 2., 3., 4.])
    residual = latency - module
    expected = np.diff(np.percentile(latency, [50, 99], method="linear"))[0]
    assert np.isclose(endpoint_gap(module, latency), -2.04)
    assert np.isclose(endpoint_gap(module, latency) + endpoint_gap(residual, latency), expected)
    assert correlation(np.ones(5), latency) is None
    assert np.isclose(correlation(latency, latency), 1.)


def test_scan_work_matches_ordered_cuda_loop():
    for keys in ([1, 1, 1, 2, 1, 2, 3], [3, 2, 1], [1] * 12):
        for cap in (1, 2, 10):
            expected = 0
            for i, key in enumerate(keys):
                matches = 0
                for previous in keys[:i]:
                    expected += 1
                    matches += previous == key
                    if matches > 1 and matches >= cap:
                        break
            actual, unique = scan_work(np.array(keys), cap)
            assert actual == expected
            assert unique == len(set(keys))
