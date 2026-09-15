"""Tests for source readiness reconstruction and single-pair selection."""

from pathlib import Path
import sqlite3

import pytest

from closeloop_analyzer.target_selection import (
    KernelRecord,
    TraceSelectionError,
    TraceTargetSelector,
)


def make_trace(path):
    """Create a minimal Nsight-shaped two-client trace."""
    with sqlite3.connect(str(path)) as db:
        db.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
          start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
          correlationId INTEGER, nameId INTEGER, returnValue INTEGER,
          callchainId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          greenContextId INTEGER, streamId INTEGER, correlationId INTEGER,
          globalPid INTEGER, demangledName INTEGER, shortName INTEGER,
          mangledName INTEGER, launchType INTEGER, cacheConfig INTEGER,
          registersPerThread INTEGER, gridX INTEGER, gridY INTEGER,
          gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
          staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
          localMemoryPerThread INTEGER, localMemoryTotal INTEGER,
          gridId INTEGER, sharedMemoryExecuted INTEGER, graphNodeId INTEGER,
          sharedMemoryLimitConfig INTEGER);
        CREATE TABLE TARGET_INFO_CUDA_STREAM(
          streamId INTEGER, hwId INTEGER, vmId INTEGER, processId INTEGER,
          contextId INTEGER, priority INTEGER, flag INTEGER);
        """)
        db.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [
                (1, "cudaLaunchKernel_v7000"),
                (2, "aggressor"),
                (3, "victim_a"),
                (4, "victim_b"),
            ],
        )
        for pid in (10, 20):
            db.execute(
                "INSERT INTO TARGET_INFO_CUDA_STREAM VALUES "
                "(7,0,0,?,1,0,0)",
                (pid,),
            )
        global_a = 10 << 24
        global_v = 20 << 24
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
            "(0,10,0,?,1,1,0,NULL)",
            (global_a,),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
            "(20,30,0,?,2,1,0,NULL)",
            (global_v,),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
            "(31,32,0,?,3,1,0,NULL)",
            (global_v,),
        )

        def kernel(start, end, gid, correlation, name, grid):
            values = [
                start,
                end,
                0,
                1,
                0,
                7,
                correlation,
                gid,
                name,
                name,
                None,
                1,
                0,
                16,
                grid,
                1,
                1,
                32,
                1,
                1,
                0,
                0,
                0,
                0,
                correlation,
                0,
                None,
                0,
            ]
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                values,
            )

        kernel(100, 2_000_100, global_a, 1, 2, 1)
        kernel(1_000_000, 1_000_100, global_v, 2, 3, 1)
        kernel(1_000_100, 1_000_200, global_v, 3, 4, 1)


def test_selects_one_aggregate_pair_and_preserves_victim_dag(tmp_path):
    """The selected victim signature includes ordered kernels and edges."""
    trace = tmp_path / "profile.sqlite"
    make_trace(trace)
    result = TraceTargetSelector(trace, tmp_path).select()
    assert result["selection_policy"]["selected_case_count"] == 1
    assert result["schema"] == "target_selection_v2"
    assert result["selection_policy"]["victim_set"] == (
        "gpu_execution_overlap_all_streams"
    )
    assert result["selection_policy"]["coverage_calculated"] is False
    episode = result["source_episode"]
    assert [item["name"] for item in episode["victim_kernels"]] == [
        "victim_a",
        "victim_b",
    ]
    assert episode["victim_dependencies"] == [
        {
            "source_kernel_id": episode["victim_kernels"][0][
                "source_kernel_id"
            ],
            "target_kernel_id": episode["victim_kernels"][1][
                "source_kernel_id"
            ],
            "kind": "same_stream",
        }
    ]
    assert result["selected_pair"]["occurrence_count"] == 1
    assert all(
        item["overlap_duration_ns"] > 0
        for item in episode["victim_kernels"]
    )


def kernel(kernel_id, stream, start, end, launch_start, context=1,
           graph=None, api="cuLaunchKernel"):
    """Create one compact victim record for overlap-policy unit tests."""
    return KernelRecord(
        kernel_id, 20, context, stream, 0, kernel_id, f"k{kernel_id}",
        start, end, launch_start, launch_start + 1, api,
        (1, 1, 1), (32, 1, 1), 16, 0, 0, graph, 1,
    )


def aggressor():
    """Create a one-millisecond cross-client aggressor."""
    return KernelRecord(
        99, 10, 1, 1, 0, 99, "aggressor", 100, 200, 80, 81,
        "cuLaunchKernel", (1, 1, 1), (32, 1, 1), 16, 0, 0, None, 1,
    )


def overlap_set(tmp_path, victims, unresolved=None):
    """Invoke overlap construction without requiring a full SQLite trace."""
    selector = TraceTargetSelector(tmp_path / "unused.sqlite", tmp_path)
    streams = selector._stream_sequences(victims)
    return selector._overlap_set(
        victims[0], aggressor(), victims, streams,
        {20: unresolved or []}, [],
    )


def test_strict_half_open_boundaries_and_all_stream_collection(tmp_path):
    """Touching boundaries do not overlap; every overlapping stream does."""
    assert TraceTargetSelector._strictly_overlaps(100, 200, 199, 201)
    assert not TraceTargetSelector._strictly_overlaps(100, 200, 200, 201)
    assert not TraceTargetSelector._strictly_overlaps(100, 200, 99, 100)
    victims = [
        kernel(1, 7, 90, 120, 85),
        kernel(2, 8, 150, 210, 90),
        kernel(3, 7, 200, 220, 220),
    ]
    selected = overlap_set(tmp_path, victims)
    assert [item.kernel_id for item in selected.kernels] == [1, 2]
    assert {item.stream_id for item in selected.kernels} == {7, 8}


@pytest.mark.parametrize("failure", ["post_checkpoint", "event", "graph"])
def test_rejects_non_self_contained_overlap_sets(tmp_path, failure):
    """Tail producers, unresolved events, and graph launches are rejected."""
    victims = [
        kernel(1, 7, 110, 150, 85),
        kernel(2, 8, 120, 160, 95),
    ]
    unresolved = []
    if failure == "post_checkpoint":
        victims.append(kernel(3, 9, 210, 220, 90))
    elif failure == "event":
        unresolved = [90]
    else:
        victims[0] = kernel(1, 7, 110, 150, 85, graph=4)
    with pytest.raises(TraceSelectionError):
        overlap_set(tmp_path, victims, unresolved)


def test_rejects_overlapping_victim_contexts(tmp_path):
    """One candidate never mixes or silently drops CUDA contexts."""
    victims = [
        kernel(1, 7, 110, 150, 85, context=1),
        kernel(2, 8, 120, 160, 90, context=2),
    ]
    with pytest.raises(TraceSelectionError, match="multiple.*contexts"):
        overlap_set(tmp_path, victims)


def test_live_matcher_prefers_exact_mangled_symbol():
    """Nsight display-name abbreviations never weaken symbol matching."""
    kernel = KernelRecord(
        1, 10, 1, 7, 0, 2,
        "void cutlass::Kernel2<gemm>(T1::Params)",
        100, 200, 90, 95, "cuLaunchKernel",
        (1, 2, 3), (128, 1, 1), 32, 0, 49152, None, 1,
        "_ZN7cutlass7Kernel2I4gemmEEvNT_6ParamsE",
    )
    matcher = TraceTargetSelector._launch_matcher(kernel)
    assert matcher["symbol"].endswith("(T1::Params)")
    assert matcher["mangled_symbol"] == (
        "_ZN7cutlass7Kernel2I4gemmEEvNT_6ParamsE"
    )


def test_5s_v5_is_overlap_complete_or_fails_without_closure_fallback():
    """The authoritative trace either qualifies strictly or stops cleanly."""
    root = (
        Path(__file__).parents[3] / "outputs" / ("tt-mps-detr-deeplab-5s-v5")
    )
    if not (root / "profile.sqlite").is_file():
        pytest.skip("5s-v5 trace is not installed")
    try:
        result = TraceTargetSelector(root / "profile.sqlite", root).select()
    except TraceSelectionError as error:
        assert str(error) == "no_self_contained_overlap_episode"
        return
    assert result["selection_policy"]["selected_case_count"] == 1
    assert result["selected_pair"]["occurrence_count"] >= 1
    assert result["source_frame"]["schema"] in {
        "source_frame_identity_v1", "dataset_input_identity_v2",
    }
    episode = result["source_episode"]
    aggressor_interval = episode["aggressor_source_interval"]
    assert all(
        launch["start_ns"] < aggressor_interval["end_ns"]
        and launch["end_ns"] > aggressor_interval["start_ns"]
        and launch["overlap_duration_ns"] > 0
        for launch in episode["victim_kernels"]
    )
