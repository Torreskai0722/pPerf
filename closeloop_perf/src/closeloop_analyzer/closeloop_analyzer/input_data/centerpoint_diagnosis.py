"""Offline module/kernel associations for the eight isolated CenterPoint runs.

Run with ``python -m closeloop_analyzer.input_data.centerpoint_diagnosis ROOT``.
No additional profiling is performed. Correlations are descriptive, not causal.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path

import numpy as np

from .._common import merge_intervals
from .crossed_evidence import read_json, read_jsonl, sha256, write_csv, write_json
from ..sample_filter import POLICY, filter_frames


def correlation(x, y):
    """Pearson association; a constant series has no defined correlation."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def endpoint_gap(component, latency):
    """Component change at the total latency's linear P50/P99 endpoints.

    Components use the same frame ordering and interpolation weights, so their
    contributions add to total P99-P50. This is not the component's own P99-P50.
    """
    values = np.asarray(component)[np.argsort(latency, kind="stable")]
    positions = np.array([0.50, 0.99]) * (len(values) - 1)
    endpoints = np.interp(positions, np.arange(len(values)), values)
    return float(endpoints[1] - endpoints[0])


def diagnose(root):
    root = Path(root)
    study = root / "analysis/input2-crossed/input-data"
    destination = study / "centerpoint_diagnosis"
    summaries = [r for r in read_json(study / "per_run_summaries.json")
                 if r["phase"] == "isolated" and r["model_id"] == "centerpoint"]
    if len(summaries) != 8:
        raise ValueError("Expected eight isolated CenterPoint executions")
    all_frames, associations, runs, matched, evidence = [], [], [], [], {}
    for summary in summaries:
        run_id = summary["execution_id"]
        folder = root / "analysis/runs" / run_id / "input-data"
        raw = root / "runs" / run_id
        frames = read_json(folder / "frames.json")
        frames.sort(key=lambda r: r["inference_start_ns"])
        if (len(frames) != summary["completed_count"] or
                len({r["source_frame_id"] for r in frames}) != summary["unique_source_count"]):
            raise ValueError("Frame coverage differs from the accepted run summary")
        frames, sample_audit = filter_frames(frames)
        inputs = {str(r["input_id"]): r for r in read_jsonl(raw / "model_centerpoint_inputs.jsonl")}
        by_input = {r["input_id"]: r for r in frames}
        if len(by_input) != len(frames):
            raise ValueError("Duplicate completed input identity")
        modules, kernels, intervals = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
        streams, excluded = set(), 0
        with gzip.open(folder / "kernels.csv.gz", "rt") as source:
            for kernel in csv.DictReader(source):
                frame = by_input.get(kernel["input_id"])
                if frame is None:
                    continue  # Warmups and unresolved inputs remain in the original export.
                if kernel["source_frame_id"] != frame["source_frame_id"]:
                    raise ValueError("Kernel/source identity mismatch")
                launch = int(kernel["launch_start_ns"])
                if not frame["inference_start_ns"] <= launch < frame["inference_end_ns"]:
                    excluded += 1  # Separate pre-inference preprocessing boundary.
                    continue
                start, end = int(kernel["start"]), int(kernel["end"])
                if not frame["inference_start_ns"] <= start <= end <= frame["inference_end_ns"]:
                    raise ValueError("Kernel exceeds completed inference boundary")
                duration = (end - start) / 1e6
                modules[frame["input_id"]][kernel["module"] or "unresolved"] += duration
                kernels[frame["input_id"]][kernel["kernel_name"]] += duration
                intervals[frame["input_id"]].append((start, end))
                streams.add((kernel["pid"], kernel["contextId"], kernel["streamId"]))
        module_names = sorted({k for v in modules.values() for k in v})
        kernel_names = sorted({k for v in kernels.values() for k in v})
        rows = []
        for frame in frames:
            key = frame["input_id"]
            summed = sum(modules[key].values())
            union = sum(b - a for a, b in merge_intervals(intervals[key])) / 1e6
            # Additive accounting is valid only when recorded kernels do not overlap.
            if abs(summed - union) > 1e-6:
                raise ValueError("Overlapping kernels require non-additive accounting")
            row = dict(frame, mps_enabled=summary["mps_enabled"],
                       input_point_count=inputs[key]["input_point_count"],
                       gpu_ms=summed, nonkernel_ms=frame["latency_ms"] - union)
            row.update({"module:" + m: modules[key][m] for m in module_names})
            for name in ("point_to_voxelidx_kernel", "determin_voxel_num"):
                row["kernel:" + name] = sum(v for k, v in kernels[key].items()
                                           if k.startswith("void " + name + "<"))
            rows.append(row)
        latency = np.array([r["latency_ms"] for r in rows])
        identity = {k: summary[k] for k in ("execution_id", "scene_id", "mps_enabled")}
        components = {"module:" + m: [r["module:" + m] for r in rows] for m in module_names}
        components.update({k: [r[k] for r in rows] for k in
                           ("nonkernel_ms", "input_point_count", "preprocess_ms", "decode_ms")})
        components.update({"kernel:" + k: [kernels[r["input_id"]][k] for r in rows]
                           for k in kernel_names})
        for component, values in components.items():
            values = np.asarray(values)
            additive = component.startswith(("module:", "kernel:")) or component == "nonkernel_ms"
            associations.append(dict(identity, component=component, count=len(rows),
                units="points" if component == "input_point_count" else "ms",
                mean=float(np.mean(values)), std=float(np.std(values)),
                pearson=correlation(values, latency),
                first_difference_pearson=correlation(np.diff(values), np.diff(latency)),
                covariance_share=float(np.mean((values-values.mean()) * (latency-latency.mean())) / np.var(latency)) if additive else None,
                total_latency_endpoint_gap=endpoint_gap(values, latency) if additive else None))
        gap_sum = sum(endpoint_gap(components[k], latency) for k in components
                      if k.startswith("module:") or k == "nonkernel_ms")
        gap = float(np.diff(np.percentile(latency, [50, 99], method="linear"))[0])
        if not np.isclose(gap_sum, gap):
            raise ValueError("Tail-gap components do not sum to total")
        runs.append(dict(identity, completed_count=len(rows),
                         sample_filter=sample_audit,
                         unique_source_count=len({r["source_frame_id"] for r in rows}),
                         P50_ms=float(np.percentile(latency, 50, method="linear")),
                         P99_minus_P50_ms=gap, streams=sorted(streams),
                         pre_inference_kernels_excluded=excluded))
        all_frames.extend(rows)
        for path in (folder / "frames.json", folder / "kernels.csv.gz",
                     raw / "model_centerpoint_inputs.jsonl", raw / "config.yaml"):
            evidence[str(path)] = sha256(path)
        write_csv(root / "analysis/runs" / run_id / "input-data/centerpoint_diagnosis/frames.csv", rows)
    for scene in sorted({r["scene_id"] for r in all_frames}):
        off = {r["source_frame_id"]: r for r in all_frames if r["scene_id"] == scene and not r["mps_enabled"]}
        on = {r["source_frame_id"]: r for r in all_frames if r["scene_id"] == scene and r["mps_enabled"]}
        common = sorted(off.keys() & on.keys())
        for key in common:
            row = dict(scene_id=scene, source_frame_id=key)
            row.update({"delta:" + k: on[key][k] - off[key][k] for k in off[key]
                        if k.startswith("module:") or k in ("latency_ms", "nonkernel_ms")})
            matched.append(row)
    write_csv(destination / "frames.csv", all_frames)
    write_csv(destination / "associations.csv", associations)
    write_csv(destination / "matched_frame_deltas.csv", matched)
    write_json(destination / "summary.json", {
        "scope": "Eight isolated CenterPoint executions; completed non-warmup inference through CUDA completion",
        "sample_filter_policy": POLICY,
        "runs": runs, "sources_sha256": evidence, "analyzer_sha256": sha256(__file__),
        "limitations": ["One execution per condition; temporally dependent frames; sparse-tail P99",
                        "Module/kernel GPU time is a component of latency: correlation is not causal proof",
                        "Nonkernel time includes copies, host work, launch gaps and synchronization; not identified as CPU time",
                        "Existing top-level annotations do not identify individual neural-network layers",
                        "Endpoint contributions interpolate the same total-latency-ranked frames; they are not independent component percentiles"],
    })
    return destination


def report(destination):
    """Retain a readable diagnosis and a plot alongside the numeric evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(destination)
    frames = list(csv.DictReader((destination / "frames.csv").open()))
    associations = list(csv.DictReader((destination / "associations.csv").open()))
    matched = list(csv.DictReader((destination / "matched_frame_deltas.csv").open()))
    geometry = {r["source_frame_id"]: r for r in csv.DictReader((destination / "input_geometry.csv").open())}
    runs = read_json(destination / "summary.json")["runs"]
    geometry_rows, comparisons = [], []
    lines = ["# CenterPoint isolated inference diagnosis", "",
             "The strongest measured module association in every scene and both MPS modes is GPU hard voxelization inside `data_preprocessor`. This is inside the CUDA-completed inference boundary; the separately logged CPU preprocessing and message decoding are outside it.", "",
             "All plots and metrics use each execution's P1–P99-filtered inference frames. Cutoffs and exclusions are recorded in summary.json; P50/P99 are recomputed on retained frames. Components share the same frame mask. GPU module durations use existing correlated launch attribution. Included kernels lie inside their inference boundaries and do not overlap, allowing additive accounting. Difference correlations use successive retained observations.", "",
             "| Scene | MPS | Frames | P50 ms | P99-P50 ms | Voxelization mean ms | Pearson r | Consecutive-difference r |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        key = run["execution_id"]
        row = next(r for r in associations if r["execution_id"] == key and r["component"] == "module:data_preprocessor")
        lines.append(f"| {run['scene_id']} | {'on' if run['mps_enabled'] else 'off'} | {run['completed_count']} | {run['P50_ms']:.3f} | {run['P99_minus_P50_ms']:.3f} | {float(row['mean']):.3f} | {float(row['pearson']):.3f} | {float(row['first_difference_pearson']):.3f} |")
        records = [r for r in frames if r["run_id"] == key]
        for metric in ("raw_points", "voxel_input_points", "occupied_voxels", "point_scan_iterations"):
            x = np.array([float(geometry[r["source_frame_id"]][metric]) for r in records])
            for timing in ("latency_ms", "module:data_preprocessor", "kernel:point_to_voxelidx_kernel", "kernel:determin_voxel_num"):
                y = np.array([float(r[timing]) for r in records])
                geometry_rows.append(dict(execution_id=key, scene_id=run["scene_id"],
                    mps_enabled=run["mps_enabled"], metric=metric, timing=timing,
                    count=len(records), mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                    pearson=correlation(x, y), first_difference_pearson=correlation(np.diff(x), np.diff(y))))
    lines += ["", "## Mechanism and scope", "",
        "The retained configuration uses deterministic hard voxelization. Two measured kernels dominate this module: `point_to_voxelidx_kernel` scans preceding points for matching voxel coordinates, stopping when its point-per-voxel limit is met; `determin_voxel_num` walks the points sequentially and is launched with one block and one thread. Input size, spatial occupancy and source order can therefore affect their work. Source implementation hashes and resolved CPU pipeline settings are retained in `input_geometry_provenance.json`.", "",
        "The actual inferencer pipeline pads nine missing sweeps with copies of the current cloud (removing close points from the added copies), then applies the spatial range filter. `input_geometry.csv` reconstructs this CPU pipeline for every unique timed source frame. These are reconstructed inputs/logical scan iterations, not recorded GPU voxel counts or hardware cycles. Raw message point counts alone are not the voxelizer's workload.", "",
        "| Scene | MPS | Mean voxel input points | Mean occupied voxels | Logical point-scan work vs inference r |",
        "|---|---|---:|---:|---:|"]
    for run in runs:
        records = [r for r in geometry_rows if r["execution_id"] == run["execution_id"] and r["timing"] == "latency_ms"]
        point = next(r for r in records if r["metric"] == "voxel_input_points")
        occupied = next(r for r in records if r["metric"] == "occupied_voxels")
        scan = next(r for r in records if r["metric"] == "point_scan_iterations")
        lines.append(f"| {run['scene_id']} | {'on' if run['mps_enabled'] else 'off'} | {point['mean']:.0f} | {occupied['mean']:.0f} | {scan['pearson']:.3f} |")
    lines += ["", "Input geometry explains substantially more variation in scenes 0398, 0184 and 0245 than in 0770. In 0770, voxelization still tracks inference closely while reconstructed work is relatively stable; input geometry alone does not explain its runtime variation. Kernel timing fluctuations on identical source frames also remain between modes.", "",
        "## Matching actual source frames between modes", "",
        "| Scene | Common retained frames | Filtered-sample gap change ms | Common-frame gap change ms | Correlation of voxelization and total per-frame mode changes |",
        "|---|---:|---:|---:|---:|"]
    scenes = ["scene-0770", "scene-0398", "scene-0184", "scene-0245"]
    for scene in scenes:
        records = [r for r in matched if r["scene_id"] == scene]
        keys = {r["source_frame_id"] for r in records}
        gaps = []
        full_gaps = []
        for mode in (False, True):
            values = [float(r["latency_ms"]) for r in frames if r["scene_id"] == scene
                      and r["mps_enabled"] == str(mode) and r["source_frame_id"] in keys]
            gaps.append(float(np.diff(np.percentile(values, [50, 99], method="linear"))[0]))
            full_gaps.append(next(r["P99_minus_P50_ms"] for r in runs
                                 if r["scene_id"] == scene and r["mps_enabled"] == mode))
        corr = correlation([float(r["delta:module:data_preprocessor"]) for r in records],
                           [float(r["delta:latency_ms"]) for r in records])
        comparisons.append(dict(scene_id=scene, common_frames=len(keys),
            full_gap_change_ms=full_gaps[1]-full_gaps[0], common_gap_change_ms=gaps[1]-gaps[0],
            voxelization_delta_pearson=corr))
        lines.append(f"| {scene} | {len(keys)} | {full_gaps[1]-full_gaps[0]:.3f} | {gaps[1]-gaps[0]:.3f} | {corr:.3f} |")
    lines += ["", "## Accounting for the observed tail-gap change", "",
        "Entries are MPS-on minus MPS-off, in milliseconds. For each run, each component is evaluated on the same interpolated frames defining the total P50 and P99. The three component columns add to the total change, but this is descriptive endpoint accounting, not an intervention or a module's own percentile gap.", "",
        "| Scene | Total gap change | Voxelization contribution change | Other kernels contribution change | Nonkernel contribution change |",
        "|---|---:|---:|---:|---:|"]
    for comparison in comparisons:
        scene = comparison["scene_id"]
        changes = {}
        for component in ("module:data_preprocessor", "nonkernel_ms"):
            values = [next(float(r["total_latency_endpoint_gap"]) for r in associations
                           if r["scene_id"] == scene and r["mps_enabled"] == str(mode)
                           and r["component"] == component) for mode in (False, True)]
            changes[component] = values[1] - values[0]
        total = comparison["full_gap_change_ms"]
        voxel, residual = changes["module:data_preprocessor"], changes["nonkernel_ms"]
        lines.append(f"| {scene} | {total:.3f} | {voxel:.3f} | {total-voxel-residual:.3f} | {residual:.3f} |")
    lines += ["", "Each condition has one execution and temporally dependent observations. All full-sample P99 estimates have fewer than 1,000 observations; common-frame subsets below 100 are especially fragile. Mode differences remain observational because execution order, runtime state and frame timing are not independently replicated. A module's contribution is part of total latency, so high correlation alone is not causal proof.", "",
        "The input-driven voxelization mechanism is supported by implementation inspection, reconstructed workload and recorded kernel timings. The precise cause of occasional stalls and the smaller MPS-versus-disabled tail difference remains unresolved. No finer neural-network layer labels are inferred from the top-level annotations. Resolving the residual mechanism would require targeted repeated timing and launch/memory/scheduler evidence; no additional GPU or layer-detail runs were scheduled.", "",
        "`associations.csv` also reports additive changes at the total latency's P50/P99 endpoint frames. They sum to total P99-P50, but are not each module's independent P99-P50 and can be negative. `nonkernel_ms` includes memory copies, host work, launch gaps and synchronization, not exclusively CPU execution.", "",
        "Reproduce after sourcing ROS and the workspace:", "",
        "```bash", "python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2", "python3 -m closeloop_profiler.centerpoint_input_geometry /mmdetection3d_ros2/analysis_outputs/input2", "python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only", "```"]
    write_csv(destination / "geometry_associations.csv", geometry_rows)
    write_csv(destination / "mode_comparisons.csv", comparisons)
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(9, 6))
    for scene, marker in zip(scenes, ("o", "s", "^", "D")):
        for mode, color in (("False", "#4c78a8"), ("True", "#e68a2e")):
            rows = [r for r in frames if r["scene_id"] == scene and r["mps_enabled"] == mode]
            ax.scatter([float(r["module:data_preprocessor"]) for r in rows],
                       [float(r["latency_ms"]) for r in rows], s=15, alpha=.65,
                       marker=marker, color=color, linewidths=.2, edgecolors="#333333",
                       label=scene[6:] + (" MPS off" if mode == "False" else " MPS on"))
    ax.set(xlabel="GPU voxelization / data_preprocessor time (ms)",
           ylabel="Completed inference time (ms)", title="CenterPoint: voxelization tracks inference variation")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    fig.text(.5, .015, "P1–P99 filtered; one execution per scene/mode. Per-run correlations in the report.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(destination / "voxelization_correlation.png", dpi=180)
    fig.savefig(destination / "voxelization_correlation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    destination = Path(args.artifact_root) / "analysis/input2-crossed/input-data/centerpoint_diagnosis"
    if args.report_only:
        report(destination)
    else:
        diagnose(args.artifact_root)
    print(destination)
