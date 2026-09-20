"""Attribute isolated DINO variation using retained, completed inference traces."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path

import numpy as np

from .._common import intersection_duration, merge_intervals, nvtx_ranges, runtime_events
from .centerpoint_diagnosis import correlation, endpoint_gap
from .crossed_evidence import profile_connection, read_json, sha256, write_csv, write_json


def component_stats(values, latency):
    """Describe association and additive accounting, without significance tests."""
    values, latency = np.asarray(values), np.asarray(latency)
    return dict(mean_ms=float(values.mean()), std_ms=float(values.std()),
                P50_ms=float(np.percentile(values, 50, method="linear")),
                P99_ms=float(np.percentile(values, 99, method="linear")),
                pearson=correlation(values, latency),
                first_difference_pearson=correlation(np.diff(values), np.diff(latency)),
                covariance_share=float(np.mean((values-values.mean()) *
                    (latency-latency.mean())) / np.var(latency)),
                total_latency_endpoint_gap_ms=endpoint_gap(values, latency))


def diagnose(root):
    root = Path(root)
    study = root / "analysis/input2-crossed/input-data"
    destination = study / "dino_diagnosis"
    selected = [r for r in read_json(study / "per_run_summaries.json")
                if r["phase"] == "isolated" and r["model_id"] == "dino"]
    if len(selected) != 8:
        raise ValueError("Expected eight isolated DINO executions")
    all_frames, associations, kernel_stats, runs, evidence = [], [], [], [], {}
    for summary in selected:
        run_id = summary["execution_id"]
        folder = root / "analysis/runs" / run_id / "input-data"
        frames = sorted(read_json(folder / "frames.json"), key=lambda r: r["inference_start_ns"])
        by_input = {r["input_id"]: r for r in frames}
        if len(by_input) != len(frames) or len(frames) != summary["completed_count"]:
            raise ValueError("Completed frame count/identity differs")
        modules, kernels, counts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        intervals, streams, excluded = defaultdict(list), set(), 0
        with gzip.open(folder / "kernels.csv.gz", "rt") as source:
            for kernel in csv.DictReader(source):
                key = kernel["input_id"]
                frame = by_input.get(key)
                if frame is None:
                    continue
                if kernel["source_frame_id"] != frame["source_frame_id"]:
                    raise ValueError("Kernel source-frame attribution differs")
                launch = int(kernel["launch_start_ns"])
                if not frame["inference_start_ns"] <= launch < frame["inference_end_ns"]:
                    excluded += 1
                    continue
                start, end = int(kernel["start"]), int(kernel["end"])
                if not frame["inference_start_ns"] <= start <= end <= frame["inference_end_ns"]:
                    raise ValueError("Kernel exceeds inference completion boundary")
                module = kernel["module"] or "unresolved"
                duration = (end - start) / 1e6
                modules[key][module] += duration
                kernels[key][module, kernel["kernel_name"]] += duration
                counts[key][module, kernel["kernel_name"]] += 1
                intervals[key].append((start, end))
                streams.add((kernel["contextId"], kernel["streamId"]))
        names = sorted({m for v in modules.values() for m in v})
        rows = []
        for frame in frames:
            key = frame["input_id"]
            total = sum(modules[key].values())
            union = sum(b-a for a, b in merge_intervals(intervals[key])) / 1e6
            if abs(total-union) > 1e-6:
                raise ValueError("Overlapping kernels prevent additive decomposition")
            row = dict(frame, mps_enabled=summary["mps_enabled"], gpu_ms=total,
                       nonkernel_ms=frame["latency_ms"]-union,
                       kernel_count=sum(counts[key].values()))
            row.update({"module:"+m: modules[key][m] for m in names})
            rows.append(row)
        latency = np.array([r["latency_ms"] for r in rows])
        identity = {k: summary[k] for k in ("execution_id", "scene_id", "mps_enabled")}
        components = ["module:"+m for m in names] + ["nonkernel_ms"]
        for component in components + ["gpu_ms", "preprocess_ms", "decode_ms"]:
            values = [r[component] for r in rows]
            stats = component_stats(values, latency)
            if component in ("preprocess_ms", "decode_ms"):
                stats["covariance_share"] = stats["total_latency_endpoint_gap_ms"] = None
            associations.append(dict(identity, component=component, count=len(rows), **stats))
        gap = float(np.diff(np.percentile(latency, [50, 99], method="linear"))[0])
        if not np.isclose(gap, sum(endpoint_gap([r[c] for r in rows], latency) for c in components)):
            raise ValueError("Component tail gaps do not reconstruct total")
        for module, name in sorted({k for v in kernels.values() for k in v}):
            values = [kernels[r["input_id"]][module, name] for r in rows]
            number = [counts[r["input_id"]][module, name] for r in rows]
            kernel_stats.append(dict(identity, module=module, kernel_name=name,
                count=len(rows), calls_min=min(number), calls_max=max(number),
                **component_stats(values, latency)))
        runs.append(dict(identity, completed_count=len(rows),
            unique_source_count=len({r["source_frame_id"] for r in rows}),
            P50_ms=summary["P50_ms"], P99_minus_P50_ms=gap,
            kernel_count_min=min(r["kernel_count"] for r in rows),
            kernel_count_max=max(r["kernel_count"] for r in rows), streams=sorted(streams),
            pre_inference_kernels_excluded=excluded))
        all_frames.extend(rows)
        write_csv(folder / "dino_diagnosis/frames.csv", rows)
        for path in (folder / "frames.json", folder / "kernels.csv.gz",
                     root / "runs" / run_id / "config.yaml"):
            evidence[str(path)] = sha256(path)
        print(run_id, "complete", flush=True)
    write_csv(destination / "frames.csv", all_frames)
    write_csv(destination / "module_associations.csv", associations)
    write_csv(destination / "kernel_associations.csv", kernel_stats)
    write_json(destination / "summary.json", dict(runs=runs, sources_sha256=evidence,
        analyzer_sha256=sha256(__file__), helper_sha256=sha256(Path(__file__).with_name("centerpoint_diagnosis.py")),
        limitations=["One execution per condition; temporally dependent frames; sparse-tail P99",
                     "Correlations with latency components are descriptive, not causal proof",
                     "Nonkernel residual includes copies, host work, launch gaps and synchronization",
                     "Kernel timings aggregate each exact name within its recorded top-level module",
                     "Endpoint contributions use the total latency's interpolated percentile frames, not component percentiles",
                     "Existing annotations do not resolve individual layers"]))
    return destination


def kernel_gaps(start, end, intervals):
    """Complement of merged GPU kernel intervals within one inference."""
    gaps, previous = [], start
    for left, right in merge_intervals(intervals):
        if left < start or right > end:
            raise ValueError("Kernel interval outside inference")
        if left > previous:
            gaps.append((previous, left))
        previous = right
    if previous < end:
        gaps.append((previous, end))
    return gaps


def host_diagnosis(root):
    """Locate nonkernel intervals using existing CUDA API/copy and NVTX records.

    Host NVTX position indicates where a gap occurred, not CPU execution time
    or a proof that the named module caused the gap.
    """
    root = Path(root)
    destination = root / "analysis/input2-crossed/input-data/dino_diagnosis"
    summaries = read_json(destination / "summary.json")["runs"]
    rows, associations, evidence, signatures = [], [], {}, {}
    for summary in summaries:
        run_id = summary["execution_id"]
        run = root / "runs" / run_id
        frames = sorted(read_json(root / "analysis/runs" / run_id / "input-data/frames.json"),
                        key=lambda r: r["inference_start_ns"])
        starts = [r["inference_start_ns"] for r in frames]
        pid = int(read_json(run / "model_dino.json")["pid"])
        gpu, copies, api, modules = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        def index(start, end):
            found = bisect_right(starts, start) - 1
            return found if found >= 0 and end <= frames[found]["inference_end_ns"] else None

        with profile_connection(run) as connection:
            for start, end, global_pid in connection.execute(
                    "SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
                if ((int(global_pid) >> 24) & 0xFFFFFF) == pid:
                    i = index(start, end)
                    if i is not None:
                        gpu[i].append((start, end))
            for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
                for start, end, global_pid in connection.execute(f"SELECT start,end,globalPid FROM {table}"):
                    if ((int(global_pid) >> 24) & 0xFFFFFF) == pid:
                        i = index(start, end)
                        if i is not None:
                            copies[i].append((start, end))
            for events in runtime_events(connection).values():
                for event in events:
                    if event["pid"] == pid:
                        i = index(event["start"], event["end"])
                        if i is not None:
                            api[i].append(event)
            for event in nvtx_ranges(connection):
                if event["pid"] == pid and event["tag"].get("event") == "module":
                    i = index(event["start"], event["end"])
                    if i is not None:
                        modules[i].append(event)
            columns = {r[1] for r in connection.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
            fields = [f for f in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                                  "dynamicSharedMemory", "staticSharedMemory") if f in columns]
            query = "SELECT DISTINCT s.value," + ",".join("k."+f for f in fields)
            query += " FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? ORDER BY s.value"
            signatures[run_id] = dict(fields=["kernel_name"]+fields,
                signatures=sorted(connection.execute(query, (starts[0], frames[-1]["inference_end_ns"])).fetchall()))
        local = []
        for i, frame in enumerate(frames):
            gaps = kernel_gaps(frame["inference_start_ns"], frame["inference_end_ns"], gpu[i])
            gap_ms = sum(b-a for a, b in gaps) / 1e6
            row = dict(frame, mps_enabled=summary["mps_enabled"], nonkernel_ms=gap_ms,
                       copy_in_gap_ms=intersection_duration(gaps, copies[i])/1e6)
            row["gap_without_copy_ms"] = gap_ms-row["copy_in_gap_ms"]
            for name in {r["name"] for r in api[i]}:
                events = [(r["start"], r["end"]) for r in api[i] if r["name"] == name]
                row["api_gap:"+name] = intersection_duration(gaps, events)/1e6
            all_api = [(r["start"], r["end"]) for r in api[i]]
            row["gap_outside_cuda_api_ms"] = gap_ms-intersection_duration(gaps, all_api)/1e6
            for name in {r["tag"]["module"] for r in modules[i]}:
                events = [(r["start"], r["end"]) for r in modules[i] if r["tag"]["module"] == name]
                row["host_module_gap:"+name] = intersection_duration(gaps, events)/1e6
                row["host_module_wall:"+name] = sum(b-a for a, b in events)/1e6
            all_modules = [(r["start"], r["end"]) for r in modules[i]]
            row["gap_outside_module_ms"] = gap_ms-intersection_duration(gaps, all_modules)/1e6
            local.append(row)
        latency = [r["latency_ms"] for r in local]
        fields = sorted({k for r in local for k in r if k.startswith(("api_gap:", "host_module_", "gap_")) or k == "copy_in_gap_ms"})
        for field in fields:
            associations.append(dict(execution_id=run_id, scene_id=summary["scene_id"],
                mps_enabled=summary["mps_enabled"], component=field,
                **component_stats([r.get(field, 0) for r in local], latency)))
        rows.extend(local)
        for path in (run / "profile.sqlite.gz", run / "model_dino.json", run / "profile_archive.json"):
            evidence[str(path)] = sha256(path)
        write_csv(root / "analysis/runs" / run_id / "input-data/dino_diagnosis/host_frames.csv", local)
        print(run_id, "host intervals complete", flush=True)
    write_csv(destination / "host_frames.csv", rows)
    write_csv(destination / "host_associations.csv", associations)
    write_json(destination / "host_provenance.json", dict(sources_sha256=evidence,
        analyzer_sha256=sha256(__file__), kernel_launch_signatures=signatures))


def report(root):
    """Summarize observed associations without assigning unmeasured causes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    destination = root / "analysis/input2-crossed/input-data/dino_diagnosis"
    def rows(name):
        with (destination / name).open() as source:
            return list(csv.DictReader(source))
    frames, host = rows("frames.csv"), rows("host_frames.csv")
    modules, kernels = rows("module_associations.csv"), rows("kernel_associations.csv")
    host_stats = rows("host_associations.csv")
    summary = read_json(destination / "summary.json")
    detailed, sensitivity, comparisons = [], [], []
    lines = ["# DINO isolated inference diagnosis", "",
        "Scope: eight executions, four scenes and two MPS modes; 1,848 completed non-warmup inferences. No additional GPU runs or profiling changes. All percentile statistics use every completed observation and NumPy's linear method.", "",
        "DINO's variation cannot be assigned to one dominant GPU kernel. Time outside GPU kernels is the largest additive covariance contributor in all eight executions (60–88%). The largest host-module-localized part of that time occurs while the decoder's NVTX range is active. This locates a scheduling/dispatch interval; it does not measure CPU execution or prove the decoder's mathematical operations caused it.", "",
        "| Scene | MPS | P50 ms | P99-P50 ms | Nonkernel r | Decoder-localized gap r | Encoder GPU r |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for run in summary["runs"]:
        key = run["execution_id"]
        a = {r["component"]: r for r in modules if r["execution_id"] == key}
        h = {r["component"]: r for r in host_stats if r["execution_id"] == key}
        n, d, e = a["nonkernel_ms"], h["host_module_gap:decoder"], a["module:encoder"]
        lines.append(f"| {run['scene_id']} | {'on' if run['mps_enabled'] else 'off'} | {run['P50_ms']:.3f} | {run['P99_minus_P50_ms']:.3f} | {float(n['pearson']):.3f} | {float(d['pearson']):.3f} | {float(e['pearson']):.3f} |")
        detailed.append(dict(run, nonkernel_gap_ms=float(n["total_latency_endpoint_gap_ms"]),
            gpu_gap_ms=float(a["gpu_ms"]["total_latency_endpoint_gap_ms"]),
            decoder_localized_gap_ms=float(d["total_latency_endpoint_gap_ms"])))
        records = sorted([r for r in frames if r["run_id"] == key], key=lambda r: int(r["input_id"]))
        host_by_input = {r["input_id"]: r for r in host if r["run_id"] == key}
        after_first = records[1:]
        latency = [float(r["latency_ms"]) for r in after_first]
        sensitivity.append(dict(execution_id=key, count=len(after_first),
            first_latency_ms=float(records[0]["latency_ms"]),
            first_malloc_gap_ms=float(host_by_input[records[0]["input_id"]]["api_gap:cudaMalloc"]),
            nonkernel_r=correlation([float(r["nonkernel_ms"]) for r in after_first], latency),
            encoder_r=correlation([float(r["module:encoder"]) for r in after_first], latency),
            decoder_localized_gap_r=correlation([float(host_by_input[r["input_id"]]["host_module_gap:decoder"]) for r in after_first], latency)))
    lines += ["", "## Which kernels?", "",
        "Among individual exact-name kernel families, encoder `ampere_sgemm_128x64_tn` has the largest positive covariance contribution in every execution. It is called 36 times per inference, totaling approximately 18.3–18.5 ms, but contributes only 2–15% of total latency variance under the additive covariance accounting. Its Pearson r is 0.18–0.49. This is a kernel family aggregated within the encoder, not one launch or one identified neural-network layer.", "",
        "The decoder's own GPU time is approximately 6.23–6.33 ms and correlates weakly with total latency (r=0.04–0.18). Its host-side gaps are much more strongly associated (r=0.59–0.84). GPU deformable-attention kernels are not dominant variability contributors: encoder aggregate standard deviations are about 0.017–0.023 ms. Every completed inference has 1,280 kernels; all eight runs have the same 272 distinct kernel-name/grid/block/shared-memory signatures. This supports stable launch structure, but does not by itself prove identical memory-access behavior.", "",
        "## Tail accounting in milliseconds", "",
        "Components are evaluated on the same interpolated frames defining total P50 and P99. GPU + nonkernel contributions add to the total gap. The decoder-localized column is part of nonkernel time, not an additional component. Values can be negative; these are not independent module percentiles.", "",
        "| Scene | MPS | Total gap | GPU contribution | Nonkernel contribution | Decoder-localized portion |",
        "|---|---|---:|---:|---:|---:|"]
    for row in detailed:
        lines.append(f"| {row['scene_id']} | {'on' if row['mps_enabled'] else 'off'} | {row['P99_minus_P50_ms']:.3f} | {row['gpu_gap_ms']:.3f} | {row['nonkernel_gap_ms']:.3f} | {row['decoder_localized_gap_ms']:.3f} |")
    lines += ["", "## Concrete startup mechanism", "",
        "The harness calls `profiler.release_cached_memory()` after its five warmups, and that method calls `torch.cuda.empty_cache()`. Every run shows approximately 3.0–3.5 ms of kernel-free time inside cudaMalloc on its first measured inference, plus approximately 0.10–0.13 ms on the second. This explains a startup allocation contribution despite warmup. The inspected warmup image and the first images of all four scenes are each 1600×900, so this is not evidence of a different warmup image resolution.", "",
        "This allocation contribution is zero at the interpolated P50/P99 endpoint frames in all eight runs: it explains the first-frame spike, not the reported P99-P50 gaps. `startup_sensitivity.csv` reports correlations with only the first measured frame excluded as a diagnostic; stored samples and official percentiles remain unchanged. Excluding that frame reduces decoder-gap correlations to 0.38–0.79, showing that startup amplifies the full-sample association.", "",
        "PyTorch documents that empty_cache releases unused allocator cache: https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html", "",
        "## Across scenes and matching inputs", "",
        "The scene-to-scene P50 spread is only 0.362 ms with MPS off and 0.442 ms with MPS on. Tail-gap spreads are 1.623 and 1.402 ms respectively. The largest tail gap occurs in 0184 with MPS off and 0245 with MPS on, rather than one consistently slow input scene.", "",
        "| Scene | Matching source frames | Cross-mode latency r | Cross-mode r excluding first frame |",
        "|---|---:|---:|---:|"]
    scenes = ["scene-0770", "scene-0398", "scene-0184", "scene-0245"]
    for scene in scenes:
        off = {r["source_frame_id"]: r for r in frames if r["scene_id"] == scene and r["mps_enabled"] == "False"}
        on = {r["source_frame_id"]: r for r in frames if r["scene_id"] == scene and r["mps_enabled"] == "True"}
        common = sorted(off.keys() & on.keys())
        later = [k for k in common if off[k]["input_id"] != "0" and on[k]["input_id"] != "0"]
        corr = correlation([float(off[k]["latency_ms"]) for k in common], [float(on[k]["latency_ms"]) for k in common])
        after = correlation([float(off[k]["latency_ms"]) for k in later], [float(on[k]["latency_ms"]) for k in later])
        comparisons.append(dict(scene_id=scene, common_source_frames=len(common), latency_r=corr, after_first_latency_r=after))
        lines.append(f"| {scene} | {len(common)} | {corr:.3f} | {after:.3f} |")
    lines += ["", "## What remains unresolved", "",
        "Observed location: kernel-free intervals, most strongly localized to the decoder's host range. Candidate mechanisms include Python/framework dispatch, allocator behavior, CPU scheduling and driver launch pacing. CUDA copies/memsets occupy only about 0.32–0.33 ms per inference; their covariance contribution is about 0.2–1.2%, providing little support for GPU copy duration as the dominant source. Most variable kernel-free time lies outside recorded CUDA API calls.", "",
        "The primary traces disable CPU sampling, backtraces and CPU context-switch collection. They cannot distinguish those host mechanisms or resolve finer decoder operations. One execution per scene/mode and temporally dependent frames do not establish an image-content-induced effect. All P99 estimates here are sparse-tail estimates (<1,000 observations). Correlation with a component of total latency is not causal proof. Multiple kernel comparisons are descriptive, with no significance claim.", "",
        "A causal follow-up would hold the exact image sequence fixed across independent executions and collect CPU stacks/context switches alongside existing CUDA launches. A separately authorized allocator-cache control would test the startup mechanism. No such runs or configuration changes were made.", "",
        "Reproduce after sourcing ROS and the workspace:", "", "```bash",
        "python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2",
        "python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --host-only",
        "python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only", "```"]
    write_csv(destination / "tail_accounting.csv", detailed)
    write_csv(destination / "startup_sensitivity.csv", sensitivity)
    write_csv(destination / "matched_source_comparison.csv", comparisons)
    (destination / "report.md").write_text("\n".join(lines)+"\n")
    fig, ax = plt.subplots(figsize=(9, 6))
    for scene, marker in zip(scenes, ("o", "s", "^", "D")):
        for mode, color in (("False", "#4c78a8"), ("True", "#e68a2e")):
            selected = [r for r in host if r["scene_id"] == scene and r["mps_enabled"] == mode]
            ax.scatter([float(r["host_module_gap:decoder"]) for r in selected],
                       [float(r["latency_ms"]) for r in selected], s=15, alpha=.65,
                       marker=marker, color=color, linewidths=.2, edgecolors="#333333",
                       label=scene[6:]+(" MPS off" if mode == "False" else " MPS on"))
    ax.set(xlabel="Kernel-free time inside the decoder host range (ms)",
           ylabel="Completed inference time (ms)", title="DINO: decoder host gaps track latency variation")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    fig.text(.5, .015, "All completed non-warmup frames, including initial allocations. Gap location is not CPU execution time.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(destination / "decoder_gap_correlation.png", dpi=180)
    fig.savefig(destination / "decoder_gap_correlation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.artifact_root)
    elif args.host_only:
        host_diagnosis(args.artifact_root)
    else:
        print(diagnose(args.artifact_root))
