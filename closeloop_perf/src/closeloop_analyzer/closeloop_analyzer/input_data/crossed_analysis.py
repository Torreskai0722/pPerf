"""Frozen selection and repeated four-cell input sensitivity analysis."""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from .crossed_evidence import (
    analyze_execution, archive_index, inspect_execution, metrics, read_json, sha256,
    write_csv, write_json,
)
from . import crossed_evidence
from .. import sample_filter
from ..sample_filter import POLICY, filter_frames, p1_p99

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def block_id(pair, mode):
    """Use the manifest's authored pair order."""
    return f"{'+'.join(pair)}:mps-{'on' if mode else 'off'}"


def choose_scenes(pair, scenes, measurements):
    """Exact ties follow model/scene order; all-tied A and B are distinct."""
    ranges = {model: max(measurements[model][s]["R"] for s in scenes) -
              min(measurements[model][s]["R"] for s in scenes) for model in pair}
    model = max(pair, key=lambda m: ranges[m])
    a = min(scenes, key=lambda s: measurements[model][s]["R"])
    b = max(scenes, key=lambda s: measurements[model][s]["R"])
    if a == b:
        b = next(s for s in scenes if s != a)
    return {"selected_model": model, "A": a, "B": b, "R_ranges": ranges,
            "measurements": measurements,
            "tie_rule": "authored pair model order, then authored scene order; distinct A/B"}


def freeze_selection(path, selection):
    """Never silently overwrite selection evidence or change chosen scenes."""
    path = Path(path)
    if path.exists():
        if read_json(path) != selection:
            raise ValueError("selection.json is frozen and differs from current evidence")
    else:
        write_json(path, selection)
        path.chmod(0o444)
    return selection


def collect(manifest, output_root, phases):
    """Revalidate retained evidence before using or caching per-run results."""
    summaries, frames, blockers, archives = [], {}, [], []
    cache = {}
    for entry in manifest["executions"]:
        for attempt in entry.get("attempts", []):
            retained = {"run_id": Path(attempt["artifact_directory"]).name,
                        "role": "excluded_attempt", "accepted": False,
                        "rejection_reasons": [attempt.get("blocker", "replaced attempt")]}
            try:
                retained.update(archive_index(attempt["artifact_directory"]))
            except (OSError, ValueError, KeyError) as exc:
                retained["archive_blocker"] = str(exc)
            archives.append(retained)
        for decision in entry.get("reuse", []):
            if decision.get("archives"):
                archives.append({"run_id": Path(decision["path"]).name,
                                 "role": "reuse_candidate", "accepted": decision["accepted"],
                                 "rejection_reasons": decision["reasons"], **decision["archives"]})
        if entry["phase"] not in phases:
            continue
        if entry["status"] != "validated":
            blockers.append({"run_id": entry["run_id"], "phase": entry["phase"],
                             "reason": entry.get("blocker", "no validated execution recorded")})
            continue
        actual = entry["artifact_directory"]
        try:
            if actual not in cache:
                config = yaml.safe_load(Path(entry["config_path"]).read_text())
                if sha256(entry["config_path"]) != entry["config_sha256"]:
                    raise ValueError("generated configuration hash changed")
                evidence = inspect_execution(actual, config, manifest["hardware"])
                if not evidence["accepted"]:
                    raise ValueError("; ".join(evidence["reasons"]))
                if evidence["file_hashes"] != entry["evidence"]["file_hashes"]:
                    raise ValueError("accepted execution evidence hashes changed")
                destination = Path(output_root) / "analysis/runs" / Path(actual).name / "input-data"
                provenance_path = destination / "analysis_provenance.json"
                identity = {"evidence_hashes": evidence["file_hashes"],
                            "sample_filter_sha256": sha256(sample_filter.__file__),
                            "analyzer_sha256": sha256(crossed_evidence.__file__)}
                prior = read_json(provenance_path) if provenance_path.is_file() else {}
                if (prior.get("identity") == identity and prior.get("outputs")
                        and all((destination / name).is_file() and sha256(destination / name) == digest
                                for name, digest in prior["outputs"].items())):
                    result, run_frames = read_json(destination / "summary.json"), read_json(destination / "frames.json")
                else:
                    result, run_frames = analyze_execution(entry, output_root)
                    write_json(provenance_path, {"identity": identity, "outputs": {
                        name: sha256(destination / name) for name in (
                            "summary.json", "frames.json", "frames.csv", "frame_coverage.csv", "unmatched.json", "kernels.csv.gz")}})
                cache[actual] = result, run_frames
                archives.append({"run_id": Path(actual).name, "role": "validated_execution", **evidence["archives"]})
            result, run_frames = cache[actual]
            identity = {k: entry[k] for k in ("phase", "pair", "mps_enabled", "lidar_scene", "camera_scene", "repetition", "cell")}
            identity["execution_id"] = Path(actual).name
            identity["slot_id"] = entry["run_id"]
            identity["selection_evidence"] = entry["selection_evidence"]
            for row in result["models"]:
                summaries.append({**identity, **row})
            frames[entry["run_id"]] = run_frames
        except (OSError, ValueError, KeyError) as exc:
            blockers.append({"run_id": entry["run_id"], "phase": entry["phase"], "reason": str(exc)})
    return summaries, frames, blockers, archives


def select(manifest, summaries, destination, filename="selection.json"):
    """Require all forty screening executions before freezing ten selections."""
    # JSON object key sorting must not change the authored tie-break order.
    scenes = list(yaml.safe_load(Path(manifest["study_path"]).read_text())["conditions"])
    selection = {"schema": "input2_crossed_selection_v1", "study_sha256": manifest["study_sha256"],
                 "window_ns": manifest["window_ns"], "percentile_method": "numpy.linear after P1–P99 filtering",
                 "sample_filter_policy": POLICY,
                 "R_definition": "(P99-P50)/P50", "selections": {}}
    for mode in manifest["study"]["mps_modes"]:
        for pair in manifest["study"]["pairs"]:
            measurements = {model: {} for model in pair}
            hashes = {}
            warnings = []
            for row in summaries:
                if row["phase"] != "screening" or row["pair"] != pair or row["mps_enabled"] != mode:
                    continue
                if row["lidar_scene"] != row["camera_scene"]:
                    raise ValueError("screening must use same-scene bags")
                if row["scene_id"] in measurements[row["model_id"]]:
                    raise ValueError("screening has duplicate execution cells")
                measurements[row["model_id"]][row["scene_id"]] = row
                entry = next(e for e in manifest["executions"] if e["run_id"] == row["slot_id"])
                hashes[row["slot_id"]] = sha256(Path(entry["artifact_directory"]) / "run_manifest.json")
                if row["sample_warning"]:
                    warnings.append(f"{row['model_id']}/{row['scene_id']}: {row['sample_warning']}; unique={row['unique_source_count']}")
            if any(set(values) != set(scenes) for values in measurements.values()):
                raise ValueError(f"selection blocked: both models need four valid scenes in {block_id(pair, mode)}")
            selection["selections"][block_id(pair, mode)] = {
                **choose_scenes(pair, scenes, measurements), "sample_warnings": warnings,
                "source_run_hashes": hashes,
                "source_evidence_hashes": {run_id: next(e["evidence"]["file_hashes"] for e in manifest["executions"] if e["run_id"] == run_id)
                                           for run_id in hashes}}
    return freeze_selection(destination / filename, selection)


def distribution(values):
    """Summarize execution-level values without treating frames as replicates."""
    values = list(map(float, values))
    return {"executions": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "minimum": min(values), "maximum": max(values), "range": max(values) - min(values)}


def contrasts(cells, modality):
    """First cell letter is LiDAR input; second is camera input."""
    co = (("AB", "AA"), ("BB", "BA")) if modality == "lidar" else (("BA", "AA"), ("BB", "AB"))
    own = (("BA", "AA"), ("BB", "AB")) if modality == "lidar" else (("AB", "AA"), ("BB", "BA"))
    result = [("co_runner", f"{b}-{a}", cells[b] - cells[a]) for b, a in co]
    result += [("own_input", f"{b}-{a}", cells[b] - cells[a]) for b, a in own]
    return result + [("interaction", "BB-BA-AB+AA", cells["BB"] - cells["BA"] - cells["AB"] + cells["AA"])]


def report_tables(manifest, summaries, frames, selection):
    """Compute per-repetition contrasts, isolation deltas, and frame controls."""
    grouped, isolated = defaultdict(dict), {}
    for row in summaries:
        if row["phase"] == "confirmation":
            grouped[(tuple(row["pair"]), row["mps_enabled"], row["model_id"], row["repetition"])][row["cell"]] = row
        elif row["phase"] == "isolated":
            isolated[(row["model_id"], row["mps_enabled"], row["scene_id"])] = row
    effects, common = [], []
    fields = ("R", "P50_ms", "P99_ms", "P99_minus_P50_ms", "throughput_hz", "analysis_count", "analysis_unique_source_count")
    for (pair, mode, model, repetition), cells in grouped.items():
        if set(cells) != {"AA", "AB", "BA", "BB"}:
            continue
        modality = manifest["study"]["models"][model]["modality"]
        selected = selection["selections"][block_id(pair, mode)]
        identity = {"pair": "+".join(pair), "mps_enabled": mode, "model_id": model, "repetition": repetition,
                    "A": selected["A"], "B": selected["B"]}
        for metric in fields:
            for kind, contrast, value in contrasts({c: r[metric] for c, r in cells.items()}, modality):
                row = {**identity, "metric": metric, "effect": kind, "contrast": contrast, "value": value}
                if kind == "own_input":
                    a = isolated.get((model, mode, selected["A"]))
                    b = isolated.get((model, mode, selected["B"]))
                    row["isolated_B_minus_A"] = b[metric] - a[metric] if a and b else None
                    row["paired_minus_isolated_effect"] = value - row["isolated_B_minus_A"] if a and b else None
                effects.append(row)
        co = (("AB", "AA"), ("BB", "BA")) if modality == "lidar" else (("BA", "AA"), ("BB", "AB"))
        for b, a in co:
            streams = []
            duplicates = set()
            for cell in (a, b):
                by_source = defaultdict(list)
                selected_frames, _ = filter_frames([r for r in frames[cells[cell]["slot_id"]] if r["model_id"] == model])
                for row in selected_frames:
                    by_source[row["source_frame_id"]].append(row)
                duplicates.update(source for source, records in by_source.items() if len(records) != 1)
                streams.append({source: records[0] for source, records in by_source.items() if len(records) == 1})
            left, right = streams
            shared = sorted(left.keys() & right.keys())
            row = {**identity, "contrast": f"{b}-{a}", "common_source_count": len(shared),
                   "left_only_source_ids": sorted(left.keys() - right.keys()),
                   "right_only_source_ids": sorted(right.keys() - left.keys()), "source_frame_ids": shared,
                   "excluded_duplicate_source_ids": sorted(duplicates)}
            if shared:
                lm = metrics([left[s]["latency_ms"] for s in shared], len(shared), cells[a]["elapsed_seconds"], filtered=True)
                rm = metrics([right[s]["latency_ms"] for s in shared], len(shared), cells[b]["elapsed_seconds"], filtered=True)
                row.update(common_R_difference=rm["R"] - lm["R"], common_P50_difference_ms=rm["P50_ms"] - lm["P50_ms"],
                           sample_warning=lm["sample_warning"],
                           paired_frame_mean_difference_ms=float(np.mean([right[s]["latency_ms"] - left[s]["latency_ms"] for s in shared])))
            common.append(row)
    return effects, common


def aggregate(rows, keys, metric="value"):
    """Group individual executions, preserving the number of repetitions."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row[metric])
    return [{**dict(zip(keys, key)), **distribution(values)} for key, values in grouped.items()]


def write_findings(summaries, destination):
    """Keep the narrative synchronized with the current filtered measurements."""
    baselines = {(r["model_id"], r["scene_id"], r["mps_enabled"]): r
                 for r in summaries if r["phase"] == "isolated"}
    lines = ["# Input2-crossed findings — P1–P99 filtered", "",
             "All current latency statistics and plots use the same retained per-execution/model frames. "
             "P50/P99 are recomputed after filtering; original cutoffs and exclusions are retained. "
             "Full evidence and the historical selection remain unchanged. "
             "See [complete tables and contrasts](report.md), [baseline violins](isolated_baselines.md), "
             "[CenterPoint diagnosis](centerpoint_diagnosis/report.md), and [DINO diagnosis](dino_diagnosis/report.md).", "",
             "The practical predictability threshold is an absolute change of approximately 1 ms in P99−P50. "
             "The table uses a strict ≤1 ms comparison. Isolated mode comparisons have one execution per condition "
             "and are observations, not statistical equivalence tests. 3DSSD and YOLOv3 are assessed separately.", "",
             "| Model | Scene | MPS-off P99−P50 ms | MPS-on P99−P50 ms | On−off ms | Within 1 ms |",
             "|---|---|---:|---:|---:|---|"]
    for model, scene in sorted({(m, s) for m, s, _ in baselines}):
        off, on = baselines.get((model, scene, False)), baselines.get((model, scene, True))
        if not off or not on:
            continue
        a, b = off["P99_minus_P50_ms"], on["P99_minus_P50_ms"]
        lines.append(f"| {model} | {scene} | {a:.3f} | {b:.3f} | {b-a:+.3f} | {abs(b-a) <= 1} |")
    lines += ["", "Crossed contrasts compare actual executed input combinations at fixed co-runner or own input. "
              "Their execution-level means, standard deviations and ranges preserve independent repetitions. "
              "Common-frame comparisons intersect the retained source identities. Completion coverage and "
              "observed throughput remain evidence about the full run; filtered throughput uses retained count "
              "over the original elapsed duration. Mechanism correlations are descriptive and do not establish causation.", "",
              "Historical selection.json identifies the actual completed matrix. Revised P1–P99 screening choices "
              "are in selection_p1_p99.json; no confirmation runs are relabeled to match revised choices. "
              "Filtered samples remain temporally dependent; P99 with fewer than 100 retained observations is "
              "especially fragile and fewer than 1,000 is sparse-tail. No additional runs were scheduled."]
    (destination / "findings.md").write_text("\n".join(lines) + "\n")


def mode_comparisons(summaries):
    """Compare modes only when actual ordered scene combinations coincide."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in summaries:
        if row["phase"] != "confirmation":
            continue
        key = ("+".join(row["pair"]), row["model_id"], row["lidar_scene"], row["camera_scene"])
        grouped[key][row["mps_enabled"]].append(row)
    result = []
    for key, modes in grouped.items():
        if set(modes) != {False, True}:
            continue
        for field in ("R", "P50_ms", "P99_ms", "P99_minus_P50_ms", "throughput_hz", "analysis_unique_source_count"):
            off = distribution([r[field] for r in modes[False]])
            on = distribution([r[field] for r in modes[True]])
            result.append({**dict(zip(("pair", "model_id", "lidar_scene", "camera_scene"), key)),
                           "metric": field, "on_minus_off_mean": on["mean"] - off["mean"],
                           **{"off_" + k: v for k, v in off.items()}, **{"on_" + k: v for k, v in on.items()}})
    return result


def plots(summaries, destination):
    """Show all executions alongside means and execution-level spread."""
    groups = defaultdict(list)
    for row in summaries:
        if row["phase"] == "confirmation":
            groups[(tuple(row["pair"]), row["mps_enabled"], row["model_id"])].append(row)
    for (pair, mode, model), rows in groups.items():
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        cells = ("AA", "AB", "BA", "BB")
        for axis, field in zip(axes, ("R", "P50_ms", "throughput_hz")):
            for i, cell in enumerate(cells):
                values = [r[field] for r in rows if r["cell"] == cell]
                if values:
                    axis.scatter([i] * len(values), values, s=22, alpha=.65)
                    axis.errorbar(i, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                                  color="black", fmt="_", capsize=5)
            axis.set_xticks(range(4), cells)
            axis.set_ylabel(field)
            axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"{'+'.join(pair)} | {model} | MPS {'on' if mode else 'off'} | P1–P99 filtered")
        figure.tight_layout()
        path = destination / "plots" / f"{'+'.join(pair)}-{int(mode)}-{model}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)


def isolated_violin_plots(manifest, summaries, frames, destination):
    """Use the same P1–P99 sample for baseline plots and reported metrics."""
    scenes = manifest["scene_order"]
    modes = manifest["study"]["mps_modes"]
    grouped = {}
    for row in summaries:
        if row["phase"] != "isolated":
            continue
        key = (row["model_id"], row["mps_enabled"], row["scene_id"])
        if key in grouped:
            raise ValueError(f"baseline violins require one execution per condition: {key}")
        samples = [r for r in frames[row["slot_id"]] if r["model_id"] == row["model_id"]]
        values = np.asarray([r["latency_ms"] for r in samples], dtype=float)
        if (len(values) != row["completed_count"] or not len(values)
                or any(r["scene_id"] != row["scene_id"] for r in samples)
                or len({r["source_frame_id"] for r in samples}) != row["unique_source_count"]
                or np.any(~np.isfinite(values)) or np.any(values <= 0)):
            raise ValueError(f"baseline frame evidence differs from summary: {key}")
        grouped[key] = (row, values)
    plot_root = destination / "plots/isolated"
    plot_root.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_root / "isolated_baselines.pdf"
    outputs, display_counts = [], []
    colors = {False: "#4c78a8", True: "#e68a2e"}
    with PdfPages(pdf_path) as pdf:
        for model in manifest["study"]["models"]:
            figure, axis = plt.subplots(figsize=(10, 5.6))
            for mode in modes:
                for position, scene in enumerate(scenes):
                    sample = grouped.get((model, mode, scene))
                    if sample is None:
                        continue
                    row, values = sample
                    mask, audit = p1_p99(values)
                    displayed = values[mask]
                    display_counts.append({"model_id": model, "mps_enabled": mode, "scene_id": scene,
                                           "execution_id": row["execution_id"], "completed_count": len(values),
                                           "unique_source_count": row["unique_source_count"], **audit,
                                           "displayed_count": len(displayed)})
                    violin = axis.violinplot(
                        [displayed], positions=[position + (.18 if mode else -.18)], widths=.32, showmedians=True,
                        showextrema=True, quantiles=[[.25, .75]], points=200,
                    )
                    for body in violin["bodies"]:
                        body.set_facecolor(colors[mode])
                        body.set_edgecolor("#333333")
                        body.set_alpha(.65)
                    for name in ("cmedians", "cquantiles", "cmins", "cmaxes", "cbars"):
                        violin[name].set_color("#333333")
                        violin[name].set_linewidth(1.0 if name == "cmedians" else .65)
            axis.set_xticks(range(len(scenes)), [s.removeprefix("scene-") for s in scenes])
            axis.set_xlim(-.6, len(scenes) - .4)
            axis.set_xlabel("Scene")
            axis.set_ylabel("Inference time (ms)")
            axis.set_title(f"{model} — single-model baseline (P1–P99 filtered)")
            axis.grid(axis="y", alpha=.2)
            axis.legend(handles=[Patch(facecolor=colors[mode], edgecolor="#333333", alpha=.65,
                                       label=f"MPS {'on' if mode else 'off'}") for mode in modes], frameon=False)
            figure.tight_layout()
            path = plot_root / f"{model}.png"
            figure.savefig(path, dpi=180)
            pdf.savefig(figure)
            plt.close(figure)
            outputs.append(path.relative_to(destination).as_posix())
    counts_path = destination / "isolated_violin_display.csv"
    write_csv(counts_path, display_counts)
    index = destination / "isolated_baselines.md"
    lines = ["# Single-model baseline inference times", "",
             "Scenes are on the x-axis; CUDA-complete inference time in milliseconds is on the y-axis. "
             "Each model has one plot with MPS-off (blue, left) and MPS-on (orange, right) next to each other at each scene. "
             "Each violin uses the same P1–P99-filtered observations as the analysis metrics. "
             "Original per-execution cutoffs use NumPy linear percentiles; metrics are recomputed after filtering. "
             "Raw recordings and the historical execution selection remain unchanged. "
             "External decode and preprocessing are excluded from this inference range. "
             "Counts, unique source-frame counts, cutoffs, and omitted counts are retained in the linked table; missing evidence is left empty. "
             "Within-execution frames are temporally dependent and are not independent repetitions.", "",
             "[Download all models as a multipage PDF](plots/isolated/isolated_baselines.pdf) · "
             "[Display cutoffs and sample counts](isolated_violin_display.csv)", ""]
    for path in outputs:
        lines.extend([f"## {Path(path).stem}", "", f"![{Path(path).stem} baseline violins]({path})", ""])
    index.write_text("\n".join(lines))
    return {"index": str(index), "pdf": str(pdf_path), "plots": outputs, "display_counts": str(counts_path)}


def analyze(source, output_root, options):
    """Analyze the experiment ledger using the repository study/run layouts."""
    manifest = read_json(source)
    if sha256(manifest["study_path"]) != manifest["study_sha256"]:
        raise ValueError("authored study differs from frozen experiment")
    phase = options.get("phase", "report")
    if phase not in ("select", "report"):
        raise ValueError("crossed analysis phase must be select or report")
    expected_counts = {"isolated": 56, "screening": 40}
    if phase == "report":
        expected_counts["confirmation"] = 120
    for name, count in expected_counts.items():
        if sum(e["phase"] == name for e in manifest["executions"]) != count:
            raise ValueError(f"incomplete {name} matrix: expected {count} execution slots")
    if len({e["run_id"] for e in manifest["executions"]}) != len(manifest["executions"]):
        raise ValueError("experiment contains duplicate execution slots")
    destination = Path(output_root) / "analysis/input2-crossed/input-data"
    destination.mkdir(parents=True, exist_ok=True)
    phases = ("screening",) if phase == "select" else ("screening", "isolated", "confirmation")
    summaries, frames, blockers, archives = collect(manifest, output_root, phases)
    write_json(destination / "blockers.json", blockers)
    write_json(destination / "trace_archive_index.json", archives)
    write_json(destination / "per_run_summaries.json", summaries)
    write_csv(destination / "kernel_attribution_coverage.csv", [
        {"execution_id": execution_id, **read_json(
            Path(output_root) / "analysis/runs" / execution_id / "input-data/summary.json")["kernel_attribution"]}
        for execution_id in sorted({row["execution_id"] for row in summaries})])
    write_csv(destination / "per_run_summaries.csv", [{**r, "pair": "+".join(r["pair"] or []),
        "sample_filter": json.dumps(r["sample_filter"], sort_keys=True),
        "coverage": json.dumps(r["coverage"], sort_keys=True)} for r in summaries])
    write_csv(destination / "frame_coverage_summary.csv", [
        {**{k: r[k] for k in ("slot_id", "execution_id", "phase", "model_id", "mps_enabled", "lidar_scene", "camera_scene", "cell", "repetition")},
         **r["coverage"], "unique_source_count": r["unique_source_count"], "unmatched_count": r["unmatched_count"]}
        for r in summaries])
    if phase == "select":
        name = "selection.json"
        if (destination / name).exists() and read_json(destination / name).get("sample_filter_policy") != POLICY:
            name = "selection_p1_p99.json"
        selected = select(manifest, summaries, destination, name)
        return {"selection": str(destination / name), "selections": len(selected["selections"]), "blockers": blockers}
    selection_path = destination / "selection.json"
    if not manifest.get("selection") or sha256(selection_path) != manifest["selection"]["sha256"]:
        raise ValueError("report requires the frozen selection used by confirmation")
    selection = read_json(selection_path)
    # The completed matrix keeps its actual historical A/B identities.
    # Revised screening choices are reported separately, never substituted.
    if selection.get("sample_filter_policy") != POLICY:
        select(manifest, summaries, destination, "selection_p1_p99.json")
    effects, common = report_tables(manifest, summaries, frames, selection)
    write_csv(destination / "four_cell_contrasts.csv", effects)
    aggregate_effects = aggregate(effects, ("pair", "mps_enabled", "model_id", "A", "B", "metric", "effect", "contrast"))
    write_csv(destination / "four_cell_contrast_summary.csv", aggregate_effects)
    cells = [{**r, "pair": "+".join(r["pair"])} for r in summaries if r["phase"] == "confirmation"]
    four_cells = []
    for metric in ("R", "P50_ms", "P99_ms", "P99_minus_P50_ms", "throughput_hz", "analysis_unique_source_count"):
        four_cells.extend({**r, "metric": metric} for r in aggregate(cells, ("pair", "mps_enabled", "model_id", "cell", "lidar_scene", "camera_scene"), metric))
    write_csv(destination / "four_cell_summary.csv", four_cells)
    write_json(destination / "common_processed_frames.json", common)
    write_csv(destination / "common_processed_frame_comparisons.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in common])
    write_csv(destination / "mps_matched_scenes.csv", mode_comparisons(summaries))
    scene_modes = defaultdict(set)
    for row in cells:
        scene_modes[(row["pair"], row["model_id"], row["lidar_scene"], row["camera_scene"])].add(row["mps_enabled"])
    write_json(destination / "mps_unmatched_conditions.json", [
        {**dict(zip(("pair", "model_id", "lidar_scene", "camera_scene"), key)),
         "available_modes": sorted(modes), "reason": "no valid other-mode execution for this actual scene combination"}
        for key, modes in scene_modes.items() if len(modes) != 2])
    plots(summaries, destination)
    baselines = isolated_violin_plots(manifest, summaries, frames, destination)
    warnings = [f"{r['slot_id']}/{r['model_id']}: {r['sample_warning']}; unique={r['unique_source_count']}" for r in summaries if r["sample_warning"]]
    controls = [r for r in aggregate_effects if r["pair"] == "3dssd+yolov3" and r["metric"] == "R"]
    lines = ["# Controlled environment sensitivity", "",
             f"Validated analysis rows: {len(summaries)}. Blocked execution slots: {len(blockers)}.", "",
             "[Single-model baseline violin plots by scene](isolated_baselines.md) "
             "([all models as PDF](plots/isolated/isolated_baselines.pdf)).", "",
             "All plots and performance statistics use one inclusive P1–P99 latency filter per execution/model, with original cutoffs computed using NumPy linear percentiles. P50, P99, P99−P50, and R=(P99−P50)/P50 are recomputed on the retained sample. Each execution remains a separate observation; means, sample standard deviations, and ranges summarize executions. Raw coverage and archive evidence remain intact. Per-run summaries record cutoffs and excluded source/input IDs.", "",
             "The frozen selection.json records the historical decisions used to execute this matrix. selection_p1_p99.json records revised screening choices under the new policy; it does not change the actual A/B scenes of completed executions.", "",
             "The first cell letter selects LiDAR input and the second selects camera input. LiDAR co-runner contrasts are AB−AA and BB−BA; camera co-runner contrasts are BA−AA and BB−AB. Own-input contrasts are compared with the corresponding single isolated A/B measurements. Isolation has one execution per scene/mode, so its uncertainty cannot be estimated from repetitions.", "",
             "Observed associations are the same-scene screening differences. The crossed contrasts support input effects under the fixed replay/resource protocol; changes in processed-frame coverage can mediate these effects. Common-processed-frame comparisons are supplementary and condition on completion. They do not establish an internal GPU mechanism.", "",
             "Expected bag records are not observed publications. Relay publication is the measured publication boundary. Missing bag-to-relay coverage is upstream missing coverage; published inputs absent from callbacks are dropped/overwritten without a queue-internal mechanism claim.", "",
             "Inference NVTX ranges include recorded CUDA completion; decode and preprocessing use the same retained frame identities and remain separate. throughput_hz is the filtered inference count divided by the original replay-resume-to-max(window-end,last-completion) duration. observed_throughput_hz and completed_count retain actual completion evidence; elapsed and drain boundaries are unchanged.", "",
             "Sequential frames are temporally dependent. P99 with <100 observations is especially fragile; <1,000 is a sparse-tail estimate. Unique-source counts and individual-run warnings are retained. Targeted longer common-window measurements are recommended for unstable tails; this study does not automatically add repetitions.", "",
             "Kernel exports retain names, timing, process/context/stream IDs, launch correlations, and supported model/input/module attribution. Coverage includes unresolved kernel counts and summed GPU kernel duration (overlap is not collapsed). Depth-zero annotations cannot resolve individual layers; module/layer hooks with correlated launches would be needed. No layer-detail runs were scheduled.", "",
             "MPS comparisons include only identical actual ordered scene combinations. Selection evidence reused in AA/BB is marked; selection-conditioned comparisons may be optimistic.", "",
             "## 3DSSD and YOLOv3 assessed separately", "",
             "Use the requested approximately 1 ms margin for absolute changes in P99−P50. Small observed effects do not establish statistical equivalence with one isolated execution per condition.", "",
             "| Model | MPS | Effect | Contrast | Mean ΔR | SD | Range |", "|---|---|---|---|---:|---:|---:|"]
    for r in controls:
        lines.append(f"| {r['model_id']} | {r['mps_enabled']} | {r['effect']} | {r['contrast']} | {r['mean']:.5g} | {r['std']} | {r['range']:.5g} |")
    lines += ["", "## Remaining evidence blockers", ""] + [f"- {r['run_id']}: {r['reason']}" for r in blockers]
    lines += ["", "## All pairs: execution-level input effects", "",
              "| Pair | Model | MPS | Effect | Contrast | Mean ΔR | SD | Min | Max |",
              "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in aggregate_effects:
        if r["metric"] == "R":
            lines.append(f"| {r['pair']} | {r['model_id']} | {r['mps_enabled']} | {r['effect']} | {r['contrast']} | {r['mean']:.5g} | {r['std']} | {r['minimum']:.5g} | {r['maximum']:.5g} |")
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    write_findings(summaries, destination)
    write_json(destination / "sample_warnings.json", warnings)
    write_json(destination / "longer_measurement_candidates.json", [
        {"pair": row["pair"], "model_id": row["model_id"], "mps_enabled": row["mps_enabled"],
         "lidar_scene": row["lidar_scene"], "camera_scene": row["camera_scene"],
         "cell": row["cell"], "repetition": row["repetition"],
         "analysis_count": row["analysis_count"], "analysis_unique_source_count": row["analysis_unique_source_count"],
         "priority": "especially fragile" if row["analysis_count"] < 100 else "sparse tail",
         "recommendation": "Use longer matched source recordings in a separately specified common-window follow-up; retain temporal-dependence checks. Do not add repetitions to this matrix."}
        for row in summaries if row["phase"] == "confirmation" and row["analysis_count"] < 1000])
    result = {"report": str(destination / "report.md"), "validated_execution_slots": len(frames),
              "sample_filter_policy": POLICY,
              "isolated_baselines": baselines,
              "unique_executions": len({row["execution_id"] for row in summaries}),
              "blocked_execution_slots": len(blockers), "complete": not blockers,
              "limitations": ["sparse tails", "temporal dependence", "depth-zero layer attribution", "selection-conditioned AA/BB reuse"]}
    write_json(destination / "summary.json", result)
    return result
