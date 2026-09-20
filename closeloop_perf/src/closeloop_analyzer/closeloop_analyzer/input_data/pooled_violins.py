"""Export confirmation inference-time violins with three executions pooled.

Run after sourcing ROS and the workspace:
python3 -m closeloop_analyzer.input_data.pooled_violins MANIFEST --output-root ROOT
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .crossed_analysis import PdfPages, plt
from .crossed_evidence import read_json, sha256, write_csv, write_json


def pool_confirmation(summaries, frames):
    """Concatenate observations, retaining repeated source frames across runs."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in summaries:
        if row["phase"] != "confirmation":
            continue
        samples = [r for r in frames[row["slot_id"]] if r["model_id"] == row["model_id"]]
        values = np.asarray([r["latency_ms"] for r in samples], dtype=float)
        if (len(values) != row["completed_count"] or not len(values)
                or np.any(~np.isfinite(values)) or np.any(values <= 0)
                or any(r["scene_id"] != row["scene_id"] for r in samples)
                or len({r["source_frame_id"] for r in samples}) != row["unique_source_count"]):
            raise ValueError(f"inference evidence differs from summary: {row['slot_id']}")
        grouped[(tuple(row["pair"]), row["mps_enabled"], row["model_id"])][row["cell"]].append((row, samples))
    result = {}
    for key, cells in grouped.items():
        if set(cells) != {"AA", "AB", "BA", "BB"}:
            raise ValueError(f"incomplete four-cell evidence: {key}")
        result[key] = {}
        for cell, runs in cells.items():
            if (len(runs) != 3 or {r["repetition"] for r, _ in runs} != {1, 2, 3}
                    or len({r["execution_id"] for r, _ in runs}) != 3
                    or len({(r["lidar_scene"], r["camera_scene"]) for r, _ in runs}) != 1):
                raise ValueError(f"expected three distinct matching executions: {key}/{cell}")
            runs.sort(key=lambda item: item[0]["repetition"])
            combined = [sample for _, samples in runs for sample in samples]
            result[key][cell] = {
                "values": [r["latency_ms"] for r in combined],
                "unique_source_count": len({r["source_frame_id"] for r in combined}),
                "execution_ids": [r["execution_id"] for r, _ in runs],
                "lidar_scene": runs[0][0]["lidar_scene"],
                "camera_scene": runs[0][0]["camera_scene"],
            }
    return result


def plot_pooled(pooled, destination):
    """Write one four-cell violin figure per pair, model, and MPS mode."""
    directory = destination / "plots/pooled_confirmation"
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / "pooled_inference_times.pdf"
    outputs, counts = [], []
    cells = ("AA", "AB", "BA", "BB")
    with PdfPages(pdf_path) as pdf:
        for (pair, mode, model), data in sorted(pooled.items()):
            figure, axis = plt.subplots(figsize=(10, 5.6))
            violin = axis.violinplot(
                [data[cell]["values"] for cell in cells], positions=range(4),
                widths=.75, showmedians=True, showextrema=True,
                quantiles=[[.25, .75]] * 4, points=250,
            )
            for body, color in zip(violin["bodies"], ("#4c78a8", "#e68a2e", "#59a14f", "#e15759")):
                body.set_facecolor(color)
                body.set_edgecolor("#333333")
                body.set_alpha(.65)
            for name in ("cmedians", "cquantiles", "cmins", "cmaxes", "cbars"):
                violin[name].set_color("#333333")
                violin[name].set_linewidth(1.1 if name == "cmedians" else .7)
            labels = []
            for cell in cells:
                record = data[cell]
                labels.append(f"{cell}\n{record['lidar_scene'].removeprefix('scene-')} / "
                              f"{record['camera_scene'].removeprefix('scene-')}\nn={len(record['values'])}")
                counts.append({"pair": "+".join(pair), "model_id": model, "mps_enabled": mode,
                               "cell": cell, "completed_count": len(record["values"]),
                               **{k: record[k] for k in ("lidar_scene", "camera_scene", "unique_source_count")},
                               "execution_ids": ";".join(record["execution_ids"])})
            axis.set_xticks(range(4), labels)
            axis.set_xlabel("Input scenes (LiDAR / camera)")
            axis.set_ylabel("Inference time (ms)")
            axis.set_title(f"{'+'.join(pair)} | {model} | MPS {'on' if mode else 'off'}")
            axis.grid(axis="y", alpha=.2)
            figure.tight_layout()
            path = directory / f"{'+'.join(pair)}-{int(mode)}-{model}.png"
            figure.savefig(path, dpi=180)
            pdf.savefig(figure)
            plt.close(figure)
            outputs.append(path.relative_to(destination).as_posix())
    write_csv(destination / "pooled_inference_counts.csv", counts)
    index = destination / "pooled_confirmation_violins.md"
    lines = ["# Pooled inference-time violin plots", "",
             "Each violin treats the three executions of one confirmation condition as one combined sample. "
             "Every completed non-warmup inference is included once, with no tail trimming. "
             "Repeated source frames across executions retain their separate latency observations. "
             "Models, MPS modes, and actual input-scene combinations remain separate. "
             "Inference time includes CUDA completion; external decode/preprocessing are excluded. "
             "These are pooled descriptive distributions, not estimates of between-execution uncertainty.", "",
             "[All plots as a multipage PDF](plots/pooled_confirmation/pooled_inference_times.pdf) · "
             "[Sample counts and source executions](pooled_inference_counts.csv)", ""]
    for path in outputs:
        lines.extend([f"## {Path(path).stem}", "", f"![Pooled inference times]({path})", ""])
    index.write_text("\n".join(lines))
    return {"index": str(index), "pdf": str(pdf_path), "plots": outputs,
            "conditions": len(counts), "inference_observations": sum(r["completed_count"] for r in counts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = read_json(args.manifest)
    destination = args.output_root / "analysis/input2-crossed/input-data"
    summaries = read_json(destination / "per_run_summaries.json")
    frames, sources = {}, {}
    for entry in manifest["executions"]:
        if entry["phase"] != "confirmation":
            continue
        if entry["status"] != "validated":
            raise ValueError(f"unvalidated confirmation slot: {entry['run_id']}")
        execution = Path(entry["artifact_directory"]).name
        directory = args.output_root / "analysis/runs" / execution / "input-data"
        provenance = read_json(directory / "analysis_provenance.json")
        if provenance["identity"]["evidence_hashes"] != entry["evidence"]["file_hashes"]:
            raise ValueError(f"source evidence identity differs: {execution}")
        for name in ("frames.json", "summary.json"):
            if sha256(directory / name) != provenance["outputs"][name]:
                raise ValueError(f"analyzed evidence hash differs: {execution}/{name}")
        source_summary = read_json(directory / "summary.json")
        for row in (r for r in summaries if r["phase"] == "confirmation" and r["slot_id"] == entry["run_id"]):
            original = next(r for r in source_summary["models"] if r["model_id"] == row["model_id"])
            if (row["execution_id"] != execution or any(row[k] != original[k] for k in original)
                    or any(row[k] != entry[k] for k in ("pair", "cell", "mps_enabled", "repetition", "lidar_scene", "camera_scene"))):
                raise ValueError(f"summary differs from source execution: {entry['run_id']}")
        frames[entry["run_id"]] = read_json(directory / "frames.json")
        sources[execution] = {name: provenance["outputs"][name] for name in ("frames.json", "summary.json")}
    pooled = pool_confirmation(summaries, frames)
    expected = {(tuple(pair), mode, model) for pair in manifest["study"]["pairs"]
                for mode in manifest["study"]["mps_modes"] for model in pair}
    if set(pooled) != expected:
        raise ValueError("incomplete confirmation model/pair/mode matrix")
    result = plot_pooled(pooled, destination)
    write_json(destination / "pooled_violin_provenance.json", {
        "manifest_sha256": sha256(args.manifest), "source_exports": sources,
        "plotter_sha256": sha256(__file__), "pooling": "concatenate all three executions without trimming or deduplication",
        **result,
    })
    print(result)


if __name__ == "__main__":
    main()
