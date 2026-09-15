#!/usr/bin/env python3
"""Build per-mode inference summaries and paired violin plots for input1."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "frame_metrics.csv"
CONDITIONS = {
    "clear": (0, "Clear", 0.0),
    "r7p5": (1, "7.5 mm/h", 7.5),
    "r15": (2, "15 mm/h", 15.0),
    "r25": (3, "25 mm/h", 25.0),
    "r50": (4, "50 mm/h", 50.0),
}
MODES = {"on": "mps", "off": "non_mps"}
COLORS = {"lidar": "#1f77b4", "image": "#ff7f0e"}
FIELDS = (
    "model_pair", "model", "modality", "condition", "condition_id",
    "rain_rate_mm_per_hour", "inference_time_range_p0_p99_ms", "p0_ms",
    "p99_ms", "max_ms", "min_ms", "mean_ms",
    "number_of_frames_executed", "median_ms", "p25_ms", "p75_ms",
)


def load_groups():
    """Read validated frame measurements and group inference times."""
    groups = defaultdict(list)
    pair_models = {}
    with SOURCE.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            assert row["run_valid"] == "True"
            mode, pair, condition, model = (
                row["mps_mode"], row["pair_id"], row["condition_id"],
                row["model_id"],
            )
            assert mode in MODES and condition in CONDITIONS
            modality = "lidar" if model == row["lidar_model"] else "image"
            assert model == row[
                "lidar_model" if modality == "lidar" else "camera_model"
            ]
            pair_models.setdefault(pair, {})[modality] = model
            groups[mode, pair, condition, modality].append(
                float(row["inference_e2e_ms"])
            )

    pairs = sorted(pair_models)
    for mode in MODES:
        for pair in pairs:
            assert set(pair_models[pair]) == set(COLORS)
            for condition in CONDITIONS:
                for modality in COLORS:
                    assert groups[mode, pair, condition, modality]
    return groups, pair_models


def summary_rows(groups, pair_models, mode):
    """Calculate the requested statistics for one MPS mode."""
    rows = []
    for pair in sorted(pair_models):
        for condition in sorted(CONDITIONS, key=lambda key: CONDITIONS[key][0]):
            for modality in ("lidar", "image"):
                values = np.asarray(
                    groups[mode, pair, condition, modality], dtype=float
                )
                p0, p25, median, p75, p99 = np.percentile(
                    values, [0, 25, 50, 75, 99]
                )
                rows.append({
                    "model_pair": pair,
                    "model": pair_models[pair][modality],
                    "modality": modality,
                    "condition": CONDITIONS[condition][1],
                    "condition_id": condition,
                    "rain_rate_mm_per_hour": CONDITIONS[condition][2],
                    "inference_time_range_p0_p99_ms": p99 - p0,
                    "p0_ms": p0,
                    "p99_ms": p99,
                    "max_ms": values.max(),
                    "min_ms": values.min(),
                    "mean_ms": values.mean(),
                    "number_of_frames_executed": len(values),
                    "median_ms": median,
                    "p25_ms": p25,
                    "p75_ms": p75,
                })
    return rows


def write_summary(rows, path):
    """Write stable, human-readable numeric values to CSV."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.3f}" if isinstance(value, np.floating) else value)
                for key, value in row.items()
            })


def plot_pair(groups, pair_models, mode, pair):
    """Plot side-by-side P1-P99 distributions across conditions."""
    conditions = sorted(CONDITIONS, key=lambda key: CONDITIONS[key][0])
    records = {"Condition": [], "Inference time (ms)": [], "Modality": []}
    for condition in conditions:
        for modality in ("lidar", "image"):
            values = groups[mode, pair, condition, modality]
            p1, p99 = np.percentile(values, [1, 99])
            values = [value for value in values if p1 <= value <= p99]
            assert values and p1 <= min(values) <= max(values) <= p99
            records["Condition"].extend([CONDITIONS[condition][1]] * len(values))
            records["Inference time (ms)"].extend(values)
            records["Modality"].extend([modality] * len(values))

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=records,
        x="Condition",
        y="Inference time (ms)",
        hue="Modality",
        order=[CONDITIONS[key][1] for key in conditions],
        hue_order=["lidar", "image"],
        palette=COLORS,
        cut=0,
        inner="quart",
        density_norm="width",
        common_norm=False,
        linewidth=1.1,
        legend=False,
        ax=axis,
    )
    axis.set_title(
        f"{pair_models[pair]['lidar']} + {pair_models[pair]['image']} | "
        f"{'MPS' if mode == 'on' else 'Non-MPS'} | P1-P99"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = ROOT / "violin_plots" / MODES[mode] / f"{pair}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_readme(pair_count, row_count):
    """Document scope and field semantics next to the generated artifacts."""
    (ROOT / "inference_summary_README.md").write_text(f"""# Input1 inference summary

The two requested tables contain {row_count} rows each: {pair_count} model pairs × 5 conditions × 2 models. All source runs passed the existing campaign validation, and `number_of_frames_executed` counts completed, profiler-matched frames.

- [`mps_summary.csv`](mps_summary.csv): MPS-enabled runs.
- [`non_mps_summary.csv`](non_mps_summary.csv): MPS-disabled runs.
- [`violin_plots/mps`](violin_plots/mps): one MPS plot per model pair.
- [`violin_plots/non_mps`](violin_plots/non_mps): one non-MPS plot per model pair.

Inference time is the validated `inference_e2e_ms` NVTX duration. `inference_time_range_p0_p99_ms` is `p99_ms - p0_ms`; `p0_ms` and `p99_ms` are also retained so the interval endpoints are explicit. Percentiles use NumPy's default linear interpolation. Times are milliseconds and rounded to three decimal places in the tables. Violin distributions are generated with Seaborn's `sns.violinplot`, filter each model-condition group to p1-p99, use blue for LiDAR and orange for image models, and show quartiles as inner lines. The tables retain all frames, including values outside p1-p99.
""", encoding="utf-8")


def main():
    groups, pair_models = load_groups()
    expected_rows = len(pair_models) * len(CONDITIONS) * len(COLORS)
    for mode, name in MODES.items():
        rows = summary_rows(groups, pair_models, mode)
        assert len(rows) == expected_rows
        write_summary(rows, ROOT / f"{name}_summary.csv")
        for pair in sorted(pair_models):
            plot_pair(groups, pair_models, mode, pair)
    write_readme(len(pair_models), expected_rows)
    print(f"Wrote 2 tables ({expected_rows} rows each) and "
          f"{len(pair_models) * len(MODES)} violin plots to {ROOT}")


if __name__ == "__main__":
    main()
