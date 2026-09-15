#!/usr/bin/env python3
"""Summarize LiDAR and object characteristics for selected nuScenes scenes."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml


GRIDS = {
    "centerpoint": (
        np.array([-54.0, -54.0, -5.0]),
        np.array([54.0, 54.0, 3.0]),
        np.array([0.075, 0.075, 0.2]),
    ),
    "pointpillars": (
        np.array([-50.0, -50.0, -5.0]),
        np.array([50.0, 50.0, 3.0]),
        np.array([0.25, 0.25, 8.0]),
    ),
}
THREEDSSD_RANGE = (
    np.array([0.0, -40.0, -5.0]),
    np.array([70.0, 40.0, 3.0]),
)
PLOTS = {
    "centerpoint_voxel_count": ("voxel_count", "Occupied voxel count"),
    "pointpillars_voxel_count": ("pillar_count", "Occupied pillar count"),
    "raw_point_count": ("point_count", "Raw LiDAR point count"),
    "pointpillars_point_density_per_m3": (
        "point_density", "LiDAR point density (points/m³)"),
    "object_count": ("object_count", "Annotated object count"),
}


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream objects from a top-level JSON array using the standard library."""
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open() as stream:
        while True:
            chunk = stream.read(chunk_size)
            buffer += chunk
            position = 0
            while True:
                while (position < len(buffer)
                       and buffer[position] in " \t\r\n,[]"):
                    position += 1
                if position == len(buffer):
                    buffer = ""
                    break
                try:
                    value, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    buffer = buffer[position:]
                    break
                yield value
            if not chunk:
                if buffer.strip(" \t\r\n,[]"):
                    raise ValueError(f"incomplete JSON array in {path}")
                return


def discover_scenes(input_root):
    """Return unique source scenes declared by experiment run configs."""
    scenes = {}
    for path in sorted(input_root.glob("*/config.yaml")):
        config = yaml.safe_load(path.read_text())
        variation = config.get("input_variation", {})
        replay = config.get("replay", {})
        name = variation.get("source_scene_name")
        token = replay.get("scene_token")
        if token:
            if token in scenes and scenes[token] != name:
                raise ValueError(f"scene token {token} has conflicting names")
            scenes[token] = name
    if not scenes:
        raise ValueError(f"no scene tokens found under {input_root}")
    return scenes


def select_scenes(metadata_root, selectors=None, input_root=None):
    """Resolve scene names or tokens against the dataset's scene table."""
    rows = json.loads((metadata_root / "scene.json").read_text())
    available = {row["token"]: row["name"] for row in rows}
    if input_root is not None:
        selectors = list(discover_scenes(input_root))
    if selectors is None:
        return available
    lookup = {row["name"]: row["token"] for row in rows}
    lookup.update({token: token for token in available})
    unknown = sorted(set(selectors) - lookup.keys())
    if unknown:
        raise ValueError(f"unknown scenes in {metadata_root}: {', '.join(unknown)}")
    return {lookup[value]: available[lookup[value]] for value in selectors}


def load_metadata(metadata_root, scene_tokens):
    """Load only metadata needed by the selected scenes."""
    scenes = {
        row["token"]: row
        for row in json.loads((metadata_root / "scene.json").read_text())
        if row["token"] in scene_tokens
    }
    samples = {
        row["token"]: row
        for row in json.loads((metadata_root / "sample.json").read_text())
        if row["scene_token"] in scene_tokens
    }
    sample_tokens = set(samples)
    lidar = []
    for row in iter_json_array(metadata_root / "sample_data.json"):
        if (row["sample_token"] in sample_tokens
                and "LIDAR_TOP" in row["filename"]):
            lidar.append(row)
    annotations = {token: {} for token in sample_tokens}
    for row in iter_json_array(metadata_root / "sample_annotation.json"):
        if row["sample_token"] in sample_tokens:
            annotations[row["sample_token"]][row["instance_token"]] = (
                np.asarray(row["translation"][:2], dtype=float))
    return scenes, samples, sorted(lidar, key=lambda row: row["timestamp"]), annotations


def occupied_voxels(points, lower, upper, size):
    """Count occupied cells and in-range points for a voxel grid."""
    xyz = points[:, :3]
    mask = np.all((xyz >= lower) & (xyz < upper), axis=1)
    cells = ((xyz[mask] - lower) / size).astype(np.int32)
    if not len(cells):
        return 0, 0
    shape = ((upper - lower) / size).astype(np.int64)
    ids = (cells[:, 0].astype(np.int64) * shape[1] * shape[2]
           + cells[:, 1].astype(np.int64) * shape[2] + cells[:, 2])
    return int(np.unique(ids).size), int(mask.sum())


def object_metrics(sample_token, timestamp, samples, annotations):
    """Interpolate object centers for a non-keyframe LiDAR timestamp."""
    current = samples[sample_token]
    centers = annotations[sample_token].copy()
    previous = samples.get(current["prev"])
    if previous and timestamp < current["timestamp"]:
        span = current["timestamp"] - previous["timestamp"]
        weight = (timestamp - previous["timestamp"]) / span
        before = annotations[previous["token"]]
        for instance in centers.keys() & before.keys():
            centers[instance] = (
                before[instance] + weight * (centers[instance] - before[instance]))
    count = len(centers)
    if count < 2:
        return count, None
    xy = np.asarray(list(centers.values()))
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return count, float(distances.min(axis=1).mean())


def frame_metrics(row, scene_name, dataset_root, samples, annotations):
    """Calculate all requested metrics for one LiDAR frame."""
    path = dataset_root / row["filename"]
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"invalid nuScenes point cloud: {path}")
    points = points.reshape(-1, 5)
    result = {
        "scene": scene_name,
        "scene_token": samples[row["sample_token"]]["scene_token"],
        "lidar_token": row["token"],
        "sample_token": row["sample_token"],
        "timestamp": row["timestamp"],
        "is_key_frame": row["is_key_frame"],
        "raw_point_count": len(points),
    }
    for model, (lower, upper, size) in GRIDS.items():
        voxel_count, point_count = occupied_voxels(points, lower, upper, size)
        grid_cells = int(np.prod((upper - lower) / size))
        volume = float(np.prod(upper - lower))
        result.update({
            f"{model}_point_count": point_count,
            f"{model}_point_density_per_m3": point_count / volume,
            f"{model}_voxel_count": voxel_count,
            f"{model}_voxel_density_pct": 100.0 * voxel_count / grid_cells,
            f"{model}_points_per_voxel": point_count / voxel_count
            if voxel_count else None,
        })
    lower, upper = THREEDSSD_RANGE
    in_range = np.all((points[:, :3] >= lower) & (points[:, :3] < upper), axis=1)
    result["threedssd_point_count"] = int(in_range.sum())
    result["threedssd_point_density_per_m3"] = (
        result["threedssd_point_count"] / float(np.prod(upper - lower)))
    count, mean_nn = object_metrics(
        row["sample_token"], row["timestamp"], samples, annotations)
    result["object_count"] = count
    result["object_mean_nearest_neighbor_2d_m"] = mean_nn
    return result


def summarize(values):
    """Return distribution and relative-variation statistics."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict.fromkeys((
            "min", "max", "range", "mean", "std", "cov", "p25", "median",
            "p75", "iqr", "variation_iqr_over_p50", "p99",
            "p99_minus_p50_over_p50",
        ))
    p25, median, p75, p99 = np.percentile(values, [25, 50, 75, 99])
    mean = values.mean()
    std = values.std()
    spread = values.max() - values.min()
    iqr = p75 - p25
    return {
        "min": values.min(), "max": values.max(), "range": spread,
        "mean": mean, "std": std, "cov": std / mean if mean else None,
        "p25": p25, "median": median, "p75": p75, "iqr": iqr,
        "variation_iqr_over_p50": iqr / median if median else None,
        "p99": p99,
        "p99_minus_p50_over_p50": (p99 - median) / median
        if median else None,
    }


def write_csv(path, rows):
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(frame_csv, output_dir):
    """Write one scene-level violin plot for each requested metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    with frame_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    scenes = sorted({row["scene"] for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, (filename, ylabel) in PLOTS.items():
        records = {
            "Scene": [row["scene"] for row in rows],
            ylabel: [float(row[metric]) for row in rows],
        }
        figure, axis = plt.subplots(figsize=(12, 6))
        sns.violinplot(
            data=records, x="Scene", y=ylabel, order=scenes, cut=0,
            inner="quart", density_norm="width", color="#4c78a8", ax=axis,
        )
        axis.set_xlabel("Scene name")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} distribution by scene")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{filename}_violin.png", dpi=180)
        plt.close(figure)


def self_check():
    """Exercise scene selection, missing statistics, and frame metrics."""
    stats = summarize([1, 2, 3, 4])
    assert stats["p25"] == 1.75 and stats["median"] == 2.5
    assert stats["p75"] == 3.25 and stats["variation_iqr_over_p50"] == 0.6
    points = np.array([[0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0]])
    assert occupied_voxels(
        points, np.zeros(3), np.ones(3), np.ones(3))[0] == 1
    assert all(value is None for value in summarize([]).values())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "array.json"
        path.write_text('[{"value": 1}, {"value": 2}]')
        assert [row["value"] for row in iter_json_array(path, 3)] == [1, 2]
        (root / "scene.json").write_text(json.dumps([
            {"token": "token-a", "name": "scene-a"},
            {"token": "token-b", "name": "scene-b"},
        ]))
        assert select_scenes(root, ["scene-b"]) == {"token-b": "scene-b"}
        assert select_scenes(root, ["scene-a", "token-b", "token-a"]) == {
            "token-a": "scene-a", "token-b": "scene-b"}
        assert len(select_scenes(root)) == 2
        try:
            select_scenes(root, ["unknown"])
        except ValueError as error:
            assert "unknown scenes" in str(error)
        else:
            raise AssertionError("unknown scene accepted")
        run = root / "runs" / "example"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text("replay:\n  scene_token: token-b\n")
        assert select_scenes(root, input_root=root / "runs") == {
            "token-b": "scene-b"}
        (root / "sample.json").write_text(json.dumps([
            {"token": "sample-b", "scene_token": "token-b",
             "timestamp": 10, "prev": ""},
        ]))
        row = {"token": "lidar-b", "sample_token": "sample-b",
               "timestamp": 10, "is_key_frame": True,
               "filename": "LIDAR_TOP.bin"}
        (root / "sample_data.json").write_text(json.dumps([row]))
        (root / "sample_annotation.json").write_text("[]")
        points.astype(np.float32).tofile(root / row["filename"])
        scenes, samples, lidar, annotations = load_metadata(root, {"token-b"})
        assert set(scenes) == {"token-b"} and len(lidar) == 1
        metrics = frame_metrics(lidar[0], "scene-b", root, samples, annotations)
        assert metrics["scene"] == "scene-b" and metrics["raw_point_count"] == 2
        assert metrics["object_count"] == 0
        assert metrics["object_mean_nearest_neighbor_2d_m"] is None
        main([
            "--scenes", "scene-b", "token-b", "--metadata-root", str(root),
            "--dataset-root", str(root), "--output-dir", str(root / "output"),
            "--no-plots",
        ])
        with (root / "output" / "frame_metrics.csv").open() as stream:
            frames = list(csv.DictReader(stream))
        assert len(frames) == 1 and frames[0]["scene_token"] == "token-b"
        with (root / "output" / "distribution_summary.csv").open() as stream:
            summaries = {row["metric"]: row for row in csv.DictReader(stream)}
        assert summaries["raw_point_count"]["mean"] == "2.0"
        assert summaries["object_mean_nearest_neighbor_2d_m"]["valid_count"] == "0"


def main(argv=None):
    """Analyze selected scenes independently of any experiment campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenes", nargs="+", metavar="NAME_OR_TOKEN",
                           help="One or more nuScenes scene names or tokens")
    selection.add_argument("--all-scenes", action="store_true",
                           help="Analyze every scene in the metadata directory")
    selection.add_argument("--input-root", type=Path,
                           help="Alternatively discover scenes from */config.yaml")
    parser.add_argument("--metadata-root", type=Path,
                        help="Directory containing nuScenes JSON tables")
    parser.add_argument("--dataset-root", type=Path,
                        help="Directory containing the point-cloud paths in metadata")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "analysis_outputs/close_loop_perf/scene_metrics"))
    parser.add_argument("--plot-frame-csv", type=Path)
    parser.add_argument("--plot-dir", type=Path,
                        help="Plot directory (default: OUTPUT_DIR/plots)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Write CSV metrics without generating plots")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return
    if args.plot_frame_csv:
        plot_metrics(args.plot_frame_csv, args.plot_dir or args.output_dir / "plots")
        return

    if not (args.scenes or args.all_scenes or args.input_root):
        parser.error("choose --scenes, --all-scenes, or --input-root")
    if args.metadata_root is None or args.dataset_root is None:
        parser.error("--metadata-root and --dataset-root are required for analysis")
    try:
        selected = select_scenes(args.metadata_root, args.scenes, args.input_root)
    except ValueError as error:
        parser.error(str(error))
    scenes, samples, lidar, annotations = load_metadata(
        args.metadata_root, set(selected))
    populated = {samples[row["sample_token"]]["scene_token"] for row in lidar}
    missing = set(selected) - populated
    if missing or not selected:
        parser.error("no LiDAR frames found for scenes: " + ", ".join(
            selected[token] for token in sorted(missing)))
    frame_rows = []
    for index, row in enumerate(lidar, 1):
        scene_token = samples[row["sample_token"]]["scene_token"]
        frame_rows.append(frame_metrics(
            row, selected[scene_token], args.dataset_root, samples, annotations))
        if index % 100 == 0:
            print(f"processed {index}/{len(lidar)} frames", flush=True)

    write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    identifiers = set(frame_rows[0]) - {
        "scene", "scene_token", "lidar_token", "sample_token", "timestamp",
        "is_key_frame",
    }
    summary_rows = []
    for token, name in sorted(selected.items(), key=lambda item: item[1]):
        scene_rows = [row for row in frame_rows if row["scene_token"] == token]
        for metric in sorted(identifiers):
            values = [row[metric] for row in scene_rows if row[metric] is not None]
            summary_rows.append({
                "scene": name,
                "scene_token": token,
                "frame_count": len(scene_rows),
                "metric": metric,
                "valid_count": len(values),
                "missing_count": len(scene_rows) - len(values),
                **summarize(values),
            })
    write_csv(args.output_dir / "distribution_summary.csv", summary_rows)
    if not args.no_plots:
        plot_metrics(args.output_dir / "frame_metrics.csv",
                     args.plot_dir or args.output_dir / "plots")
    print(f"wrote {len(frame_rows)} frames from {len(scenes)} scenes to "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
