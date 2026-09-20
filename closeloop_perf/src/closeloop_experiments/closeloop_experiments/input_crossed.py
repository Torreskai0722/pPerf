"""Orchestrate the frozen, common-window input2 crossed experiment."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from closeloop_analyzer.input_data.crossed_evidence import inspect_execution
from closeloop_testbed.resource_control import GPUClockLock, detect_cpu_topology
from . import corrected_input_variation as campaign
from .config import load_run_config, schema_v2_config
from .controlled_bags import (
    WINDOW_NS, TOPICS, construct_bag, inspect_source,
    sha256, validate_bag, write_json,
)
from .runner import ExperimentRunner, detect_gpu_hardware


SCENES = ("scene-0770", "scene-0398", "scene-0184", "scene-0245")
PAIRS = (("3dssd", "yolov3"), ("centerpoint", "dino"),
         ("centerpoint", "yolov3"), ("pointpillars", "vit-upernet"),
         ("pointpillars", "mask-rcnn"))
CELL_ORDERS = (("AA", "AB", "BB", "BA"), ("BA", "BB", "AB", "AA"),
               ("AB", "AA", "BA", "BB"))


def load_study(path, check_paths=True):
    """Load the existing model inventory with the exact crossed contract."""
    study = campaign.load_study(path, check_paths=check_paths)
    data = study["data"]
    if (tuple(map(tuple, data.get("pairs", []))) != PAIRS
            or tuple(data["conditions"]) != SCENES
            or set(data["models"]) != {m for p in PAIRS for m in p}
            or data["mps_modes"] != [False, True]
            or data.get("common_window_ns") != WINDOW_NS
            or data.get("confirmation_repetitions") != 3
            or data.get("replay_passes") != 1):
        raise ValueError("input2-crossed requires the authored five pairs, seven models, four scenes, two modes, frozen window, and 1/3 executions")
    return study


def block_id(pair, mode):
    """Name one authored pair/mode block."""
    return f"{'+'.join(pair)}:mps-{'on' if mode else 'off'}"


def planned_runs(study, phase, selection=None):
    """Keep execution repetitions independent of the single replay pass."""
    result = []
    models = list(study["data"]["models"])

    def add(run_id, pair, mode, lidar, image, repetition=1, cell=None):
        names = list(pair)
        result.append({"run_id": run_id, "phase": phase,
                       "pair": list(pair) if len(pair) == 2 else None,
                       "models": names, "mps_enabled": mode,
                       "lidar_scene": lidar, "camera_scene": image,
                       "repetition": repetition, "replay_passes": 1,
                       "cell": cell, "status": "planned", "reuse": [],
                       "selection_evidence": phase == "screening"})

    if phase == "isolated":
        for mode in (False, True):
            for index, model in enumerate(models):
                for scene in SCENES:
                    add(f"i2-i{index}{int(mode)}-{scene[-4:]}", (model,), mode, scene, scene)
    elif phase == "screening":
        for mode in (False, True):
            for scene in SCENES:
                for index, pair in enumerate(PAIRS):
                    add(f"i2-s{index}{int(mode)}-{scene[-4:]}", pair, mode, scene, scene)
    elif phase == "confirmation":
        if selection is None:
            raise ValueError("confirmation requires frozen selection.json")
        blocks = [(index, pair, mode) for mode in (False, True) for index, pair in enumerate(PAIRS)]
        for repetition, order in enumerate(CELL_ORDERS, 1):
            rotated = blocks[repetition - 1:] + blocks[:repetition - 1]
            for index, pair, mode in rotated:
                selected = selection["selections"][block_id(pair, mode)]
                scenes = {"A": selected["A"], "B": selected["B"]}
                for cell in order:
                    add(f"i2-c{index}{int(mode)}-{cell.lower()}-{repetition}", pair, mode,
                        scenes[cell[0]], scenes[cell[1]], repetition, cell)
    else:
        raise ValueError(f"unknown phase {phase}")
    return result


def selection_record(path, manifest):
    """Require an immutable, complete selection backed by this screening."""
    path = Path(path).resolve()
    selection = json.loads(path.read_text())
    if (selection.get("schema") != "input2_crossed_selection_v1"
            or selection.get("study_sha256") != manifest["study_sha256"]
            or set(selection.get("selections", {})) != {block_id(p, m) for p in PAIRS for m in (False, True)}):
        raise ValueError("selection identity or ten-block inventory differs")
    for key, row in selection["selections"].items():
        if row["A"] == row["B"] or row["A"] not in SCENES or row["B"] not in SCENES:
            raise ValueError(f"selection requires distinct authored scenes: {key}")
        for run_id, expected in row["source_run_hashes"].items():
            entry = next(r for r in manifest["executions"] if r["run_id"] == run_id)
            if sha256(Path(entry["artifact_directory"]) / "run_manifest.json") != expected:
                raise ValueError(f"selection source run changed: {run_id}")
            for filename, digest in row["source_evidence_hashes"][run_id].items():
                if sha256(Path(entry["artifact_directory"]) / filename) != digest:
                    raise ValueError(f"selection source evidence changed: {run_id}/{filename}")
    reference = {"path": str(path), "sha256": sha256(path)}
    if manifest.get("selection") and manifest["selection"] != reference:
        raise ValueError("frozen selection differs; refusing to reselect")
    return selection, reference


def run_config(study, entry, bag_manifest, ordinal):
    """Reuse primary profiling/model setup and assign resources by modality."""
    planned = {"condition_id": entry["lidar_scene"],
               "models": sorted(entry["models"], key=lambda m: study["data"]["models"][m]["modality"]),
               "mps_enabled": entry["mps_enabled"]}
    config = campaign._run_config(study, entry["run_id"], planned, ordinal)
    config["run"]["experiment"] = "controlled own-input and co-runner input sensitivity"
    config["replay"].update(controlled_bag_manifest=str(bag_manifest),
                              controlled_bag_manifest_sha256=sha256(bag_manifest),
                              bag_directory=str(Path(bag_manifest).parent / "bag"), repeat_count=1)
    for model in config["models"]:
        index = 0 if model["modality"] == "image" else 1
        model["cpu_affinity"] = list(range(index * 6, (index + 1) * 6))
        model["launch_offset_seconds"] = index
    config["input_variation"].update(replicate=entry["repetition"],
                                      tenancy="single" if len(entry["models"]) == 1 else "two_model")
    config["input_variation"]["corruption"].update(dataset_manifest=str(bag_manifest),
                                                          dataset_manifest_sha256=sha256(bag_manifest))
    return schema_v2_config(config)


def paths(artifact_root):
    """Return the experiment's generated configuration directory and ledger."""
    root = Path(artifact_root).resolve() / "generated_configs/input2-crossed"
    return root, root / "experiment_manifest.json"


def assert_unchanged_streams(bags):
    """Require bit-identical untouched streams across all constructed cells."""
    streams = {}
    for path in bags.values():
        manifest = json.loads(Path(path).read_text())
        for modality, scene in manifest["input_scenes"].items():
            key = (modality, scene["scene_id"])
            digest = manifest["validation"]["stream_sha256"][TOPICS[modality]]
            if key in streams and streams[key] != digest:
                raise ValueError(f"unchanged controlled stream differs: {key}")
            streams[key] = digest


def prepare(study, artifact_root, bag_root, phase, selection_path=None):
    """Prepare only missing bags/configs and audit historical reuse read-only."""
    root, manifest_path = paths(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    study["paths"]["output_root"] = Path(artifact_root).resolve() / "runs"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["study_sha256"] != study["sha256"]:
            raise ValueError("immutable study changed")
    else:
        sources = {scene: inspect_source(scene, condition) for scene, condition in study["data"]["conditions"].items()}
        if min(s["duration_ns"] for s in sources.values()) != WINDOW_NS:
            raise ValueError("shortest verified source duration differs from frozen 19.250761 seconds")
        manifest = {"schema": "input2_crossed_experiment_v1", "study_id": "input2-crossed",
                    "study_path": str(study["source"]), "study_sha256": study["sha256"],
                    "study": study["data"], "sources": sources, "window_ns": WINDOW_NS,
                    "bag_root": str(Path(bag_root).resolve()), "bags": {}, "executions": [],
                    "hardware": {"gpu_hardware": detect_gpu_hardware(0), "cpu_topology": detect_cpu_topology()},
                    "historical_execution_order": [], "execution_order": [], "selection": None}
        write_json(manifest_path, manifest)
    if str(Path(bag_root).resolve()) != manifest["bag_root"]:
        raise ValueError("controlled bag root differs from prepared experiment")
    manifest["scene_order"] = list(SCENES)
    selection = None
    if phase == "confirmation":
        selection, manifest["selection"] = selection_record(selection_path, manifest)
    requested = (planned_runs(study, "isolated") + planned_runs(study, "screening")
                 if phase == "screening" else planned_runs(study, phase, selection))
    combinations = list(dict.fromkeys((e["lidar_scene"], e["camera_scene"]) for e in requested))
    for lidar, image in combinations:
        path = construct_bag({"lidar": manifest["sources"][lidar], "image": manifest["sources"][image]}, bag_root)
        manifest["bags"][f"{lidar}+{image}"] = str(path)
        write_json(manifest_path, manifest)
    assert_unchanged_streams(manifest["bags"])
    existing = {entry["run_id"]: entry for entry in manifest["executions"]}
    used = {e.get("artifact_directory") for e in manifest["executions"] if e["phase"] == "confirmation" and e["status"] == "validated"}
    for proposed in requested:
        entry = existing.get(proposed["run_id"], proposed)
        bag_path = manifest["bags"][f"{entry['lidar_scene']}+{entry['camera_scene']}"]
        config = run_config(study, entry, bag_path, len(existing) if entry["run_id"] not in existing else entry["planned_order"])
        config_path = root / (entry["run_id"] + ".yaml")
        content = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists() and config_path.read_text() != content:
            raise ValueError(f"immutable generated config differs: {config_path}")
        if not config_path.exists():
            config_path.write_text(content)
            config_path.chmod(0o444)
        load_run_config(config_path, check_paths=False, artifact_root=artifact_root)
        if entry["run_id"] in existing:
            continue
        entry.update(config_path=str(config_path), config_sha256=sha256(config_path),
                     model_hashes={m: {k: v for k, v in study["data"]["models"][m].items() if k.endswith("sha256")}
                                   for m in entry["models"]},
                     controlled_bag_manifest=bag_path, planned_order=len(existing),
                     artifact_directory=str(study["paths"]["output_root"] / entry["run_id"]))
        candidates = []
        if entry["phase"] == "screening":
            candidates.append(study["paths"]["output_root"] / f"{int(entry['mps_enabled'])}-{entry['lidar_scene']}-{'-'.join(entry['pair'])}")
        elif entry["phase"] == "confirmation":
            candidates.extend(Path(e["artifact_directory"]) for e in manifest["executions"] if
                              e["phase"] == "screening" and e["pair"] == entry["pair"] and
                              e["mps_enabled"] == entry["mps_enabled"] and
                              (e["lidar_scene"], e["camera_scene"]) == (entry["lidar_scene"], entry["camera_scene"]))
        for candidate in candidates:
            if not candidate.exists():
                entry["reuse"].append({"path": str(candidate), "accepted": False, "reasons": ["candidate does not exist"]})
                continue
            evidence = inspect_execution(candidate, config, manifest["hardware"])
            if str(candidate) in used:
                evidence["accepted"] = False
                evidence["reasons"].append("execution already counted for this confirmation matrix")
            entry["reuse"].append({"path": str(candidate), **evidence})
            if entry["phase"] == "screening":
                manifest["historical_execution_order"].append({"path": str(candidate),
                    "started_at": evidence.get("started_at"), "finished_at": evidence.get("finished_at"),
                    "reused": evidence["accepted"]})
            if evidence["accepted"]:
                origin = next((e for e in manifest["executions"] if e["artifact_directory"] == str(candidate)), {})
                entry.update(status="validated", artifact_directory=str(candidate), evidence=evidence,
                             selection_evidence=True, reused_execution=True,
                             source_execution_id=candidate.name,
                             execution_order=origin.get("execution_order"),
                             started_at=evidence.get("started_at"), finished_at=evidence.get("finished_at"))
                used.add(str(candidate))
                break
        manifest["executions"].append(entry)
        existing[entry["run_id"]] = entry
        manifest["historical_execution_order"].sort(key=lambda e: e.get("started_at") or "")
        write_json(manifest_path, manifest)
    return manifest


def run(study, artifact_root, phase, selection_path=None, run_ids=(), limit=None, dry_run=False, retry_blocked=False):
    """Execute missing slots in planned order, preserving all failed attempts."""
    _root, manifest_path = paths(artifact_root)
    manifest = json.loads(manifest_path.read_text())
    if manifest["study_sha256"] != study["sha256"]:
        raise ValueError("immutable study changed")
    if phase == "confirmation":
        selection_record(selection_path, manifest)
    actual_hardware = {"gpu_hardware": detect_gpu_hardware(0), "cpu_topology": detect_cpu_topology()}
    if actual_hardware != manifest["hardware"]:
        raise ValueError("study hardware changed")
    known = {e["run_id"] for e in manifest["executions"] if e["phase"] == phase}
    if set(run_ids) - known:
        raise ValueError("requested run ID is absent from prepared phase")
    completed = 0
    for entry in manifest["executions"]:
        if entry["phase"] != phase or (run_ids and entry["run_id"] not in run_ids):
            continue
        path = Path(entry["config_path"])
        if sha256(path) != entry["config_sha256"]:
            raise ValueError(f"generated config changed: {path}")
        config = load_run_config(path, artifact_root=str(artifact_root))
        if Path(entry["artifact_directory"]).exists():
            evidence = inspect_execution(entry["artifact_directory"], config.data, manifest["hardware"])
            previous = entry.get("evidence", {})
            if previous.get("accepted") and evidence.get("file_hashes") != previous.get("file_hashes"):
                evidence["accepted"] = False
                evidence["reasons"].append("validated execution evidence changed since acceptance")
            entry["evidence"] = evidence
            entry["status"] = "validated" if evidence["accepted"] else "blocked"
            if not evidence["accepted"]:
                entry["blocker"] = "; ".join(evidence["reasons"])
            write_json(manifest_path, manifest)
            if evidence["accepted"] or not retry_blocked or dry_run:
                continue
            if limit is not None and completed >= limit:
                break
            # Failed attempts remain at their original paths with their exact
            # configurations. A new execution gets a new ID and config file.
            attempts = entry.setdefault("attempts", [])
            attempts.append({k: copy.deepcopy(entry[k]) for k in (
                "artifact_directory", "config_path", "config_sha256", "evidence", "blocker",
                "execution_order", "started_at", "finished_at") if k in entry})
            data = copy.deepcopy(config.data)
            data["run"]["id"] = f"{entry['run_id']}-a{len(attempts) + 1}"
            path = Path(entry["config_path"]).parent / (data["run"]["id"] + ".yaml")
            with path.open("x") as output:
                output.write(yaml.safe_dump(data, sort_keys=False))
            path.chmod(0o444)
            config = load_run_config(path, artifact_root=str(artifact_root))
            entry.update(config_path=str(path), config_sha256=sha256(path),
                         artifact_directory=str(config.run_directory))
            entry.pop("blocker", None)
        if limit is not None and completed >= limit:
            break
        if dry_run:
            continue  # Planning must not create runner directories that impede resume.
        entry["status"] = "running"
        entry["execution_order"] = len(manifest["execution_order"])
        entry["started_at"] = datetime.now(timezone.utc).isoformat()
        manifest["execution_order"].append(config.data["run"]["id"])
        write_json(manifest_path, manifest)
        control = GPUClockLock(config.data["gpu"]["index"], study["data"]["graphics_clock_mhz"], study["data"]["memory_clock_mhz"])
        try:
            try:
                with control:
                    ExperimentRunner(config).run()
            finally:
                if config.run_directory.exists():
                    write_json(config.run_directory / "input_variation_clock_control.json", {
                        "requested_graphics_clock_mhz": study["data"]["graphics_clock_mhz"],
                        "requested_memory_clock_mhz": study["data"]["memory_clock_mhz"],
                        "control": control.evidence,
                    })
            campaign.archive_profiles(config.run_directory)
            evidence = inspect_execution(config.run_directory, config.data, manifest["hardware"])
            entry.update(evidence=evidence, status="validated" if evidence["accepted"] else "blocked")
            if not evidence["accepted"]:
                entry["blocker"] = "; ".join(evidence["reasons"])
        except Exception as exc:
            entry.update(status="blocked", blocker=f"{type(exc).__name__}: {exc}")
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(json.dumps({"run_id": entry["run_id"], "status": entry["status"], "blocker": entry.get("blocker")}), flush=True)
        completed += 1
    return {"phase": phase, "counts": {state: sum(e["phase"] == phase and e["status"] == state for e in manifest["executions"])
                                          for state in ("planned", "running", "validated", "blocked")}}


def main(argv=None):
    """Implement the public input-data prepare/run phases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "prepare", "validate-bags", "run", "run-isolated"))
    parser.add_argument("study")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--bag-root")
    parser.add_argument("--phase", choices=("screening", "confirmation"), default="screening")
    parser.add_argument("--selection")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-blocked", action="store_true")
    args = parser.parse_args(argv)
    study = load_study(args.study)
    if args.command == "plan":
        result = {phase: len(planned_runs(study, phase)) for phase in ("isolated", "screening")}
        result["confirmation"] = 120
    elif args.command == "prepare":
        if not args.bag_root or (args.phase == "confirmation" and not args.selection):
            parser.error("prepare requires --bag-root; confirmation also requires --selection")
        manifest = prepare(study, args.artifact_root, args.bag_root, args.phase, args.selection)
        result = {"manifest": str(paths(args.artifact_root)[1]), "execution_slots": len(manifest["executions"]), "bags": len(manifest["bags"])}
    elif args.command == "validate-bags":
        manifest = json.loads(paths(args.artifact_root)[1].read_text())
        result = {key: validate_bag(path) for key, path in manifest["bags"].items()}
        assert_unchanged_streams(manifest["bags"])
    else:
        result = run(study, args.artifact_root, "isolated" if args.command == "run-isolated" else args.phase,
                     args.selection, args.run_id, args.limit, args.dry_run, args.retry_blocked)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("counts", {}).get("blocked") else 0
