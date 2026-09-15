"""Join relay publish records to model callback-entry records."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _key(record, timestamp_field):
    return (
        record.get("segment_id"),
        record.get("input_topic"),
        record.get(timestamp_field),
        record.get("timestamp_occurrence", 0),
    )


def analyze_communication(config, run_directory, output_directory=None):
    """Write per-message relay-to-callback latency and loss evidence."""
    run_directory = Path(run_directory)
    output_directory = Path(output_directory or run_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    relay_path = run_directory / "communication_relay.jsonl"
    if not relay_path.is_file():
        return None
    relay_records = _jsonl(relay_path)
    models_by_topic = defaultdict(list)
    callbacks = {}
    callback_records = []
    for model in config["models"]:
        models_by_topic[model["input_topic"]].append(model["id"])
        path = run_directory / f"model_{model['id']}_inputs.jsonl"
        if not path.is_file():
            continue
        for record in _jsonl(path):
            record["model_id"] = model["id"]
            callbacks[(model["id"],) + _key(
                record, "ros_header_timestamp_ns"
            )] = record
            callback_records.append(record)

    rows = []
    matched_callback_ids = set()
    counters = Counter()
    per_scene = defaultdict(Counter)
    for relay_index, relay in enumerate(relay_records):
        counters["relay_messages"] += 1
        counters["relay_duplicates"] += int(relay.get("duplicate", False))
        counters["relay_out_of_order"] += int(
            relay.get("out_of_order", False)
        )
        publish_call_ns = (
            relay["relay_post_publish_monotonic_ns"]
            - relay["relay_pre_publish_monotonic_ns"]
        )
        for model_id in models_by_topic.get(relay["input_topic"], []):
            callback = callbacks.get(
                (model_id,) + _key(
                    relay, "original_source_timestamp_ns"
                )
            )
            row = {
                **relay,
                "model_id": model_id,
                "relay_record_index": relay_index,
                "relay_publish_call_ns": publish_call_ns,
                "matched": callback is not None,
                "model_callback_entry_monotonic_ns": None,
                "communication_latency_ns": None,
            }
            scene = per_scene[(
                relay.get("scene_index"), relay.get("scene_token"), model_id,
                relay.get("input_topic"),
            )]
            scene["relay_messages"] += 1
            if callback is None:
                counters["dropped_or_overwritten"] += 1
                scene["dropped_or_overwritten"] += 1
            else:
                entry_ns = callback["model_callback_entry_monotonic_ns"]
                previous_exit_ns = callback.get(
                    "previous_callback_exit_monotonic_ns"
                )
                pre_publish_ns = relay["relay_pre_publish_monotonic_ns"]
                busy_residual_ns = max(
                    0,
                    (previous_exit_ns or pre_publish_ns) - pre_publish_ns,
                )
                row["model_callback_entry_monotonic_ns"] = entry_ns
                row["model_callback_exit_monotonic_ns"] = callback.get(
                    "model_callback_exit_monotonic_ns"
                )
                row["previous_callback_exit_monotonic_ns"] = previous_exit_ns
                row["previous_callback_busy_at_relay"] = busy_residual_ns > 0
                row["previous_callback_busy_residual_ns"] = busy_residual_ns
                row["model_message_order"] = callback.get("message_order")
                row["model_input_id"] = callback.get("input_id")
                row["communication_latency_ns"] = (
                    entry_ns - pre_publish_ns
                )
                row["post_busy_communication_latency_ns"] = (
                    row["communication_latency_ns"] - busy_residual_ns
                )
                for field in (
                    "decode_start_monotonic_ns", "decode_end_monotonic_ns",
                    "model_pipeline_start_monotonic_ns",
                    "model_pipeline_end_monotonic_ns", "completed",
                ):
                    row[field] = callback.get(field)
                counters["matched"] += 1
                scene["matched"] += 1
                matched_callback_ids.add(id(callback))
            rows.append(row)

    for callback in callback_records:
        if id(callback) not in matched_callback_ids:
            counters["unmatched_callbacks"] += 1
            counters["callback_duplicates"] += int(
                callback.get("duplicate", False)
            )
            counters["callback_out_of_order"] += int(
                callback.get("out_of_order", False)
            )

    fields = sorted({key for row in rows for key in row})
    with (output_directory / "communication_latency.csv").open(
            "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    testbed_path = run_directory / "testbed_result.json"
    testbed = json.loads(testbed_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "communication_summary_v1",
        **dict(counters),
        "overwritten_observed": 0,
        "dropped_observed": 0,
        "loss_classification": (
            "DDS does not expose depth-1 overwrite versus transport drop; "
            "unmatched relay records remain dropped_or_overwritten"
        ),
        "per_scene": [
            {
                "scene_index": key[0],
                "scene_token": key[1],
                "model_id": key[2],
                "input_topic": key[3],
                **dict(value),
            }
            for key, value in sorted(per_scene.items())
        ],
        "playback_intervals": testbed.get("playback_intervals", []),
        "relay_status": json.loads(
            (run_directory / "communication_relay_status.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    (output_directory / "communication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
