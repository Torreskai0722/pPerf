"""Shared identities and buffered records for communication profiling."""

import json
from pathlib import Path


RELAY_ID = "__communication_relay__"


def raw_topic(source_topic):
    """Return the internal topic used between rosbag and the relay."""
    return "/closeloop/raw" + source_topic


def relay_routes(config):
    """Return source, raw, and model-input topic routes in replay order."""
    remappings = config["replay"]["remappings"]
    return [
        {
            "source_topic": source,
            "raw_topic": raw_topic(source),
            "input_topic": remappings.get(source, source),
        }
        for source in config["replay"]["topics"]
    ]


class BatchedJsonlWriter:
    """Write JSONL records in bounded batches."""

    def __init__(self, path, batch_size=256):
        self.output = Path(path).open("w", encoding="utf-8")
        self.batch_size = batch_size
        self.records = []
        self.batches_written = 0

    def write(self, record):
        """Buffer one record and flush a full batch."""
        self.records.append(record)
        if len(self.records) >= self.batch_size:
            self.flush()

    def flush(self):
        """Write all buffered records."""
        if not self.records:
            return
        self.output.writelines(
            json.dumps(record, sort_keys=True) + "\n"
            for record in self.records
        )
        self.output.flush()
        self.records.clear()
        self.batches_written += 1

    def close(self):
        """Flush and close the underlying file."""
        self.flush()
        self.output.close()


class StreamOrderTracker:
    """Assign stream and duplicate ordinals while detecting reordering."""

    def __init__(self):
        self.counts = {}
        self.occurrences = {}
        self.last_timestamps = {}

    def observe(self, segment_id, topic, timestamp_ns):
        """Return order evidence for one observed message."""
        stream = (segment_id, topic)
        order = self.counts.get(stream, 0)
        self.counts[stream] = order + 1
        identity = (segment_id, topic, timestamp_ns)
        occurrence = self.occurrences.get(identity, 0)
        self.occurrences[identity] = occurrence + 1
        previous = self.last_timestamps.get(stream)
        out_of_order = (
            previous is not None
            and timestamp_ns is not None
            and previous is not None
            and timestamp_ns < previous
        )
        if timestamp_ns is not None:
            self.last_timestamps[stream] = timestamp_ns
        return {
            "message_order": order,
            "timestamp_occurrence": occurrence,
            "duplicate": occurrence > 0,
            "out_of_order": out_of_order,
        }
