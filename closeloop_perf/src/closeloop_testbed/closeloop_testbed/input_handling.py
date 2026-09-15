"""Input identity and queue handling shared by model processes."""

from closeloop_profiler.source_frames import message_header_timestamp_ns


def configured_input_queue_depth(model_config):
    """Return configured ROS input history depth."""
    return model_config.get("input_queue_depth", 10)


def model_input_record(model_id, input_id, input_topic, message,
                       callback_entry_ns=None, segment=None, order=None):
    """Return the identity record written outside the inference NVTX range."""
    record = {
        "schema": "model_input_identity_v1",
        "model_id": str(model_id),
        "input_id": str(input_id),
        "input_topic": str(input_topic),
        "ros_header_timestamp_ns": message_header_timestamp_ns(message),
    }
    if callback_entry_ns is not None:
        record["model_callback_entry_monotonic_ns"] = callback_entry_ns
    if segment:
        record.update(segment)
    if order:
        record.update(order)
    return record
