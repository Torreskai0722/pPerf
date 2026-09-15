"""Model selection, status, and recorder evidence helpers."""

import json


def select_model(config, model_id):
    """Select exactly one configured model by stable identity."""
    matches = [item for item in config["models"] if item["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"expected one model with ID {model_id!r}")
    return matches[0]


def write_status(path, status):
    """Atomically replace a model process status record."""
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_profile_evidence(status, profiler):
    """Copy deterministic architecture evidence into a model status."""
    status.update({
        "architecture_profile": profiler.architecture_profile,
        "architecture_profile_sha256": profiler.architecture_profile_sha256,
        "observed_method_bindings": sorted(profiler.observed_method_bindings),
        "observed_module_bindings": sorted(profiler.observed_module_bindings),
        "profiler_id": getattr(profiler, "profiler_id", "model_level1_v1"),
    })
    if hasattr(profiler, "observed_preprocess_steps"):
        status["observed_preprocess_steps"] = sorted(
            profiler.observed_preprocess_steps)
    if hasattr(profiler, "pipeline_manifest"):
        status["pipeline_manifest"] = profiler.pipeline_manifest
