"""Select the in-process recorders requested by a run."""

from pathlib import Path
from typing import Any, Dict


def recording_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the validated recording selection."""
    return data["recording"]


def scheduler_script_path() -> Path:
    """Return the packaged scheduler evidence recorder."""
    return Path(__file__).with_name("bpf") / "preprocess_scheduler.bt"


def create_model_profiler(data: Dict[str, Any], inferencer: Any,
                          model_id: str) -> Any:
    """Construct the recorder for one model process."""
    recording = recording_config(data)
    model = next(item for item in data["models"] if item["id"] == model_id)
    arguments = (
        inferencer.model,
        data["run"]["id"],
        data["replay"]["scene_token"],
        model_id,
        model["architecture_profile"],
        model["module_annotation_depth"],
    )
    capsule = recording.get("capsule")
    if capsule:
        from .capsule_profiler import GpuForensicsProfiler
        return GpuForensicsProfiler(*arguments, options=capsule)
    if "preprocessing" in recording["scopes"]:
        from .preprocess_profiler import PreprocessProfiler
        return PreprocessProfiler(
            *arguments,
            inferencer=inferencer,
            options=recording.get("preprocessing", {}),
        )
    from .profiler import ModelProfiler
    return ModelProfiler(*arguments)
