"""Per-process environment construction for testbed model nodes."""

from closeloop_testbed.resource_control import thread_environment


def model_environment(config, model, run_directory):
    """Return early native-pool and optional MPS variables for one model."""
    environment = thread_environment(model)
    cta = model.get("nvbit_cta_profile")
    if cta:
        targets = cta["targets"]
        prefix = str(run_directory) + "/model_" + model["id"]
        mode = cta.get("mode", "controlled")
        environment.update({
            "LD_PRELOAD": cta["tracker_library"],
            "PPERF_NVBIT_CTA_MODE": mode,
            "PPERF_NVBIT_CTA_MODEL_ID": model["id"],
            "PPERF_NVBIT_CTA_OUTPUT_PREFIX": prefix,
            "PPERF_NVBIT_CTA_TARGET_LABELS": ",".join(
                target["label"] for target in targets
            ),
        })
        if "record_capacity" in cta:
            environment["PPERF_NVBIT_CTA_RECORD_CAPACITY"] = str(
                cta["record_capacity"]
            )
        if mode in ("passive", "mixed"):
            environment.update({
                "PPERF_NVBIT_CTA_SEQUENCE_STARTS": ",".join(
                    str(target["sequence_start"]) for target in targets
                ),
                "PPERF_NVBIT_CTA_SEQUENCE_ENDS": ",".join(
                    str(target["sequence_end"]) for target in targets
                ),
            })
        if mode in ("controlled", "mixed"):
            environment.update({
                "PPERF_NVBIT_CTA_TARGET_NAME": cta["kernel_name"],
                "PPERF_NVBIT_CTA_TARGET_GRID": ",".join(
                    str(value) for value in cta["grid"]
                ),
                "PPERF_NVBIT_CTA_TARGET_BLOCK": ",".join(
                    str(value) for value in cta["block"]
                ),
                "PPERF_NVBIT_CTA_TARGET_OCCURRENCES": ",".join(
                    str(target["kernel_occurrence"])
                    for target in targets
                ),
                "PPERF_NVBIT_CTA_ALIGN_LABELS": ",".join(
                    target["label"] for target in targets
                    if target.get("align_launch", False)
                ),
                "PPERF_NVBIT_CTA_BARRIER_PARTICIPANTS": ",".join(
                    cta["barrier_participants"]
                ),
            })
    if not config["gpu"]["mps_enabled"]:
        return environment
    root = str(run_directory)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]["index"]),
        "CUDA_MPS_PIPE_DIRECTORY": root + "/p",
        "CUDA_MPS_LOG_DIRECTORY": root + "/mps/log",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(model["mps_percentage"]),
    })
    return environment


def fma_environment(config, run_directory):
    """Return the run-owned MPS environment for the FMA sidecar."""
    fma = config["synthetic_fma"]
    root = str(run_directory)
    return {
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]["index"]),
        "CUDA_MPS_PIPE_DIRECTORY": root + "/p",
        "CUDA_MPS_LOG_DIRECTORY": root + "/mps/log",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(fma["mps_percentage"]),
    }


def fma_worker_environment(config, environment):
    """Add the CTA tracker only to the native FMA worker environment."""
    fma = config["synthetic_fma"]
    result = dict(environment)
    result.update({
        "LD_PRELOAD": fma["cta"]["tracker_library"],
        "PPERF_NVBIT_CTA_RECORD_CAPACITY": str(
            fma["cta"]["record_capacity"]
        ),
    })
    return result
