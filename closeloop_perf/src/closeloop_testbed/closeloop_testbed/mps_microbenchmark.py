"""Three-trial synthetic capsule matrix for MPS stage validation."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class SyntheticCondition:
    """One controlled two-process MPS microbenchmark condition."""

    condition_id: str
    stage: int
    intended_status: str
    mode: str
    parameters: Dict[str, Any]
    trials: int = 3


def validation_matrix(task_slot_knee: int) -> List[SyntheticCondition]:
    """Return three-trial conditions around four stage boundaries."""
    if task_slot_knee < 2:
        raise ValueError("task-slot knee must leave a below-knee condition")
    return [
        SyntheticCondition(
            "stage1-at-admission-knee",
            1,
            "supported",
            "persistent_grid",
            {"pending_grids": task_slot_knee},
        ),
        SyntheticCondition(
            "stage1-below-admission-knee",
            1,
            "ruled_out",
            "persistent_grid",
            {"pending_grids": task_slot_knee - 1},
        ),
        SyntheticCondition(
            "stage2-priority-order",
            2,
            "supported",
            "dispatch_order",
            {"pending_grids": task_slot_knee - 1, "controlled_priority": True},
        ),
        SyntheticCondition(
            "stage3-exact-tpc-mask",
            3,
            "supported",
            "tpc_eligibility",
            {"validation_only_mask": True},
        ),
        SyntheticCondition(
            "stage4-residency-boundary",
            4,
            "supported",
            "residency",
            {
                "sweep_registers": True,
                "sweep_shared_memory": True,
                "sweep_warps": True,
            },
        ),
    ]


def validate_trials(
    condition: SyntheticCondition, observed: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Require at least two matching trials and no direct WDU claim."""
    records = list(observed)
    if len(records) != 3:
        raise ValueError("every synthetic condition requires three trials")
    matches = sum(
        record.get("status") == condition.intended_status for record in records
    )
    false_direct = any(
        bool(record.get("direct_wdu_observation"))
        for record in records
        if condition.stage in (1, 2)
    )
    return {
        "condition": asdict(condition),
        "matching_trials": matches,
        "false_direct_wdu_claim": false_direct,
        "validated": matches >= 2 and not false_direct,
        "trials": records,
    }


def run_binary(
    binary: Path, condition: SyntheticCondition, output_directory: Path
) -> List[Dict[str, Any]]:
    """Run a preconfigured two-process condition exactly three times."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for trial in range(3):
        output = output_directory / f"{condition.condition_id}-{trial}.json"
        command = [
            str(binary),
            "--mode",
            condition.mode,
            "--condition-json",
            json.dumps(condition.parameters, sort_keys=True),
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True)
        records.append(json.loads(output.read_text(encoding="utf-8")))
    return records
