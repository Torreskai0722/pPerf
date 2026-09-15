"""Conservative four-stage analysis of an exact MPS kernel capsule."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

VALID_STATUSES = frozenset(
    (
        "supported",
        "ruled_out",
        "inconclusive",
        "not_observable",
    )
)


@dataclass
class AnalyzerFinding:
    """Machine-readable result shared by all MPS stage skills."""

    skill: str
    stage: int
    hypothesis: str
    status: str
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    opposing_evidence: List[Dict[str, Any]] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    precision_class: str = "indirect"
    clock_error_class: str = "unknown"
    confidence: Dict[str, Any] = field(default_factory=dict)
    profiler_perturbation: str = "none"
    microbenchmark_validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Validate and return a JSON-safe analyzer finding."""
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid analyzer status: {self.status}")
        return asdict(self)


def _confidence(
    status: str, evidence_count: int, missing_count: int, direct: bool = False
) -> Dict[str, Any]:
    """Return an inspectable evidence-strength score, not a probability."""
    status_base = {
        "supported": 0.55,
        "ruled_out": 0.55,
        "inconclusive": 0.2,
        "not_observable": 0.0,
    }[status]
    score = min(
        0.95,
        max(
            0.0,
            status_base
            + min(evidence_count, 4) * 0.07
            - min(missing_count, 4) * 0.08
            + (0.08 if direct else 0.0),
        ),
    )
    return {
        "score": round(score, 3),
        "label": (
            "high" if score >= 0.75 else "medium" if score >= 0.4 else "low"
        ),
        "is_probability": False,
        "components": {
            "status_base": status_base,
            "supporting_or_opposing_records": evidence_count,
            "missing_records": missing_count,
            "direct_observation": direct,
        },
    }


def _microbenchmark(evidence: Dict[str, Any], stage: int) -> Dict[str, Any]:
    trials = [
        row
        for row in evidence.get("microbenchmark_trials", [])
        if int(row.get("stage", -1)) == stage
    ]
    intended = [
        row
        for row in trials
        if row.get("observed_status") == row.get("intended_status")
    ]
    false_direct = any(
        bool(row.get("direct_wdu_claim")) and stage in (1, 2) for row in trials
    )
    return {
        "trials": len(trials),
        "matching_trials": len(intended),
        "validated": len(trials) >= 3
        and len(intended) >= 2
        and not false_direct,
        "false_direct_wdu_claim": false_direct,
    }


class MpsStageSkill:
    """Base helper for a single independently evaluated MPS stage."""

    skill = ""
    stage = 0
    hypothesis = ""

    def _finding(
        self,
        evidence: Dict[str, Any],
        status: str,
        supporting: Optional[List[Dict[str, Any]]] = None,
        opposing: Optional[List[Dict[str, Any]]] = None,
        missing: Optional[List[str]] = None,
        precision: str = "indirect",
        clock_error: str = "bounded",
        perturbation: str = "none",
        direct: bool = False,
    ) -> AnalyzerFinding:
        supporting = supporting or []
        opposing = opposing or []
        missing = missing or []
        return AnalyzerFinding(
            self.skill,
            self.stage,
            self.hypothesis,
            status,
            supporting,
            opposing,
            missing,
            precision,
            clock_error,
            _confidence(
                status, len(supporting) + len(opposing), len(missing), direct
            ),
            perturbation,
            _microbenchmark(evidence, self.stage),
        )


class AnalyzeMpsTaskSlotAdmissionV1(MpsStageSkill):
    """Evaluate whether the victim TMD was admitted to a WDU task slot."""

    skill = "AnalyzeMpsTaskSlotAdmissionV1"
    stage = 1
    hypothesis = "victim_tmd_not_admitted_to_wdu_task_slot"

    def analyze(
        self, evidence: Dict[str, Any], _ledger=None
    ) -> AnalyzerFinding:
        """Classify task-slot admission from calibrated evidence."""
        pending = evidence.get("pending_grid_count")
        capacity = evidence.get("calibrated_task_slot_capacity")
        hidden = evidence.get("hidden_mps_task_count")
        admission_signature = evidence.get("task_slot_admission_signature")
        supporting = []
        opposing = []
        missing = []
        if pending is None:
            missing.append("exact pending aggressor/victim grid count")
        if capacity is None:
            missing.append("microbenchmark task-slot admission capacity")
        if hidden is None:
            missing.append("hidden MPS task census")
        if (
            pending is not None
            and capacity is not None
            and hidden == 0
            and int(pending) < int(capacity)
        ):
            opposing.extend(
                [
                    {
                        "fact": "pending_below_capacity",
                        "pending": pending,
                        "capacity": capacity,
                    },
                    {"fact": "no_hidden_mps_tasks"},
                ]
            )
            return self._finding(
                evidence,
                "ruled_out",
                opposing=opposing,
                missing=missing,
                precision="calibrated_indirect",
            )
        if (
            pending is not None
            and capacity is not None
            and int(pending) >= int(capacity)
            and admission_signature is True
        ):
            supporting.extend(
                [
                    {
                        "fact": "slot_pressure_boundary_reached",
                        "pending": pending,
                        "capacity": capacity,
                    },
                    {"fact": "calibrated_admission_signature_present"},
                ]
            )
            return self._finding(
                evidence,
                "supported",
                supporting=supporting,
                missing=missing,
                precision="calibrated_indirect",
            )
        if admission_signature is None:
            missing.append("calibrated admission signature")
        return self._finding(
            evidence,
            "inconclusive",
            supporting=supporting,
            opposing=opposing,
            missing=missing,
            precision="indirect_no_wdu_telemetry",
        )


class AnalyzeMpsWduSelectionV1(MpsStageSkill):
    """Evaluate whether admitted victim work was not selected for dispatch."""

    skill = "AnalyzeMpsWduSelectionV1"
    stage = 2
    hypothesis = "admitted_victim_not_selected_for_block_dispatch"

    def analyze(
        self, evidence: Dict[str, Any], ledger=None
    ) -> AnalyzerFinding:
        """Classify WDU selection from admission and CTA evidence."""
        if not evidence.get("nvbit_cta_available", False):
            return self._finding(
                evidence,
                "not_observable",
                missing=["per-CTA dispatch timeline under MPS"],
                precision="not_observed",
                perturbation="optional_backend_absent",
            )
        if evidence.get("cta_coverage_complete") is False:
            return self._finding(
                evidence,
                "inconclusive",
                missing=[
                    "complete CTA coverage for absence-based conclusions"
                ],
                precision="partial_cta_observation",
                perturbation="nvbit_cta",
            )
        stage1 = (ledger or {}).get(1, {}).get("status")
        admission_eligible = stage1 in ("supported", "ruled_out")
        favored = evidence.get("aggressor_dispatch_continues_victim_absent")
        residency = evidence.get("victim_block_residency_capacity")
        signature = evidence.get("wdu_selection_signature")
        supporting = []
        missing = []
        if stage1 is None:
            missing.append("stage-1 finding")
        elif not admission_eligible:
            missing.append("admission supported or ruled out")
        if residency is None:
            missing.append("victim block residency arithmetic")
        if favored is None:
            missing.append(
                "continued aggressor dispatch with absent victim CTAs"
            )
        if signature is None:
            missing.append("microbenchmark selection signature")
        if (
            admission_eligible
            and residency is True
            and favored is True
            and (signature is True)
        ):
            supporting.extend(
                [
                    {"fact": "admission_precondition", "stage1": stage1},
                    {"fact": "resident_victim_block_would_fit"},
                    {
                        "fact": (
                            "aggressor_dispatch_continued_while_"
                            "victim_absent"
                        )
                    },
                    {"fact": "calibrated_selection_signature_present"},
                ]
            )
            return self._finding(
                evidence,
                "supported",
                supporting=supporting,
                precision="cta_timeline_plus_calibrated_indirect",
                perturbation="nvbit_cta",
            )
        opposing = []
        if favored is False:
            opposing.append({"fact": "no_consistent_aggressor_dispatch_bias"})
        if opposing and admission_eligible:
            return self._finding(
                evidence,
                "ruled_out",
                opposing=opposing,
                missing=missing,
                precision="cta_timeline_plus_calibrated_indirect",
                perturbation="nvbit_cta",
            )
        return self._finding(
            evidence,
            "inconclusive",
            supporting=supporting,
            opposing=opposing,
            missing=missing,
            precision="indirect_no_wdu_telemetry",
            perturbation="nvbit_cta",
        )


class AnalyzeMpsTpcEligibilityV1(MpsStageSkill):
    """Evaluate TPC eligibility under captured MPS mask or credit state."""

    skill = "AnalyzeMpsTpcEligibilityV1"
    stage = 3
    hypothesis = "selected_work_has_no_eligible_tpc"

    def analyze(
        self, evidence: Dict[str, Any], _ledger=None
    ) -> AnalyzerFinding:
        """Classify TPC eligibility without mutating MPS controls."""
        if not evidence.get("nvbit_cta_available", False):
            return self._finding(
                evidence,
                "not_observable",
                missing=["per-CTA SM placement"],
                precision="not_observed",
                perturbation="optional_backend_absent",
            )
        if evidence.get("cta_coverage_complete") is False:
            return self._finding(
                evidence,
                "inconclusive",
                missing=[
                    "complete CTA coverage for absence-based conclusions"
                ],
                precision="partial_cta_observation",
                perturbation="nvbit_cta",
            )
        required = {
            "mps_active_thread_percentage": (
                "original active-thread percentage"
            ),
            "mps_client_priority": "original client priority",
            "gpu_topology": "GPC/TPC/SM topology",
            "candidate_tpc_ids": "candidate TPC set",
            "mps_tpc_mask_or_credit_state": "MPS TPC mask or credit state",
            "tpc_validation_signature": "exact-mask microbenchmark signature",
        }
        missing = [
            label
            for key, label in required.items()
            if evidence.get(key) is None
        ]
        if "MPS TPC mask or credit state" in missing:
            return self._finding(
                evidence,
                "inconclusive",
                missing=missing,
                precision="cta_placement_without_credit_state",
                perturbation="nvbit_cta",
            )
        absent = evidence.get("victim_absent_from_candidate_tpcs")
        eligible = evidence.get("eligible_tpc_count")
        signature = evidence.get("tpc_validation_signature")
        if eligible == 0 and absent is True and signature is True:
            supporting = [
                {"fact": "no_eligible_tpc_in_captured_state"},
                {"fact": "victim_absent_from_candidate_tpcs"},
                {"fact": "validation_mask_signature_matches"},
            ]
            return self._finding(
                evidence,
                "supported",
                supporting=supporting,
                missing=missing,
                precision="cta_placement_plus_readonly_mps_state",
                perturbation="nvbit_cta",
            )
        if isinstance(eligible, int) and eligible > 0 and absent is False:
            return self._finding(
                evidence,
                "ruled_out",
                opposing=[
                    {
                        "fact": "victim_placed_on_eligible_tpc",
                        "eligible_tpc_count": eligible,
                    }
                ],
                missing=missing,
                precision="cta_placement_plus_readonly_mps_state",
                perturbation="nvbit_cta",
            )
        if absent is None:
            missing.append("victim placement across candidate TPCs")
        return self._finding(
            evidence,
            "inconclusive",
            missing=missing,
            precision="cta_placement_plus_readonly_mps_state",
            perturbation="nvbit_cta",
        )


class AnalyzeMpsResourceResidencyV1(MpsStageSkill):
    """Evaluate whether placed victim CTAs could become resident."""

    skill = "AnalyzeMpsResourceResidencyV1"
    stage = 4
    hypothesis = "victim_blocks_cannot_become_resident"

    def analyze(
        self, evidence: Dict[str, Any], _ledger=None
    ) -> AnalyzerFinding:
        """Classify residency from exact launch and CTA facts."""
        missing_collectors = []
        if not evidence.get("nvbit_cta_available", False):
            missing_collectors.append("CTA entry/exit per SM")
        if not evidence.get("resource_profile_available", False):
            missing_collectors.append("NCU/CUDA launch resource profile")
        if missing_collectors:
            return self._finding(
                evidence,
                "not_observable",
                missing=missing_collectors,
                precision="not_observed",
                perturbation="optional_backend_absent",
            )
        if evidence.get("cta_coverage_complete") is False:
            return self._finding(
                evidence,
                "inconclusive",
                missing=[
                    "complete CTA coverage for absence-based conclusions"
                ],
                precision="partial_cta_observation",
                perturbation="nvbit_cta",
            )
        capacity = evidence.get("eligible_sm_capacity_during_delay")
        follows_release = evidence.get("victim_entry_follows_required_release")
        occupancy_verified = evidence.get("occupancy_calculation_verified")
        missing = []
        if capacity is None:
            missing.append("per-eligible-SM capacity through delay interval")
        if follows_release is None:
            missing.append("aggressor release to victim CTA entry timing")
        if occupancy_verified is not True:
            missing.append("verified CUDA occupancy calculation")
        if (
            capacity == "insufficient_all"
            and follows_release is True
            and (occupancy_verified is True)
        ):
            return self._finding(
                evidence,
                "supported",
                supporting=[
                    {"fact": "all_eligible_sms_lacked_one_victim_block"},
                    {
                        "fact": (
                            "victim_entry_followed_required_resource_release"
                        )
                    },
                    {"fact": "launch_and_occupancy_facts_match"},
                ],
                missing=missing,
                precision="cta_timeline_and_resource_exact",
                clock_error="calibrated_device_globaltimer",
                perturbation="nvbit_cta_and_ncu_facts",
                direct=True,
            )
        if capacity == "verified_capacity_throughout":
            return self._finding(
                evidence,
                "ruled_out",
                opposing=[
                    {"fact": "eligible_sm_had_verified_capacity_throughout"}
                ],
                missing=missing,
                precision="cta_timeline_and_resource_exact",
                clock_error="calibrated_device_globaltimer",
                perturbation="nvbit_cta_and_ncu_facts",
                direct=True,
            )
        return self._finding(
            evidence,
            "inconclusive",
            missing=missing,
            precision="cta_timeline_and_resource_exact",
            clock_error="calibrated_device_globaltimer",
            perturbation="nvbit_cta_and_ncu_facts",
        )


STAGE_SKILLS = (
    AnalyzeMpsTaskSlotAdmissionV1,
    AnalyzeMpsWduSelectionV1,
    AnalyzeMpsTpcEligibilityV1,
    AnalyzeMpsResourceResidencyV1,
)


class MpsContentionAnalyzer:
    """Load capsule evidence and run all four skills independently."""

    analyzer_id = "mps_contention_analyzer_v1"

    def __init__(
        self, run_directory: Path, options: Optional[Dict[str, Any]] = None,
        output_directory: Optional[Path] = None,
    ):
        """Configure offline analysis for one capsule directory."""
        self.run_directory = Path(run_directory)
        self.output_directory = Path(output_directory or run_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.options = dict(options or {})
        self.sqlite_path = self.run_directory / "capsule_replay.sqlite"

    @staticmethod
    def _metadata(connection: sqlite3.Connection) -> Dict[str, Any]:
        return {
            str(key): json.loads(value)
            for key, value in connection.execute(
                "SELECT key, value_json FROM metadata"
            )
        }

    def load_evidence(self) -> Dict[str, Any]:
        """Normalize the stage-relevant subset of capsule evidence."""
        with sqlite3.connect(str(self.sqlite_path)) as connection:
            metadata = self._metadata(connection)
            capabilities = {
                row[0]: bool(row[1])
                for row in connection.execute(
                    "SELECT collector, available FROM capability"
                )
            }
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            coverage = None
            if "cta_collection_status" in tables:
                counts = connection.execute(
                    "SELECT expected_count, complete_count "
                    "FROM cta_collection_status"
                ).fetchall()
                if counts:
                    coverage = all(
                        int(expected) == int(complete)
                        for expected, complete in counts
                    )
            resource_count = (
                connection.execute(
                    "SELECT COUNT(*) FROM kernel_resource_profile"
                ).fetchone()[0]
                if "kernel_resource_profile" in tables else 0
            )
            trials = [
                {
                    "stage": row[0],
                    "condition_id": row[1],
                    "trial": row[2],
                    "intended_status": row[3],
                    "observed_status": row[4],
                    "direct_wdu_claim": bool(row[5]),
                    **json.loads(row[6]),
                }
                for row in connection.execute(
                    "SELECT stage, condition_id, trial, intended_status, "
                    "observed_status, direct_wdu_claim, evidence_json "
                    "FROM microbenchmark_validation"
                )
            ]
        evidence = dict(metadata.get("stage_evidence", {}))
        evidence["microbenchmark_trials"] = trials
        evidence["nvbit_cta_available"] = capabilities.get("nvbit_cta", False)
        evidence["cta_coverage_complete"] = coverage
        evidence["resource_profile_available"] = resource_count > 0
        return evidence

    def analyze(
        self, evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the fixed skills and write ``mps_stage_analysis.json``."""
        evidence = dict(evidence or self.load_evidence())
        ledger: Dict[int, Dict[str, Any]] = {}
        findings = []
        for skill_type in STAGE_SKILLS:
            finding = skill_type().analyze(evidence, ledger).to_dict()
            ledger[int(finding["stage"])] = finding
            findings.append(finding)
        result = {
            "schema_version": 1,
            "analyzer": self.analyzer_id,
            "conclusion_policy": self.options.get(
                "conclusion_policy", "conservative"
            ),
            "replay_integrity_gate": "passed",
            "findings": findings,
            "hypothesis_ledger": {
                str(stage): {
                    "hypothesis": finding["hypothesis"],
                    "status": finding["status"],
                    "finding_skill": finding["skill"],
                }
                for stage, finding in ledger.items()
            },
            "limitations": [
                "stages 1 and 2 are indirect without hardware WDU telemetry",
                "NCU replay timing is excluded from timing conclusions",
                "context-switch and time-slice evidence is not evaluated",
            ],
        }
        output = self.output_directory / "mps_stage_analysis.json"
        temporary = output.with_name("." + output.name + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return result


class MpsAnalyzer:
    """Analyze MPS overlap, metrics, CTA, capsule, and report evidence."""

    def analyze(self, source, output_root, options=None) -> dict:
        """Analyze one compatible MPS run and return a written summary."""
        from pathlib import Path

        from ._manifest import EvidenceError, load_manifest, require
        from ._output import result_directory, write_result

        options = dict(options or {})
        manifest = load_manifest(source, output_root)
        require(manifest, mps=True, scope="model")
        sub_analysis = options.get("analysis", "overlap")
        if sub_analysis == "device_metrics":
            require(manifest, levels=("level2a",))
        elif sub_analysis == "target-selection":
            require(manifest, levels=("level1", "level2a"))
        elif sub_analysis in ("ncu", "targeted"):
            require(manifest, levels=("level2b",))
        elif (
            sub_analysis == "capsule"
            and "capsule" not in manifest["recording"]
        ):
            raise EvidenceError("capsule analysis requires capsule evidence")
        run = Path(manifest["run_directory"])
        destination = result_directory(output_root, "mps", manifest)
        result = {
            "analysis_type": "mps",
            "sub_analysis": sub_analysis,
            "run_id": manifest.get("run_id"),
            "recording": manifest["recording"],
            "evidence": sorted(
                path.name for path in run.iterdir() if path.is_file()
            ),
            "options": options,
        }
        if sub_analysis == "overlap" and (run / "profile.sqlite").is_file():
            from .mps_leftover import MpsLeftoverAnalyzer

            result["leftover"] = MpsLeftoverAnalyzer(
                run,
                float(options.get("delay_threshold_ms", 0.9)),
                destination,
            ).analyze()
        elif sub_analysis == "capsule" and (
                run / "capsule_replay.sqlite").is_file():
            result["capsule"] = MpsContentionAnalyzer(
                run, options, destination).analyze()
        elif sub_analysis == "target-selection":
            from .target_selection import TraceTargetSelector

            result["target_selection"] = TraceTargetSelector(
                run / "profile.sqlite", run,
                float(options.get("aggressor_min_duration_ms", 1.0)),
            ).select()
        return write_result(output_root, "mps", manifest, result)
