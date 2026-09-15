"""Unix-socket client for the live kernel-capsule host coordinator."""

import json
from pathlib import Path
import socket
from typing import Any, Dict

from closeloop_profiler.kernel_capsule import KernelCapsule, capsule_v3_from_dict
from .kernel_replay import ReplayBackend
from .rpc_transport import receive_json_line


class CoordinatorProtocolError(RuntimeError):
    """Raised when the native coordinator violates the JSON protocol."""


class UnixAgentReplayBackend(ReplayBackend):
    """Coordinate two injected MPS replay agents through one local socket."""

    def __init__(
        self,
        socket_path: Path,
        timeout_seconds: float = 120.0,
        shutdown_on_close: bool = False,
    ):
        """Configure the local coordinator socket and request timeout."""
        self.socket_path = Path(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self.shutdown_on_close = bool(shutdown_on_close)

    def _request(self, operation: str, **payload: Any) -> Dict[str, Any]:
        request = {
            "protocol": "kernel_capsule_agent_v3",
            "operation": operation,
            **payload,
        }
        encoded = (json.dumps(request, sort_keys=True) + "\n").encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            connection.sendall(encoded)
            try:
                response = receive_json_line(connection)
            except (
                ConnectionError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raise CoordinatorProtocolError(
                    "invalid response from kernel capsule coordinator"
                ) from exc
        if response.get("protocol") != "kernel_capsule_agent_v3":
            raise CoordinatorProtocolError("coordinator protocol mismatch")
        if response.get("status") != "ok":
            raise CoordinatorProtocolError(
                str(response.get("error", "coordinator request failed"))
            )
        return dict(response.get("result", {}))

    def capture_one_instrumented_frame(
        self, capture_specification: Dict[str, Any]
    ) -> KernelCapsule:
        """Arm both agents before one post-warmup instrumented frame."""
        result = self._request(
            "capture_one_instrumented_frame",
            capture_specification=capture_specification,
        )
        return capsule_v3_from_dict(result["capsule"])

    def capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Return coordinator-probed collector capabilities."""
        return dict(self._request("capabilities").get("collectors", {}))

    def restore_checkpoints(self, capsule: KernelCapsule) -> int:
        """Restore both live client contexts at the fixed capture point."""
        result = self._request(
            "restore_checkpoints", capsule_id=capsule.capsule_id
        )
        return int(result.get("restored_client_count", 0))

    def replay(
        self, capsule: KernelCapsule, variant: str, instrumentation: str
    ) -> Dict[str, Any]:
        """Request exact launch issue with recursive callbacks disabled."""
        return self._request(
            "replay",
            capsule_id=capsule.capsule_id,
            variant=variant,
            instrumentation=instrumentation,
            submission_policy="common_epoch_dependency_burst",
            client_release_offsets_ns=capsule.client_release_offsets_ns,
            allow_non_capsule_cuda_work=False,
        )

    def aggressor_priority_range(
        self, capsule: KernelCapsule
    ) -> Dict[str, Any]:
        """Query the aggressor context's supported launch-priority range."""
        return self._request(
            "aggressor_priority_range", capsule_id=capsule.capsule_id
        )

    def replay_with_aggressor_priority(
        self, capsule: KernelCapsule, priority: int
    ) -> Dict[str, Any]:
        """Replay the pair with an explicit per-launch priority attribute."""
        return self._request(
            "replay",
            capsule_id=capsule.capsule_id,
            variant="pair",
            instrumentation="launch_priority_counterfactual",
            aggressor_priority=int(priority),
            submission_policy="common_epoch_dependency_burst",
            client_release_offsets_ns=capsule.client_release_offsets_ns,
            allow_non_capsule_cuda_work=False,
        )

    def collect_resource_profile(
        self, capsule: KernelCapsule, launch_id: str
    ) -> Dict[str, Any]:
        """Collect facts for one exact launch, excluding NCU timing."""
        return self._request(
            "collect_resource_profile",
            capsule_id=capsule.capsule_id,
            launch_id=launch_id,
            use_replay_timing=False,
        )

    def close(self) -> None:
        """Explicitly stop disposable clients when this backend owns them."""
        if not self.shutdown_on_close:
            return
        try:
            self._request("shutdown")
        except (OSError, CoordinatorProtocolError):
            pass
