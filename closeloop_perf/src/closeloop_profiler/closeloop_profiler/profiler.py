"""Model-attached NVTX instrumentation for Level-1 profiling."""

from contextlib import contextmanager
from functools import wraps
import json
import time
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .architecture_profiles import (
    ArchitectureProfileError, MethodBinding, ModuleBinding,
    get_architecture_profile
)


TAG_SCHEMA_VERSION = 1
_ResolvedMethod = Tuple[MethodBinding, Any, Any]
_ResolvedModule = Tuple[ModuleBinding, Any]


def encode_tag(run_id: str, scene_token: str, model_id: str,
               input_id: str, event_type: str,
               module_path: Optional[str] = None, *,
               owner: Optional[str] = None,
               method_path: Optional[str] = None,
               invocation: Optional[int] = None,
               architecture_profile: Optional[str] = None,
               timed_event: Optional[str] = None,
               wall_time_ns: Optional[int] = None,
               thread_time_ns: Optional[int] = None,
               non_cpu_time_ns: Optional[int] = None) -> str:
    """Encode a deterministic, machine-readable NVTX tag."""
    fields = {
        "event": event_type,
        "input": str(input_id),
        "model": model_id,
        "run": run_id,
        "scene": scene_token,
        "schema_version": TAG_SCHEMA_VERSION,
    }
    if module_path is not None:
        fields["module"] = module_path
    if owner is not None:
        fields["owner"] = owner
    if method_path is not None:
        fields["method"] = method_path
    if invocation is not None:
        fields["invocation"] = invocation
    if architecture_profile is not None:
        fields["architecture_profile"] = architecture_profile
    if timed_event is not None:
        fields["timed_event"] = timed_event
    if wall_time_ns is not None:
        fields["wall_time_ns"] = wall_time_ns
    if thread_time_ns is not None:
        fields["thread_time_ns"] = thread_time_ns
    if non_cpu_time_ns is not None:
        fields["non_cpu_time_ns"] = non_cpu_time_ns
    return "closeloop:" + json.dumps(
        fields, sort_keys=True, separators=(",", ":")
    )


def selected_module_paths(model: Any, exact_depth: int,
                          module_roots: List[str]) -> List[str]:
    """Return profile-rooted module paths at exactly the requested depth."""
    if exact_depth < 0:
        raise ValueError("module annotation depth must be non-negative")
    roots = list(module_roots)
    if exact_depth == 0:
        return roots
    return [
        name for name, _ in model.named_modules()
        if name and name.count(".") == exact_depth
        and any(name.startswith(root + ".") for root in roots)
    ]


class ModelProfiler:
    """Annotate preprocessing, inference, modules, and CUDA completion."""

    profiler_id = "model_level1_v1"

    def __init__(self, model: Any, run_id: str, scene_token: str,
                 model_id: str, architecture_profile: str,
                 module_depth: int = 0,
                 torch_module: Any = None):
        """Resolve configured bindings and install model method wrappers."""
        if torch_module is None:
            # pylint: disable-next=import-outside-toplevel
            import torch as torch_module
        self.model = model
        self.run_id = run_id
        self.scene_token = scene_token
        self.model_id = model_id
        self.module_depth = module_depth
        self.torch = torch_module
        self.profile = get_architecture_profile(architecture_profile)
        self._handles: List[Any] = []
        self._cpu_timing_starts: Dict[str, List[Tuple[int, int]]] = {}
        self._input_id = ""
        self._emit_method_ranges = False
        self._invocations: Dict[Tuple[str, str, str], int] = {}
        self._observed_method_bindings: Set[str] = set()
        self._observed_module_bindings: Set[str] = set()
        self._wrapped_attributes: List[Tuple[Any, str, bool, Any]] = []
        self._closed = False
        if module_depth < 0:
            raise ValueError("module annotation depth must be non-negative")
        resolved_methods, resolved_modules = self._validate_model()
        self._resolved_methods = resolved_methods
        self._resolved_modules = resolved_modules
        roots = [binding.module_path for binding, _ in resolved_modules]
        self._display_module_paths = set(
            selected_module_paths(self.model, self.module_depth, roots)
        )
        try:
            self._install_method_wrappers(resolved_methods)
            self._install_module_wrappers(resolved_modules)
        except Exception:
            self.close()
            raise

    @property
    def architecture_profile(self) -> str:
        """Return the selected versioned architecture profile name."""
        return self.profile.name

    @property
    def architecture_profile_sha256(self) -> str:
        """Return the canonical architecture profile definition hash."""
        return self.profile.sha256

    @property
    def observed_method_bindings(self) -> List[str]:
        """Return sorted semantic methods observed since construction."""
        return sorted(self._observed_method_bindings)

    @property
    def observed_module_bindings(self) -> List[str]:
        """Return sorted semantic modules observed since construction."""
        return sorted(self._observed_module_bindings)

    @property
    def invocations(self) -> Dict[Tuple[str, str, str], int]:
        """Return invocation counts keyed by input, target, and method."""
        return dict(self._invocations)

    def _resolve_path(self, path: str) -> Any:
        target = self.model
        if not path:
            return target
        for component in path.split("."):
            try:
                target = getattr(target, component)
            except AttributeError as exc:
                raise ArchitectureProfileError(
                    f"profile {self.profile.name!r} requires model path "
                    f"{path!r}"
                ) from exc
            if target is None:
                raise ArchitectureProfileError(
                    f"profile {self.profile.name!r} requires non-null model "
                    f"path {path!r}"
                )
        return target

    def _resolve_module(self, binding: ModuleBinding) -> Optional[Any]:
        try:
            return self._resolve_path(binding.module_path)
        except ArchitectureProfileError:
            if binding.required:
                raise
            return None

    def _validate_model(
            self) -> Tuple[List[_ResolvedMethod], List[_ResolvedModule]]:
        mro_names = {base.__name__ for base in type(self.model).__mro__}
        if not mro_names.intersection(self.profile.accepted_mro):
            accepted = ", ".join(self.profile.accepted_mro)
            actual = ", ".join(base.__name__
                               for base in type(self.model).__mro__)
            raise ArchitectureProfileError(
                f"profile {self.profile.name!r} requires MRO containing "
                f"{accepted}; model MRO is {actual}"
            )
        resolved_methods = []
        for binding in self.profile.method_bindings:
            target = self._resolve_path(binding.target_path)
            try:
                method = getattr(target, binding.method)
            except AttributeError as exc:
                if not binding.required:
                    continue
                raise ArchitectureProfileError(
                    f"profile {self.profile.name!r} requires callable method "
                    f"{binding.method_path!r}"
                ) from exc
            if not callable(method):
                raise ArchitectureProfileError(
                    f"profile {self.profile.name!r} method "
                    f"{binding.method_path!r} is not callable"
                )
            resolved_methods.append((binding, target, method))
        resolved_modules = []
        for binding in self.profile.module_bindings:
            target = self._resolve_module(binding)
            if target is None:
                continue
            entrypoint = getattr(target, binding.entrypoint, None)
            if not callable(entrypoint):
                raise ArchitectureProfileError(
                    f"profile {self.profile.name!r} requires callable "
                    f"module entrypoint "
                    f"{binding.module_path}.{binding.entrypoint!s}"
                )
            resolved_modules.append((binding, target))
        return resolved_methods, resolved_modules

    def _method_tag(self, binding: MethodBinding, invocation: int) -> str:
        return encode_tag(
            self.run_id, self.scene_token, self.model_id, self._input_id,
            "method", owner=binding.owner,
            method_path=binding.method_path, invocation=invocation,
            architecture_profile=self.profile.name
        )

    def _install_method_wrappers(
            self, resolved: List[_ResolvedMethod]) -> None:
        for binding, target, original in resolved:
            instance_attributes = getattr(target, "__dict__", {})
            had_instance_attribute = binding.method in instance_attributes
            prior_value = (instance_attributes[binding.method]
                           if had_instance_attribute else None)

            @wraps(original)
            def wrapped(_instance: Any, *args: Any,
                        _binding: MethodBinding = binding,
                        _original: Any = original, **kwargs: Any) -> Any:
                key = (self._input_id, _binding.target_path,
                       _binding.method)
                invocation = self._invocations.get(key, 0)
                self._invocations[key] = invocation + 1
                self._observed_method_bindings.add(_binding.method_path)
                visible = self._emit_method_ranges
                if visible:
                    self._push(self._method_tag(_binding, invocation))
                try:
                    return _original(*args, **kwargs)
                finally:
                    if visible:
                        self._pop()

            setattr(target, binding.method, MethodType(wrapped, target))
            self._wrapped_attributes.append(
                (target, binding.method, had_instance_attribute, prior_value)
            )

    def _install_module_wrappers(
            self, resolved: List[_ResolvedModule]) -> None:
        for binding, target in resolved:
            if binding.entrypoint == "forward":
                continue
            original = getattr(target, binding.entrypoint)
            instance_attributes = getattr(target, "__dict__", {})
            had_instance_attribute = binding.entrypoint in instance_attributes
            prior_value = (instance_attributes[binding.entrypoint]
                           if had_instance_attribute else None)

            @wraps(original)
            def wrapped(_instance: Any, *args: Any,
                        _binding: ModuleBinding = binding,
                        _original: Any = original, **kwargs: Any) -> Any:
                self._observed_module_bindings.add(_binding.module_path)
                visible = _binding.module_path in self._display_module_paths
                if visible:
                    self._push(self._tag("module", _binding.module_path))
                try:
                    output = _original(*args, **kwargs)
                    self._record_structure(_binding.module_path, output)
                    return output
                finally:
                    if visible:
                        self._pop()

            setattr(target, binding.entrypoint, MethodType(wrapped, target))
            self._wrapped_attributes.append(
                (target, binding.entrypoint, had_instance_attribute,
                 prior_value)
            )

    def _tag(self, event: str, module: Optional[str] = None) -> str:
        return encode_tag(self.run_id, self.scene_token, self.model_id,
                          self._input_id, event, module)

    def _record_structure(self, module_path: str, output: Any) -> None:
        """Mark dynamic proposal, voxel, or sampled-point counts."""
        kind = None
        count = None
        if module_path == "pts_voxel_encoder":
            value = output[0] if isinstance(output, (tuple, list)) else output
            shape = getattr(value, "shape", None)
            if shape:
                kind, count = "voxels", int(shape[0])
        elif module_path == "rpn_head":
            values = output if isinstance(output, (tuple, list)) else [output]
            boxes = [getattr(value, "bboxes", None) for value in values]
            if boxes and all(value is not None for value in boxes):
                kind, count = "proposals", sum(len(value) for value in boxes)
        elif module_path == "backbone" and isinstance(output, dict):
            levels = output.get("sa_xyz")
            shape = getattr(levels[-1], "shape", None) if levels else None
            if shape and len(shape) >= 2:
                kind, count = "sampled_points", int(shape[-2])
        if count is None:
            return
        tag = json.loads(
            self._tag("structure", module_path)[len("closeloop:"):]
        )
        tag.update({"structure_kind": kind, "structure_count": count})
        self._mark("closeloop:" + json.dumps(
            tag, sort_keys=True, separators=(",", ":")
        ))

    def _push(self, tag: str) -> None:
        self.torch.cuda.nvtx.range_push(tag)

    def _pop(self) -> None:
        self.torch.cuda.nvtx.range_pop()

    def _mark(self, tag: str) -> None:
        marker = getattr(self.torch.cuda.nvtx, "mark", None)
        if marker is not None:
            marker(tag)

    def _record_cpu_timing(
            self, timed_event: str, start: Tuple[int, int],
            module: Optional[str] = None) -> None:
        end_thread_time_ns = time.thread_time_ns()
        end_wall_time_ns = time.perf_counter_ns()
        wall_time_ns = end_wall_time_ns - start[0]
        thread_time_ns = end_thread_time_ns - start[1]
        tag = encode_tag(
            self.run_id, self.scene_token, self.model_id, self._input_id,
            "cpu_timing", module, timed_event=timed_event,
            wall_time_ns=wall_time_ns, thread_time_ns=thread_time_ns,
            non_cpu_time_ns=max(0, wall_time_ns - thread_time_ns)
        )
        self._mark(tag)

    @contextmanager
    def range(self, event: str, module: Optional[str] = None,
              record_cpu_timing: bool = False) -> Iterator[None]:
        """Create a balanced NVTX range, including on exceptions."""
        start = ((time.perf_counter_ns(), time.thread_time_ns())
                 if record_cpu_timing else None)
        self._push(self._tag(event, module))
        try:
            yield
        finally:
            self._pop()
            if start is not None:
                self._record_cpu_timing(event, start, module)

    def install_module_hooks(self) -> None:
        """Install observation and exact-depth display forward hooks."""
        self.remove_module_hooks()
        hook_targets = {}
        for binding, module in self._resolved_modules:
            if binding.entrypoint == "forward":
                hook_targets[binding.module_path] = (
                    module, True,
                    binding.module_path in self._display_module_paths
                )
        wrapper_owned = {
            binding.module_path for binding, _ in self._resolved_modules
            if binding.entrypoint != "forward"
        }
        for name, module in self.model.named_modules():
            if (name in self._display_module_paths
                    and name not in wrapper_owned
                    and name not in hook_targets):
                hook_targets[name] = (module, False, True)

        for path, (module, observe, visible) in hook_targets.items():
            def before(_module: Any, _args: Any, module_path: str = path,
                       record: bool = observe, emit: bool = visible) -> None:
                if record:
                    self._observed_module_bindings.add(module_path)
                if module_path == "data_preprocessor":
                    starts = self._cpu_timing_starts.setdefault(
                        module_path, []
                    )
                    starts.append(
                        (time.perf_counter_ns(), time.thread_time_ns())
                    )
                if emit:
                    self._push(self._tag("module", module_path))

            self._handles.append(module.register_forward_pre_hook(before))
            timed = path == "data_preprocessor"
            if not visible and not timed:
                continue

            def after(_module: Any, _args: Any, output: Any,
                      module_path: str = path, emit: bool = visible,
                      record_timing: bool = timed) -> Any:
                if emit:
                    self._pop()
                if record_timing:
                    starts = self._cpu_timing_starts.get(module_path, [])
                    if starts:
                        self._record_cpu_timing(
                            "module", starts.pop(), module_path
                        )
                self._record_structure(module_path, output)
                return output

            try:
                handle = module.register_forward_hook(after, always_call=True)
            except TypeError:  # older supported PyTorch
                handle = module.register_forward_hook(after)
            self._handles.append(handle)

    def remove_module_hooks(self) -> None:
        """Remove every installed hook."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cpu_timing_starts.clear()

    def run(self, inferencer: Any, inputs: Any, input_id: Any,
            warmup: bool = False) -> Any:
        """Run ``model.test_step``, emitting method ranges only for warmup."""
        if self._closed:
            raise RuntimeError("model profiler is closed")
        self._input_id = str(input_id)
        self.last_completion_monotonic_ns = None
        prior_method_range_state = self._emit_method_ranges
        self._emit_method_ranges = warmup
        self.install_module_hooks()
        try:
            with self.range("preprocess", record_cpu_timing=True):
                prepared = inferencer.preprocess(inputs)
            with self.range("inference", record_cpu_timing=True):
                result = self.model.test_step(prepared)
                stream = self.torch.cuda.default_stream()
                current_stream_fn = getattr(
                    self.torch.cuda, "current_stream", None
                )
                if (
                    current_stream_fn is not None
                    and current_stream_fn() != stream
                ):
                    raise RuntimeError(
                        "unexpected non-default CUDA stream after test_step"
                    )
                event = self.torch.cuda.Event(enable_timing=False)
                event.record(stream)
                event.synchronize()
                self.last_completion_monotonic_ns = time.monotonic_ns()
            return result
        finally:
            self.remove_module_hooks()
            self._emit_method_ranges = prior_method_range_state

    def validate_warmup(self) -> None:
        """Require every present configured binding during warmup."""
        required_methods = {
            binding.method_path for binding, _target, _method
            in self._resolved_methods
        }
        required_modules = {
            binding.module_path for binding, _module in self._resolved_modules
        }
        missing_methods = sorted(
            required_methods.difference(self._observed_method_bindings)
        )
        missing_modules = sorted(
            required_modules.difference(self._observed_module_bindings)
        )
        failures = []
        if missing_methods:
            failures.append("required methods: " + ", ".join(missing_methods))
        if missing_modules:
            failures.append("required modules: " + ", ".join(missing_modules))
        if failures:
            raise ArchitectureProfileError(
                f"profile {self.profile.name!r} warmup did not observe "
                + "; ".join(failures)
            )

    def release_cached_memory(self) -> None:
        """Return unused CUDA allocator memory after synchronized warmup."""
        empty_cache = getattr(self.torch.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def close(self) -> None:
        """Remove hooks and restore wrapped instance attributes exactly."""
        if self._closed:
            return
        self.remove_module_hooks()
        for target, method, had_attribute, prior_value in reversed(
                self._wrapped_attributes):
            if had_attribute:
                setattr(target, method, prior_value)
            else:
                delattr(target, method)
        self._wrapped_attributes.clear()
        self._closed = True


def profile_inference(model: Any, inferencer: Any, inputs: Any,
                      context: Dict[str, Any], module_depth: int = 0,
                      torch_module: Any = None) -> Any:
    """Run one inference through :class:`ModelProfiler`."""
    profiler = ModelProfiler(
        model, context["run_id"], context["scene_token"],
        context["model_id"], context["architecture_profile"], module_depth,
        torch_module
    )
    try:
        return profiler.run(inferencer, inputs, context["input_id"])
    finally:
        profiler.close()
