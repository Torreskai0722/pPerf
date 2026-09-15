"""Targeted preprocessing instrumentation for closed-loop inference."""

from functools import wraps
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .profiler import ModelProfiler


DEFAULT_DATA_PREPROCESSOR_METHODS = (
    "cast_data",
    "voxelize",
    "stack_batch",
    "pad_gt_masks",
    "pad_gt_sem_seg",
)


class _TransformProxy:
    """Add one profiler range around a function-style pipeline transform."""

    def __init__(self, target: Any, profiler: "PreprocessProfiler",
                 label: str):
        self.target = target
        self.profiler = profiler
        self.label = label

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.profiler._observed_preprocess_steps.add(self.label)
        with self.profiler.range(
                "preprocess_step", self.label, record_cpu_timing=True):
            return self.target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def __repr__(self) -> str:
        return repr(self.target)


def _qualified_name(value: Any) -> str:
    target = type(value)
    return f"{target.__module__}.{target.__qualname__}"


class PreprocessProfiler(ModelProfiler):
    """Profile pipeline transforms and data-preprocessor callable boundaries."""

    profiler_id = "preprocess_contention_v1"

    def __init__(
            self, *args: Any, inferencer: Any,
            options: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.inferencer = inferencer
        self.options = dict(options or {})
        self._observed_preprocess_steps: Set[str] = set()
        self._pipeline_replacements: List[Tuple[List[Any], int, Any]] = []
        self._step_attributes: List[Tuple[Any, str, bool, Any]] = []
        self._instrumented_methods: Set[Tuple[int, str]] = set()
        self._pipeline_manifest: List[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        try:
            self._install_preprocess_instrumentation()
        except Exception:
            self.close()
            raise

    @property
    def observed_preprocess_steps(self) -> List[str]:
        """Return preprocessing steps observed since construction."""
        return sorted(self._observed_preprocess_steps)

    @property
    def pipeline_manifest(self) -> List[Dict[str, Any]]:
        """Return the discovered pipeline and data-preprocessor callables."""
        return [dict(item) for item in self._pipeline_manifest]

    def _raw_inferencer(self) -> Any:
        return getattr(self.inferencer, "inferencer", self.inferencer)

    def _pipelines(self) -> Iterable[Tuple[str, Any]]:
        raw = self._raw_inferencer()
        seen = set()
        for owner_name, owner in (("inferencer", raw),
                                  ("adapter", self.inferencer)):
            for attribute in ("pipeline", "inference_pipeline"):
                pipeline = getattr(owner, attribute, None)
                if pipeline is None or id(pipeline) in seen:
                    continue
                if not hasattr(pipeline, "transforms"):
                    continue
                seen.add(id(pipeline))
                yield f"{owner_name}.{attribute}", pipeline

    def _wrap_method(self, target: Any, method_name: str, label: str,
                     event: str) -> None:
        key = (id(target), method_name)
        if key in self._instrumented_methods:
            return
        original = getattr(target, method_name)
        instance_attributes = getattr(target, "__dict__", {})
        had_attribute = method_name in instance_attributes
        prior_value = instance_attributes.get(method_name)

        @wraps(original)
        def wrapped(_instance: Any, *args: Any,
                    _original: Any = original,
                    _label: str = label, **kwargs: Any) -> Any:
            self._observed_preprocess_steps.add(_label)
            with self.range(
                    event, _label, record_cpu_timing=True):
                return _original(*args, **kwargs)

        setattr(target, method_name, MethodType(wrapped, target))
        self._step_attributes.append(
            (target, method_name, had_attribute, prior_value)
        )
        self._instrumented_methods.add(key)

    def _instrument_pipeline(self, prefix: str, pipeline: Any) -> None:
        transforms = pipeline.transforms
        while hasattr(transforms, "transforms"):
            transforms = transforms.transforms
        for index, transform in enumerate(list(transforms)):
            label = f"{prefix}[{index}]:{_qualified_name(transform)}"
            self._pipeline_manifest.append({
                "kind": "pipeline_transform",
                "label": label,
                "callable": _qualified_name(transform),
            })
            nested = getattr(transform, "transforms", None)
            if nested is not None:
                self._instrument_pipeline(label + ".transforms", transform)
            method = getattr(transform, "transform", None)
            if callable(method):
                self._wrap_method(
                    transform, "transform", label, "preprocess_step"
                )
                continue
            proxy = _TransformProxy(transform, self, label)
            transforms[index] = proxy
            self._pipeline_replacements.append((transforms, index, transform))

    def _instrument_data_preprocessor(self) -> None:
        data_preprocessor = getattr(self.model, "data_preprocessor", None)
        if data_preprocessor is None:
            return
        configured = self.options.get(
            "data_preprocessor_methods",
            list(DEFAULT_DATA_PREPROCESSOR_METHODS),
        )
        for method_name in configured:
            method = getattr(data_preprocessor, method_name, None)
            if not callable(method):
                continue
            label = (
                "data_preprocessor."
                f"{method_name}:{_qualified_name(data_preprocessor)}"
            )
            self._pipeline_manifest.append({
                "kind": "data_preprocessor_method",
                "label": label,
                "callable": (
                    f"{_qualified_name(data_preprocessor)}.{method_name}"
                ),
            })
            self._wrap_method(
                data_preprocessor, method_name, label,
                "data_preprocessor_step",
            )

    def _install_preprocess_instrumentation(self) -> None:
        for prefix, pipeline in self._pipelines():
            self._instrument_pipeline(prefix + ".transforms", pipeline)
        self._instrument_data_preprocessor()

    def close(self) -> None:
        """Restore preprocessing callables and base-profiler instrumentation."""
        for transforms, index, original in reversed(
                self._pipeline_replacements):
            transforms[index] = original
        self._pipeline_replacements.clear()
        for target, method, had_attribute, prior_value in reversed(
                self._step_attributes):
            if had_attribute:
                setattr(target, method, prior_value)
            else:
                try:
                    delattr(target, method)
                except AttributeError:
                    pass
        self._step_attributes.clear()
        self._instrumented_methods.clear()
        super().close()
