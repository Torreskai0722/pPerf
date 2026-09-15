"""Tests for architecture-aware NVTX and CUDA completion behavior."""

import json
from types import MethodType, SimpleNamespace

import pytest

from closeloop_profiler.architecture_profiles import ArchitectureProfileError
from closeloop_profiler.profiler import (
    ModelProfiler, encode_tag, selected_module_paths
)
from closeloop_profiler.preprocess_profiler import PreprocessProfiler


IMAGE_PROFILE = "mmdet_two_stage_2d_v1"
LIDAR_PROFILE = "mmdet3d_voxel_two_stage_v1"
POINT_PROFILE = "mmdet3d_point_single_stage_v1"
DETR_PROFILE = "mmdet_detr_2d_v1"
SEG_PROFILE = "mmseg_encoder_decoder_v1"


class Handle:
    """A fake removable hook handle."""

    def __init__(self, remove_callback):
        self._remove_callback = remove_callback
        self.removed = False

    def remove(self):
        """Remove the registered callback once."""
        if not self.removed:
            self._remove_callback()
            self.removed = True


class Module:
    """A small callable module with PyTorch-like forward hooks."""

    def __init__(self):
        self._pre_hooks = []
        self._post_hooks = []

    def register_forward_pre_hook(self, callback):
        """Register a forward pre-hook."""
        self._pre_hooks.append(callback)
        return Handle(lambda: self._pre_hooks.remove(callback))

    def register_forward_hook(self, callback, **_kwargs):
        """Register an always-called forward hook."""
        self._post_hooks.append(callback)
        return Handle(lambda: self._post_hooks.remove(callback))

    def forward(self, value):
        """Return the input unchanged."""
        return value

    def __call__(self, *args):
        """Invoke forward while exercising installed hooks."""
        for callback in list(self._pre_hooks):
            callback(self, args)
        try:
            output = self.forward(*args)
        except Exception:
            for callback in list(self._post_hooks):
                callback(self, args, None)
            raise
        for callback in list(self._post_hooks):
            output = callback(self, args, output)
        return output


class Backbone(Module):
    """A backbone containing one direct child."""

    def __init__(self):
        super().__init__()
        self.layer = Module()

    def forward(self, value):
        """Invoke the direct child."""
        return self.layer(value)


class RpnHead(Module):
    """A proposal head whose predict path enters forward."""

    def __init__(self):
        super().__init__()
        self.block = Module()

    def forward(self, value):
        """Invoke the head's direct child."""
        return self.block(value)

    def predict(self, value):
        """Produce proposals through the module call path."""
        return self(value)

    def predict_by_feat(self, value):
        """Represent a detail method that must remain unwrapped."""
        return value


class RoiHead(Module):
    """An ROI head whose predict path never enters its own forward."""

    def __init__(self):
        super().__init__()
        self.stage = Module()

    def predict(self, value):
        """Produce detections without calling this module's forward."""
        return self.predict_bbox(value)

    def predict_bbox(self, value):
        """Invoke a direct descendant detail stage."""
        return self.stage(value)

    def predict_mask(self, value):
        """Represent another detail method that must remain unwrapped."""
        return value


class TwoStageDetector(Module):
    """A fake accepted MMDetection architecture base."""

    def __init__(self, with_neck=True):
        super().__init__()
        self.data_preprocessor = Module()
        self.backbone = Backbone()
        self.neck = Module() if with_neck else None
        self.rpn_head = RpnHead()
        self.roi_head = RoiHead()

    def named_modules(self):
        """Return the complete fake module hierarchy."""
        modules = [
            ("", self),
            ("data_preprocessor", self.data_preprocessor),
            ("backbone", self.backbone),
            ("backbone.layer", self.backbone.layer),
        ]
        if self.neck is not None:
            modules.append(("neck", self.neck))
        modules.extend([
            ("rpn_head", self.rpn_head),
            ("rpn_head.block", self.rpn_head.block),
            ("roi_head", self.roi_head),
            ("roi_head.stage", self.roi_head.stage),
        ])
        return modules

    def extract_feat(self, value):
        """Run the configured feature modules."""
        features = self.backbone(value)
        if self.neck is not None:
            features = self.neck(features)
        return features

    def predict(self, value):
        """Exercise the two module-owned head entrypoints."""
        features = self.extract_feat(value)
        proposals = self.rpn_head.predict(features)
        return self.roi_head.predict(proposals)

    def test_step(self, value):
        """Follow the production preprocessing-to-predict path."""
        return self.predict(self.data_preprocessor(value))


class Model(TwoStageDetector):
    """A concrete fake detector whose MRO matches the profile."""


class SkippingModel(Model):
    """A detector that does not invoke its required profile modules."""

    def predict(self, value):
        """Skip every inference component."""
        return value


class FailingRpnHead(RpnHead):
    """A proposal head that fails inside its wrapped entrypoint."""

    def predict(self, _value):
        """Fail predictably."""
        raise RuntimeError("inference failed")


class FailingModel(Model):
    """A detector containing a failing proposal head."""

    def __init__(self):
        super().__init__()
        self.rpn_head = FailingRpnHead()


class PtsBBoxHead(Module):
    """A lidar head whose predict path also enters forward."""

    def predict(self, value):
        """Produce predictions through forward."""
        return self(value)

    def predict_by_feat(self, value):
        """Represent a detail method that must remain unwrapped."""
        return value


class MVXTwoStageDetector(Module):
    """A fake accepted MMDetection3D architecture base."""

    def __init__(self):
        super().__init__()
        self.data_preprocessor = Module()
        self.pts_voxel_encoder = Module()
        self.pts_middle_encoder = Module()
        self.pts_backbone = Backbone()
        self.pts_neck = Module()
        self.pts_bbox_head = PtsBBoxHead()

    def named_modules(self):
        """Return the lidar module hierarchy."""
        return [
            ("", self),
            ("data_preprocessor", self.data_preprocessor),
            ("pts_voxel_encoder", self.pts_voxel_encoder),
            ("pts_middle_encoder", self.pts_middle_encoder),
            ("pts_backbone", self.pts_backbone),
            ("pts_backbone.layer", self.pts_backbone.layer),
            ("pts_neck", self.pts_neck),
            ("pts_bbox_head", self.pts_bbox_head),
        ]

    def extract_feat(self, value):
        """Run all lidar feature modules."""
        value = self.pts_voxel_encoder(value)
        value = self.pts_middle_encoder(value)
        value = self.pts_backbone(value)
        return self.pts_neck(value)

    def predict(self, value):
        """Run feature extraction and the bbox head."""
        return self.pts_bbox_head.predict(self.extract_feat(value))

    def test_step(self, value):
        """Follow the production preprocessing-to-predict path."""
        return self.predict(self.data_preprocessor(value))


class LidarModel(MVXTwoStageDetector):
    """A concrete fake lidar detector."""


class PointBBoxHead(Module):
    """A point detector head entered through predict."""

    def predict(self, value):
        """Produce predictions through forward."""
        return self(value)


class SingleStage3DDetector(Module):
    """A fake accepted point-based MMDetection3D architecture base."""

    def __init__(self):
        super().__init__()
        self.data_preprocessor = Module()
        self.backbone = Backbone()
        self.bbox_head = PointBBoxHead()

    def named_modules(self):
        """Return the point detector hierarchy."""
        return [
            ("", self),
            ("data_preprocessor", self.data_preprocessor),
            ("backbone", self.backbone),
            ("backbone.layer", self.backbone.layer),
            ("bbox_head", self.bbox_head),
        ]

    def extract_feat(self, value):
        """Run point feature extraction."""
        return self.backbone(value)

    def predict(self, value):
        """Run feature extraction and the point bbox head."""
        return self.bbox_head.predict(self.extract_feat(value))


class PointModel(SingleStage3DDetector):
    """A concrete fake point-based detector."""


class DetrBBoxHead(Module):
    """A DETR bbox head entered through predict."""

    def predict(self, value):
        """Produce detections through the head forward path."""
        return self(value)


class DetectionTransformer(Module):
    """A fake MMDetection transformer detector base."""

    def __init__(self):
        super().__init__()
        self.data_preprocessor = Module()
        self.backbone = Backbone()
        self.neck = Module()
        self.positional_encoding = Module()
        self.encoder = Module()
        self.decoder = Module()
        self.bbox_head = DetrBBoxHead()

    def named_modules(self):
        """Return the DETR module hierarchy."""
        return [
            ("", self),
            ("data_preprocessor", self.data_preprocessor),
            ("backbone", self.backbone),
            ("backbone.layer", self.backbone.layer),
            ("neck", self.neck),
            ("positional_encoding", self.positional_encoding),
            ("encoder", self.encoder),
            ("decoder", self.decoder),
            ("bbox_head", self.bbox_head),
        ]

    def extract_feat(self, value):
        """Run the DETR image feature stages."""
        return self.neck(self.backbone(value))

    def predict(self, value):
        """Run the transformer and bbox prediction stages."""
        value = self.extract_feat(value)
        value = self.positional_encoding(value)
        value = self.encoder(value)
        value = self.decoder(value)
        return self.bbox_head.predict(value)

    def test_step(self, value):
        """Follow the production preprocessing-to-predict path."""
        return self.predict(self.data_preprocessor(value))


class DetrModel(DetectionTransformer):
    """A concrete fake DETR detector."""


class DecodeHead(Module):
    """A segmentation decode head entered through predict."""

    def predict(self, value):
        """Produce a segmentation map through the module call path."""
        return self(value)


class EncoderDecoder(Module):
    """A fake MMSeg encoder-decoder architecture."""

    def __init__(self, with_neck=True, invoke_neck=True):
        super().__init__()
        self.data_preprocessor = Module()
        self.backbone = Backbone()
        self.neck = Module() if with_neck else None
        self.decode_head = DecodeHead()
        self.invoke_neck = invoke_neck

    def named_modules(self):
        """Return the configured segmentation hierarchy."""
        modules = [
            ("", self),
            ("data_preprocessor", self.data_preprocessor),
            ("backbone", self.backbone),
            ("backbone.layer", self.backbone.layer),
        ]
        if self.neck is not None:
            modules.append(("neck", self.neck))
        modules.append(("decode_head", self.decode_head))
        return modules

    def extract_feat(self, value):
        """Extract backbone and optional neck features."""
        value = self.backbone(value)
        if self.neck is not None and self.invoke_neck:
            value = self.neck(value)
        return value

    def predict(self, value):
        """Decode extracted features."""
        return self.decode_head.predict(self.extract_feat(value))

    def test_step(self, value):
        """Follow the production preprocessing-to-predict path."""
        return self.predict(self.data_preprocessor(value))


class SegModel(EncoderDecoder):
    """A concrete fake semantic segmentation model."""


def fake_torch(actions):
    """Return the CUDA subset used by the profiler."""
    stream = object()

    class Event:
        """Fake CUDA event."""

        def __init__(self, enable_timing):
            assert not enable_timing

        def record(self, selected):
            actions.append(("record", selected))

        def synchronize(self):
            actions.append("event-sync")

    nvtx = SimpleNamespace(
        range_push=lambda tag: actions.append(("push", tag)),
        range_pop=lambda: actions.append("pop"),
        mark=lambda tag: actions.append(("mark", tag)),
    )
    cuda = SimpleNamespace(
        nvtx=nvtx,
        default_stream=lambda: stream,
        current_stream=lambda: stream,
        Event=Event,
        empty_cache=lambda: actions.append("empty-cache"),
    )
    return SimpleNamespace(cuda=cuda)


def test_input_dependent_voxel_and_proposal_counts_are_marked():
    """Fine-grained evidence records dynamic structures without timing them."""
    lidar_actions = []
    lidar = ModelProfiler(
        LidarModel(), "run", "scene", "lidar", LIDAR_PROFILE,
        torch_module=fake_torch(lidar_actions),
    )
    lidar._input_id = "7"
    lidar._record_structure(
        "pts_voxel_encoder", SimpleNamespace(shape=(17, 4))
    )
    lidar.close()
    lidar_tag = json.loads(
        next(value[1] for value in lidar_actions if value[0] == "mark")[
            len("closeloop:"):
        ]
    )
    assert (lidar_tag["structure_kind"], lidar_tag["structure_count"]) == (
        "voxels", 17
    )

    point_actions = []
    point = ModelProfiler(
        PointModel(), "run", "scene", "point", POINT_PROFILE,
        torch_module=fake_torch(point_actions),
    )
    point._input_id = "9"
    point._record_structure(
        "backbone", {"sa_xyz": [SimpleNamespace(shape=(1, 32, 3))]}
    )
    point.close()
    point_tag = json.loads(
        next(value[1] for value in point_actions if value[0] == "mark")[
            len("closeloop:"):
        ]
    )
    assert (point_tag["structure_kind"], point_tag["structure_count"]) == (
        "sampled_points", 32
    )

    image_actions = []
    image = ModelProfiler(
        Model(), "run", "scene", "image", IMAGE_PROFILE,
        torch_module=fake_torch(image_actions),
    )
    image._input_id = "8"
    image._record_structure(
        "rpn_head", [SimpleNamespace(bboxes=range(3)),
                     SimpleNamespace(bboxes=range(2))]
    )
    image.close()
    image_tag = json.loads(
        next(value[1] for value in image_actions if value[0] == "mark")[
            len("closeloop:"):
        ]
    )
    assert (image_tag["structure_kind"], image_tag["structure_count"]) == (
        "proposals", 5
    )


def make_profiler(model, actions, depth=0, profile=IMAGE_PROFILE):
    """Construct a profiler for a fake detector."""
    return ModelProfiler(model, "run", "scene", "model", profile, depth,
                         fake_torch(actions))


def decoded_tags(actions):
    """Decode every pushed closeloop tag."""
    return [json.loads(action[1].removeprefix("closeloop:"))
            for action in actions if isinstance(action, tuple)
            and action[0] == "push"]


def decoded_marks(actions):
    """Decode every closeloop instant mark."""
    return [json.loads(action[1].removeprefix("closeloop:"))
            for action in actions if isinstance(action, tuple)
            and action[0] == "mark"]


def run_once(profiler, warmup=False, input_id=7):
    """Run one fake inference or warmup."""
    inferencer = SimpleNamespace(preprocess=lambda value: value)
    return profiler.run(
        inferencer, "input", input_id, warmup=warmup
    )


def test_release_cached_memory_uses_cuda_allocator():
    """Warmup cache release delegates to the available CUDA allocator."""
    actions = []
    profiler = make_profiler(Model(), actions)

    profiler.release_cached_memory()

    assert actions == ["empty-cache"]


def module_tags(actions):
    """Return only module range tags."""
    return [tag for tag in decoded_tags(actions) if tag["event"] == "module"]


def test_exact_depth_selection_and_tag_stability():
    """Depth selects exact profile-rooted structure and excludes model root."""
    model = Model()
    roots = ["data_preprocessor", "backbone", "neck",
             "rpn_head", "roi_head"]
    assert selected_module_paths(model, 0, roots) == roots
    assert selected_module_paths(model, 1, roots) == [
        "backbone.layer", "rpn_head.block", "roi_head.stage"
    ]
    assert "" not in selected_module_paths(model, 0, roots)
    assert selected_module_paths(model, 2, roots) == []
    first = encode_tag("run", "scene", "model", "1", "inference")
    assert first == encode_tag("run", "scene", "model", "1", "inference")
    assert first.startswith("closeloop:")


def test_depth_zero_emits_only_present_profile_modules():
    """Depth zero emits the complete present two-stage image profile."""
    actions = []
    profiler = make_profiler(Model(), actions)
    run_once(profiler)
    profiler.validate_warmup()
    assert [tag["module"] for tag in module_tags(actions)] == [
        "data_preprocessor", "backbone", "neck", "rpn_head", "roi_head"
    ]


def test_preprocess_profiler_wraps_and_restores_runtime_pipeline():
    """Targeted ranges cover transform objects without editing MMlab source."""
    actions = []

    class Transform:
        """MMCV-style transform."""

        def transform(self, value):
            return value + ["method"]

        def __call__(self, value):
            return self.transform(value)

    class FunctionTransform:
        """Function-style callable transform."""

        def __call__(self, value):
            return value + ["callable"]

    class Pipeline:
        """Small Compose stand-in."""

        def __init__(self):
            self.transforms = [Transform(), FunctionTransform()]

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    model = Model()
    pipeline = Pipeline()
    originals = list(pipeline.transforms)
    raw = SimpleNamespace(pipeline=pipeline)
    adapter = SimpleNamespace(
        inferencer=raw,
        model=model,
        preprocess=lambda value: pipeline(value),
    )
    profiler = PreprocessProfiler(
        model, "run", "scene", "model", IMAGE_PROFILE, 0,
        fake_torch(actions), inferencer=adapter,
    )
    profiler.run(adapter, [], 7)
    step_tags = [
        tag for tag in decoded_tags(actions)
        if tag["event"] == "preprocess_step"
    ]
    assert len(step_tags) == 2
    assert all("pipeline.transforms" in tag["module"]
               for tag in step_tags)
    assert len(profiler.observed_preprocess_steps) == 2
    assert len(profiler.pipeline_manifest) == 2
    profiler.close()
    assert pipeline.transforms == originals
    assert "transform" not in originals[0].__dict__
    assert profiler.observed_module_bindings == [
        "backbone", "data_preprocessor", "neck", "roi_head", "rpn_head"
    ]
    profiler.close()


def test_preprocess_profiler_handles_nested_compose_pipeline():
    """Nested transforms may store a non-iterable Compose container."""
    actions = []

    class Transform:
        """MMCV-style transform."""

        def transform(self, value):
            return value + ["method"]

        def __call__(self, value):
            return self.transform(value)

    class FunctionTransform:
        """Function-style callable transform."""

        def __call__(self, value):
            return value + ["callable"]

    class Compose:
        """MMEngine-style callable container that is not itself iterable."""

        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    class NestedTransform:
        """Transform whose child transforms are held by Compose."""

        def __init__(self):
            self.transforms = Compose([Transform(), FunctionTransform()])

        def transform(self, value):
            return self.transforms(value)

        def __call__(self, value):
            return self.transform(value)

    model = Model()
    pipeline = Compose([NestedTransform()])
    original_parent = pipeline.transforms[0]
    original_children = list(original_parent.transforms.transforms)
    raw = SimpleNamespace(pipeline=pipeline)
    adapter = SimpleNamespace(
        inferencer=raw,
        model=model,
        preprocess=lambda value: pipeline(value),
    )
    profiler = PreprocessProfiler(
        model, "run", "scene", "model", IMAGE_PROFILE, 0,
        fake_torch(actions), inferencer=adapter,
    )

    profiler.run(adapter, [], 7)

    step_tags = [
        tag for tag in decoded_tags(actions)
        if tag["event"] == "preprocess_step"
    ]
    assert len(step_tags) == 3
    assert len(profiler.observed_preprocess_steps) == 3
    assert len(profiler.pipeline_manifest) == 3
    assert any(".transforms[0]" in tag["module"] for tag in step_tags)
    assert any(".transforms[1]" in tag["module"] for tag in step_tags)

    profiler.close()
    assert pipeline.transforms[0] is original_parent
    assert original_parent.transforms.transforms == original_children
    assert "transform" not in original_parent.__dict__
    assert "transform" not in original_children[0].__dict__


def test_depth_one_emits_only_invoked_direct_descendants():
    """Positive depth omits profile parents but preserves their observation."""
    actions = []
    profiler = make_profiler(Model(), actions, 1)
    run_once(profiler)
    profiler.validate_warmup()
    assert [tag["module"] for tag in module_tags(actions)] == [
        "backbone.layer", "rpn_head.block", "roi_head.stage"
    ]
    assert "backbone" in profiler.observed_module_bindings
    assert "rpn_head" in profiler.observed_module_bindings
    profiler.close()


def test_direct_roi_predict_is_one_full_stage_module_range():
    """A head entered without forward still has one module-owned range."""
    actions = []
    model = Model()
    profiler = make_profiler(model, actions)
    assert model.roi_head.predict("features") == "features"
    tags = module_tags(actions)
    assert [tag["module"] for tag in tags] == ["roi_head"]
    assert "method" not in tags[0]
    assert "owner" not in tags[0]
    assert "invocation" not in tags[0]
    profiler.close()


def test_head_wrappers_do_not_duplicate_forward_owner_ranges():
    """Predict wrappers own rpn and lidar bbox stages even if forward runs."""
    actions = []
    image_profiler = make_profiler(Model(), actions)
    run_once(image_profiler)
    assert [tag["module"] for tag in module_tags(actions)].count(
        "rpn_head") == 1
    image_profiler.close()

    actions = []
    lidar_profiler = make_profiler(
        LidarModel(), actions, profile=LIDAR_PROFILE
    )
    run_once(lidar_profiler)
    lidar_profiler.validate_warmup()
    assert [tag["module"] for tag in module_tags(actions)].count(
        "pts_bbox_head") == 1
    lidar_profiler.close()


def test_detr_profile_emits_transformer_roots_without_head_duplicates():
    """DETR uses generic profile bindings without profiler specialization."""
    actions = []
    profiler = make_profiler(
        DetrModel(), actions, profile=DETR_PROFILE
    )
    run_once(profiler)
    profiler.validate_warmup()
    modules = [tag["module"] for tag in module_tags(actions)]
    assert modules == [
        "data_preprocessor", "backbone", "neck",
        "positional_encoding", "encoder", "decoder", "bbox_head",
    ]
    assert modules.count("bbox_head") == 1
    assert not [tag for tag in decoded_tags(actions)
                if tag["event"] == "method"]
    profiler.close()


def test_mmseg_profile_emits_present_inference_roots():
    """EncoderDecoder emits its optional neck only when configured."""
    actions = []
    profiler = make_profiler(
        SegModel(), actions, profile=SEG_PROFILE
    )
    run_once(profiler)
    profiler.validate_warmup()
    assert [tag["module"] for tag in module_tags(actions)] == [
        "data_preprocessor", "backbone", "neck", "decode_head",
    ]
    profiler.close()

    actions = []
    profiler = make_profiler(
        SegModel(with_neck=False), actions, profile=SEG_PROFILE
    )
    run_once(profiler)
    profiler.validate_warmup()
    assert [tag["module"] for tag in module_tags(actions)] == [
        "data_preprocessor", "backbone", "decode_head",
    ]
    profiler.close()


def test_optional_present_module_must_be_observed_during_warmup():
    """An optional path may be absent but cannot be silently bypassed."""
    actions = []
    profiler = make_profiler(
        SegModel(invoke_neck=False), actions, profile=SEG_PROFILE
    )
    run_once(profiler)
    with pytest.raises(ArchitectureProfileError, match="neck"):
        profiler.validate_warmup()
    profiler.close()


def test_method_ranges_are_limited_to_model_methods_during_warmup():
    """Actual inference has no method ranges; warmup has only model methods."""
    actions = []
    profiler = make_profiler(Model(), actions)
    run_once(profiler)
    assert not [tag for tag in decoded_tags(actions)
                if tag["event"] == "method"]
    assert profiler.observed_method_bindings == [
        "model.extract_feat", "model.predict"
    ]
    assert profiler.invocations[("7", "", "predict")] == 1

    actions.clear()
    run_once(profiler, warmup=True, input_id="warmup-0")
    methods = [tag["method"] for tag in decoded_tags(actions)
               if tag["event"] == "method"]
    assert methods == ["model.predict", "model.extract_feat"]
    assert not {"rpn_head.predict_by_feat", "roi_head.predict_bbox",
                "roi_head.predict_mask"}.intersection(methods)
    profiler.close()


def test_ranges_warmup_completion_and_cleanup():
    """Inference records CPU timing evidence and synchronizes its event."""
    actions = []
    profiler = make_profiler(Model(), actions, 2)
    assert run_once(profiler) == "input"
    profiler.validate_warmup()
    assert "event-sync" in actions
    assert profiler._handles == []
    marks = decoded_marks(actions)
    assert [mark["timed_event"] for mark in marks] == [
        "preprocess", "module", "inference"
    ]
    assert marks[1]["module"] == "data_preprocessor"
    for mark in marks:
        assert mark["event"] == "cpu_timing"
        assert mark["wall_time_ns"] >= mark["thread_time_ns"]
        assert mark["non_cpu_time_ns"] == (
            mark["wall_time_ns"] - mark["thread_time_ns"]
        )
    profiler.close()


def test_exception_balances_ranges_and_removes_hooks():
    """A wrapped head failure closes all ranges and removes hooks."""
    actions = []
    profiler = make_profiler(FailingModel(), actions)
    with pytest.raises(RuntimeError, match="inference failed"):
        run_once(profiler)
    pushes = [action for action in actions
              if isinstance(action, tuple) and action[0] == "push"]
    assert actions.count("pop") == len(pushes)
    assert profiler._handles == []
    profiler.close()


def test_rejects_mro_required_module_and_required_entrypoint():
    """Construction rejects each required structural mismatch."""
    actions = []

    class Wrong(Module):
        """A model with the wrong architecture ancestry."""

    with pytest.raises(ArchitectureProfileError, match="MRO"):
        make_profiler(Wrong(), actions)
    missing_module = Model()
    missing_module.backbone = None
    with pytest.raises(ArchitectureProfileError, match="backbone"):
        make_profiler(missing_module, actions)
    missing_entrypoint = Model()
    missing_entrypoint.rpn_head.predict = None
    with pytest.raises(ArchitectureProfileError, match="rpn_head.predict"):
        make_profiler(missing_entrypoint, actions)


def test_optional_module_is_skipped_and_warmup_checks_required_modules():
    """A missing neck is valid while uncalled required modules fail warmup."""
    actions = []
    model = Model(with_neck=False)
    profiler = make_profiler(model, actions)
    run_once(profiler)
    profiler.validate_warmup()
    assert "neck" not in profiler.observed_module_bindings
    assert "neck" not in [tag["module"] for tag in module_tags(actions)]
    profiler.close()

    profiler = make_profiler(SkippingModel(), actions, 2)
    run_once(profiler)
    with pytest.raises(ArchitectureProfileError, match="required modules"):
        profiler.validate_warmup()
    profiler.close()


def test_close_restores_method_and_module_wrappers_exactly():
    """Close restores overrides and inherited descriptors for every wrapper."""
    actions = []
    model = Model()

    def override(self, value):
        return TwoStageDetector.predict(self, value)

    prior = MethodType(override, model)
    model.predict = prior
    profiler = make_profiler(model, actions)
    assert model.__dict__["predict"] is not prior
    assert "predict" in model.rpn_head.__dict__
    assert "predict" in model.roi_head.__dict__
    profiler.close()
    profiler.close()
    assert model.__dict__["predict"] is prior
    assert "predict" not in model.rpn_head.__dict__
    assert "predict" not in model.roi_head.__dict__
