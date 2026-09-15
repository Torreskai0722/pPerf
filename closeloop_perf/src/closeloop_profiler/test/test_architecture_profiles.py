"""Tests for immutable, versioned architecture profile definitions."""

import hashlib
import json

import pytest
import yaml

from closeloop_profiler.architecture_profiles import (
    ARCHITECTURE_PROFILES, ArchitectureProfileError,
    get_architecture_profile, load_architecture_profiles
)


def test_registry_lookup_is_immutable_and_unknown_names_fail():
    """Registry entries resolve exactly and callers cannot replace them."""
    profile = get_architecture_profile("mmdet_two_stage_2d_v1")
    assert profile is ARCHITECTURE_PROFILES[profile.name]
    with pytest.raises(TypeError):
        ARCHITECTURE_PROFILES["other"] = profile
    with pytest.raises(ArchitectureProfileError, match="unknown"):
        get_architecture_profile("unknown")


def test_profile_hash_uses_canonical_complete_definition():
    """The reported digest matches canonical JSON and remains deterministic."""
    profile = get_architecture_profile("mmdet3d_voxel_two_stage_v1")
    encoded = json.dumps(
        profile.definition(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert profile.sha256 == hashlib.sha256(encoded).hexdigest()
    assert profile.sha256 == get_architecture_profile(profile.name).sha256


def test_v1_profiles_define_exact_module_entrypoints_and_hashes():
    """Current profile names bind their complete architecture definitions."""
    image = get_architecture_profile("mmdet_two_stage_2d_v1")
    assert [(binding.module_path, binding.required, binding.entrypoint)
            for binding in image.module_bindings] == [
        ("data_preprocessor", True, "forward"),
        ("backbone", True, "forward"),
        ("neck", False, "forward"),
        ("rpn_head", True, "predict"),
        ("roi_head", True, "predict"),
    ]
    assert image.sha256 == (
        "3d6a6eea4ff948452a1f43d55901ccc5f36345dc73545c4738f2756600cb1929"
    )

    lidar = get_architecture_profile("mmdet3d_voxel_two_stage_v1")
    assert [(binding.module_path, binding.required, binding.entrypoint)
            for binding in lidar.module_bindings] == [
        ("data_preprocessor", True, "forward"),
        ("pts_voxel_encoder", True, "forward"),
        ("pts_middle_encoder", True, "forward"),
        ("pts_backbone", True, "forward"),
        ("pts_neck", False, "forward"),
        ("pts_bbox_head", True, "predict"),
    ]
    assert lidar.sha256 == (
        "bcbe6ca936bb2facce3f19f211b6e98873825b4117fcde2672b5eb5a30d9778b"
    )

    for profile in (image, lidar):
        methods = [
            binding.method_path for binding in profile.method_bindings
        ]
        assert methods == [
            "model.predict", "model.extract_feat"
        ]


def test_detr_profile_matches_mmdetection_execution_components():
    """DETR binds its transformer stages and predict-owned bbox head."""
    profile = get_architecture_profile("mmdet_detr_2d_v1")
    assert profile.accepted_mro == ("DetectionTransformer",)
    assert [(binding.module_path, binding.required, binding.entrypoint)
            for binding in profile.module_bindings] == [
        ("data_preprocessor", True, "forward"),
        ("backbone", True, "forward"),
        ("neck", True, "forward"),
        ("positional_encoding", True, "forward"),
        ("encoder", True, "forward"),
        ("decoder", True, "forward"),
        ("bbox_head", True, "predict"),
    ]
    assert [binding.method_path for binding in profile.method_bindings] == [
        "model.predict", "model.extract_feat"
    ]
    assert profile.sha256 == (
        "c561b4120f41d381cc0f868dfde3ae289e455e9419b56b218e5b173d7de5f290"
    )


def test_mmseg_profile_matches_encoder_decoder_inference_components():
    """MMSeg inference binds its backbone, optional neck, and decode head."""
    profile = get_architecture_profile("mmseg_encoder_decoder_v1")
    assert profile.accepted_mro == ("EncoderDecoder",)
    assert [(binding.module_path, binding.required, binding.entrypoint)
            for binding in profile.module_bindings] == [
        ("data_preprocessor", True, "forward"),
        ("backbone", True, "forward"),
        ("neck", False, "forward"),
        ("decode_head", True, "predict"),
    ]


def test_single_stage_profile_covers_yolo_inference_components():
    """YOLO uses its native single-stage backbone/neck/head structure."""
    profile = get_architecture_profile("mmdet_single_stage_2d_v1")
    assert profile.accepted_mro == ("SingleStageDetector",)
    assert [binding.module_path for binding in profile.module_bindings] == [
        "data_preprocessor", "backbone", "neck", "bbox_head"
    ]


def test_profile_loader_rejects_filename_and_duplicate_bindings(tmp_path):
    """External profile files have stable names and unique binding targets."""
    data = {
        "schema_version": 1,
        "name": "profile_v1",
        "accepted_mro": ["Model"],
        "method_bindings": [{
            "owner": "model",
            "target_path": "",
            "method": "predict",
            "required": True,
        }],
        "module_bindings": [{
            "module_path": "backbone",
            "required": True,
            "entrypoint": "forward",
        }],
    }
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArchitectureProfileError, match="filename"):
        load_architecture_profiles(tmp_path)

    path = tmp_path / "profile_v1.yaml"
    (tmp_path / "wrong.yaml").unlink()
    data["module_bindings"].append(dict(data["module_bindings"][0]))
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArchitectureProfileError, match="duplicate module"):
        load_architecture_profiles(tmp_path)
