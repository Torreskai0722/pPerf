"""Create deterministic, paired camera/LiDAR weather MCAPs offline."""

import argparse
from array import array
import bisect
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from xml.etree import ElementTree


CAMERA_TOPIC = "/CAM_FRONT/image_rect_compressed"
LIDAR_TOPIC = "/LIDAR_TOP"
WEATHER_TYPES = ("rain", "snow")
FIXED_POINT_COUNT = 34_752
FIXED_FEATURE_COUNT = 5
FIXED_POINT_STEP = FIXED_FEATURE_COUNT * 4
RAIN_RATES = (0.20, 0.73, 1.5625, 3.125, 7.29)
MAX_PAIR_DELTA_NS = 50_000_000
ASTRA_SEQUENCE = "scene"
ASTRA_CACHE_ROOT = Path("/mmdetection3d_ros2/data/astra_particles")
ASTRA_COMMIT = "0a9d8ca739d71d4b09f61567c6f6cea94a79868e"
WEATHER_SIMULATOR_COMMIT = (
    "19fa03b419a32902abbf0ec899ff0ca6e77b56fa"
)
RAIN_STREAK_ARCHIVE_SHA256 = (
    "b996db425aa28fee6abda3163fb4d027587bc59e3909daa45da5f495f419fd9d"
)
ASTRA_CAMERA = {
    "width": 1600,
    "height": 900,
    "nominal_hz": 12.0,
    "focal_length_mm": 5.5,
    "f_number": 1.8,
    "exposure_ms": 5.0,
    "gain": 1.0,
}
DEPTH_COMPLETION = {
    "max_depth_m": 100.0,
    "dilation_kernel": [5, 5],
    "closing_kernel": [5, 5],
    "small_hole_kernel": [7, 7],
    "large_hole_kernel": [31, 31],
    "column_extrapolation": True,
    "median_kernel": 5,
    "bilateral_diameter": 5,
    "bilateral_sigma_color": 1.5,
    "bilateral_sigma_space": 2.0,
}
_LEGACY_LISA = None
_LEGACY_LISA_MODULE = None
_LEGACY_LISA_ALPHA = {}
_PYLISA = None


class CorruptionError(RuntimeError):
    """Raised when an offline corruption would be incomplete or ambiguous."""


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base_seed, topic, timestamp_ns, index=0):
    """Derive a stable uint32 seed for one timestamped sensor payload."""
    value = f"{int(base_seed)}:{topic}:{int(timestamp_ns)}:{int(index)}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def rain_rate_slug(value):
    """Return a canonical path component such as ``7p5`` for a rain rate."""
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CorruptionError(f"invalid rain rate: {value}") from exc
    if not rate.is_finite() or rate <= 0:
        raise CorruptionError("rain rate must be finite and positive")
    return format(rate.normalize(), "f").replace(".", "p")


def resolve_astra_cache(scene_name, override=None):
    """Return an isolated particle-cache path for one nuScenes scene."""
    if (not isinstance(scene_name, str) or not scene_name
            or scene_name in (".", "..")
            or Path(scene_name).name != scene_name or "\\" in scene_name):
        raise CorruptionError(f"invalid scene name: {scene_name}")
    if override is not None:
        if not str(override).strip():
            raise CorruptionError("Astra cache path must not be empty")
        return Path(override).resolve()
    return ASTRA_CACHE_ROOT / scene_name


def paired_timestamps(camera_timestamps, lidar_timestamps,
                      tolerance_ns=MAX_PAIR_DELTA_NS):
    """Pair every LiDAR timestamp to its nearest camera within a limit."""
    cameras = sorted(map(int, camera_timestamps))
    if not cameras:
        return []
    result = []
    for lidar in sorted(map(int, lidar_timestamps)):
        index = bisect.bisect_left(cameras, lidar)
        candidates = cameras[max(0, index - 1):index + 1]
        camera = min(candidates, key=lambda value: (abs(value - lidar), value))
        if abs(camera - lidar) <= tolerance_ns:
            result.append((camera, lidar))
    return result


def pair_camera_lidar_timestamps(camera_timestamps, lidar_timestamps,
                                 tolerance_ns=MAX_PAIR_DELTA_NS):
    """Match every camera header time to one nearest clean LiDAR scan."""
    lidars = sorted(map(int, lidar_timestamps))
    if not lidars:
        raise CorruptionError("source bag has no LIDAR_TOP header timestamps")
    pairs = []
    for camera in sorted(map(int, camera_timestamps)):
        index = bisect.bisect_left(lidars, camera)
        candidates = lidars[max(0, index - 1):index + 1]
        lidar = min(candidates, key=lambda value: (abs(value - camera), value))
        delta = abs(camera - lidar)
        if delta > tolerance_ns:
            raise CorruptionError(
                f"CAM_FRONT {camera} nearest LIDAR_TOP is {delta} ns away; "
                f"limit is {tolerance_ns} ns"
            )
        pairs.append((camera, lidar, delta))
    return pairs


def validate_manifest(value):
    """Validate provenance, timestamps, pairing, and full-bag integrity."""
    errors = []
    if value.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    corruption = value.get("corruption", {})
    kind = corruption.get("type")
    severity = corruption.get("severity")
    full_clean = (
        kind == "clean" and corruption.get("full_duration") is True
    )
    full_fixed = (
        kind == "fixed" and corruption.get("full_duration") is True
    )
    full_rain = kind == "rain" and corruption.get("full_duration") is True
    full_duration = full_clean or full_fixed or full_rain
    if kind == "clean":
        if severity != 0:
            errors.append("clean data requires severity 0")
    elif kind == "fixed":
        if severity != 0 or not full_fixed:
            errors.append("fixed data requires severity 0 and full duration")
    elif kind == "snow":
        if severity not in range(1, 6):
            errors.append("snow data requires severity 1-5")
    elif full_rain:
        rate = corruption.get("rain_rate_mm_per_hour")
        if (severity is not None
                or not isinstance(rate, (int, float)) or rate <= 0):
            errors.append("full rain data requires a positive physical rate")
    elif kind == "rain":
        if severity not in range(1, 6):
            errors.append("legacy rain data requires severity 1-5")
    else:
        errors.append("unsupported corruption type")

    records = value.get("sensor_records", [])
    topics = {record.get("topic") for record in records}
    if topics != {CAMERA_TOPIC, LIDAR_TOPIC}:
        errors.append("sensor records must cover camera and LiDAR")
    if any(record.get("bag_timestamp_ns") !=
           record.get("output_bag_timestamp_ns") for record in records):
        errors.append("a bag timestamp changed")
    if any(record.get("header_timestamp_ns") !=
           record.get("output_header_timestamp_ns") for record in records):
        errors.append("a sensor header timestamp changed")
    if value.get("preserved_stream_source_sha256") != value.get(
            "preserved_stream_output_sha256"):
        errors.append("an unselected message changed")

    camera = [record for record in records
              if record.get("topic") == CAMERA_TOPIC]
    lidar = [record for record in records
             if record.get("topic") == LIDAR_TOPIC]
    if full_duration:
        duration_ns = value.get("duration_ns")
        window_start = value.get("window_start_bag_timestamp_ns")
        window_end = value.get("window_end_bag_timestamp_ns")
        if not isinstance(duration_ns, int) or duration_ns <= 0:
            errors.append("full-duration bag duration must be positive")
        elif (not isinstance(window_start, int)
              or not isinstance(window_end, int)
              or window_end - window_start != duration_ns):
            errors.append(
                "full-duration differs from bag timestamp window"
            )
        elif value.get("duration_seconds") != duration_ns / 1_000_000_000:
            errors.append("full-duration seconds differ")
        source_count = value.get("source_message_count")
        if not isinstance(source_count, int) or source_count < len(records):
            errors.append("full-duration source message count is invalid")
        pairing = value.get("pairing", {})
        if pairing.get("camera_messages") != len(camera):
            errors.append("recorded camera message count differs")
        if pairing.get("lidar_messages") != len(lidar):
            errors.append("recorded LiDAR message count differs")
        if pairing.get("paired_camera_messages") != len(camera):
            errors.append("not every camera has a clean LiDAR depth source")
        if pairing.get("depth_pair_count") != len(camera):
            errors.append("camera/LiDAR depth pairing count differs")
        deltas = [record.get("lidar_pairing_delta_ns") for record in camera]
        if any(record.get("paired_lidar_header_timestamp_ns") is None
               for record in camera) or any(
                   not isinstance(delta, int) or delta < 0
                   or delta > MAX_PAIR_DELTA_NS for delta in deltas):
            errors.append("a camera/LiDAR depth pairing exceeds 50 ms")
        if pairing.get("tolerance_ns") != MAX_PAIR_DELTA_NS:
            errors.append("camera/LiDAR pairing tolerance differs")
        if deltas and pairing.get("maximum_delta_ns") != max(deltas):
            errors.append("recorded maximum camera/LiDAR delta differs")
        depth_sources = {
            record.get("paired_lidar_header_timestamp_ns")
            for record in camera
        }
        if pairing.get("unique_depth_source_lidar_messages") != len(
                depth_sources):
            errors.append("recorded unique depth-source count differs")
    if full_clean:
        if value.get("output_message_count") != value.get(
                "source_message_count"):
            errors.append("full clean message count changed")
        if value.get("source_stream_sha256") != value.get(
                "output_stream_sha256"):
            errors.append("full clean stream changed")
        if any(record.get("source_payload_sha256") !=
               record.get("output_payload_sha256") for record in records):
            errors.append("a full clean sensor payload changed")
        output_files = value.get("output_bag_files", [])
        if (len(output_files) != 1
                or output_files[0].get("sha256") != value.get(
                    "source_bag_sha256")):
            errors.append("full clean MCAP differs from its source")
    if full_rain:
        if any(record.get("source_payload_sha256") ==
               record.get("output_payload_sha256") for record in records):
            errors.append("a selected rain payload did not change")
        if any(record.get("metadata_sample_data_token") is None
               for record in records):
            errors.append("a selected message lacks exact nuScenes metadata")
        if any(record.get("dense_depth_coverage") != 1.0
               for record in camera):
            errors.append("a camera dense depth map is incomplete")
        if any(record.get("output_points_finite") is not True
               for record in lidar):
            errors.append("a LiDAR output contains a nonfinite point")
    if full_fixed:
        assets = value.get("fixed_assets", {})
        image = assets.get("camera", {})
        points = assets.get("lidar", {})
        image_hash = image.get("sha256")
        lidar_hash = points.get("sha256")
        if (not isinstance(image_hash, str) or len(image_hash) != 64
                or not isinstance(lidar_hash, str) or len(lidar_hash) != 64):
            errors.append("fixed asset hashes are invalid")
        if (points.get("point_count") != FIXED_POINT_COUNT
                or points.get("feature_count") != FIXED_FEATURE_COUNT
                or points.get("dtype") != "float32"
                or points.get("finite") is not True):
            errors.append("fixed LiDAR asset layout is invalid")
        if any(record.get("fixed_content_sha256") != image_hash
               or record.get("output_content_sha256") != image_hash
               for record in camera):
            errors.append("a camera does not contain the fixed image")
        if any(record.get("fixed_content_sha256") != lidar_hash
               or record.get("output_content_sha256") != lidar_hash
               for record in lidar):
            errors.append("a LiDAR message does not contain the fixed points")
        if any(record.get("output_point_count") != FIXED_POINT_COUNT
               or record.get("output_height") != 1
               or record.get("output_width") != FIXED_POINT_COUNT
               or record.get("output_point_step") != FIXED_POINT_STEP
               or record.get("output_row_step")
               != FIXED_POINT_COUNT * FIXED_POINT_STEP
               or record.get("output_points_finite") is not True
               for record in lidar):
            errors.append("a fixed PointCloud2 layout is invalid")
        if value.get("output_message_count") != value.get(
                "source_message_count"):
            errors.append("fixed message count changed")
    if not full_duration:
        camera_times = [record["bag_timestamp_ns"] for record in camera]
        lidar_times = [record["bag_timestamp_ns"] for record in lidar]
        tolerance = value.get("pairing", {}).get(
            "tolerance_ns", 100_000_000
        )
        pairs = paired_timestamps(camera_times, lidar_times, tolerance)
        if len(pairs) != len(lidar_times):
            errors.append("camera/LiDAR timestamps are not fully paired")
        expected = value.get("pairing", {}).get("paired_lidar_messages")
        if expected != len(pairs):
            errors.append("recorded camera/LiDAR pairing count differs")
        if value.get("duration_seconds") != 10.0:
            errors.append(
                "experiment bags must contain exactly 10 source seconds"
            )
    return errors


def _git_output(root, *arguments):
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         *arguments], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _load_camera_module(root):
    """Import the retained snow implementation despite an obsolete import."""
    import numpy as np
    import scipy.integrate
    scipy.integrate.trapz = getattr(scipy.integrate, "trapz", np.trapz)
    root = Path(root).resolve()
    sys.path[:0] = [str(root), str(root / "utils")]
    modules = {
        name: types.ModuleType(name) for name in (
            "mmdet3d.core", "mmdet3d.core.bbox",
            "mmdet3d.core.bbox.structures",
            "mmdet3d.core.bbox.structures.utils",
        )
    }
    modules["mmdet3d.core.bbox.structures.utils"].points_cam2img = (
        lambda *_args, **_kwargs: None
    )
    for name, module in modules.items():
        sys.modules.setdefault(name, module)
    spec = importlib.util.spec_from_file_location(
        "pperf_3d_corruptions_camera", root / "Camera_corruptions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_lisa_initializer(root, kind):
    global _LEGACY_LISA, _LEGACY_LISA_MODULE, _LEGACY_LISA_ALPHA
    import numpy as np
    import scipy.integrate
    scipy.integrate.trapz = getattr(scipy.integrate, "trapz", np.trapz)
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    from utils import lisa
    _LEGACY_LISA_MODULE = lisa
    _LEGACY_LISA = lisa.LISA(mode="gunn", show_progressbar=False)
    _LEGACY_LISA_ALPHA = {}


def _legacy_lisa_point(arguments):
    point, rate, seed = arguments
    import numpy as np
    np.random.seed(seed)
    simulator = _LEGACY_LISA
    if rate not in _LEGACY_LISA_ALPHA:
        _LEGACY_LISA_ALPHA[rate] = simulator.alpha(
            simulator.Nd(simulator.D, rate)
        )
    return _LEGACY_LISA_MODULE.monte_carlo_lisa(
        x=point[0], y=point[1], z=point[2], i=point[3], Rr=rate,
        fixed_seed=False, r_min=simulator.r_min, r_max=simulator.r_max,
        signal=simulator.signal, beam_divergence=simulator.beam_divergence,
        min_diameter=simulator.min_diameter,
        range_accuracy=simulator.range_accuracy,
        refractive_index=simulator.refractive_index,
        alpha=_LEGACY_LISA_ALPHA[rate], density=simulator.density,
        diameters=simulator.diameters,
    )


def _ensure_lisa_cache(root, kind):
    """Persist the deterministic ice Mie table omitted upstream."""
    if kind != "snow":
        return None
    path = Path(root) / "utils" / "mie_1.3031_λ_905.npz"
    if not path.is_file():
        import numpy as np
        _legacy_lisa_initializer(root, kind)
        np.savez(
            path, D=_LEGACY_LISA.D, qext=_LEGACY_LISA.qext,
            qback=_LEGACY_LISA.qback,
        )
    return path


def _pylisa_initializer(root, rate):
    global _PYLISA
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import pylisa.lisa as lisa_module
    simulator = lisa_module.Lisa(atm_model="rain", mode="strongest")
    coefficients = lisa_module.alpha_beta(
        simulator.N_model(simulator.D, rate), simulator.D,
        simulator.q_ext, simulator.q_back,
    )
    lisa_module.alpha_beta = lambda *_args: coefficients
    _PYLISA = simulator


def _pylisa_point(arguments):
    point, rate, seed = arguments
    import numpy as np
    np.random.seed(seed)
    if point[3] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return _PYLISA.lisa_mc(*point, rate)


def _header_timestamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _decode_camera(message):
    import cv2
    import numpy as np
    image = cv2.imdecode(
        np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise CorruptionError("CAM_FRONT JPEG could not be decoded")
    if image.shape[:2] != (ASTRA_CAMERA["height"], ASTRA_CAMERA["width"]):
        raise CorruptionError(
            f"unexpected CAM_FRONT dimensions: {image.shape}"
        )
    return image


def _corrupt_camera(message, corruptor):
    import cv2
    image = _decode_camera(message)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(corruptor(rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", output)
    if not ok:
        raise CorruptionError("corrupted CAM_FRONT JPEG could not be encoded")
    message.data = array("B", encoded.tobytes())
    return message


def _encode_astra_camera(message, rendered_path):
    import cv2
    image = cv2.imread(str(rendered_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (
            ASTRA_CAMERA["height"], ASTRA_CAMERA["width"]):
        raise CorruptionError(
            f"invalid Astra renderer output: {rendered_path}"
        )
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise CorruptionError(
            f"could not encode Astra output: {rendered_path}"
        )
    message.data = array("B", encoded.tobytes())
    return message


def _point_dtype(message):
    import numpy as np
    formats = {1: "i1", 2: "u1", 3: "i2", 4: "u2",
               5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    names, values, offsets = [], [], []
    endian = ">" if message.is_bigendian else "<"
    for field in message.fields:
        if field.count != 1 or field.datatype not in formats:
            raise CorruptionError(
                f"unsupported PointCloud2 field {field.name}"
            )
        names.append(field.name)
        values.append(endian + formats[field.datatype])
        offsets.append(field.offset)
    return np.dtype({"names": names, "formats": values, "offsets": offsets,
                     "itemsize": message.point_step})


def _point_records(message):
    import numpy as np
    return np.frombuffer(
        bytes(message.data), dtype=_point_dtype(message),
        count=int(message.width) * int(message.height),
    ).copy()


def _corrupt_lidar_snow(message, pool, severity, seed):
    import numpy as np
    records = _point_records(message)
    required = ("x", "y", "z", "intensity")
    if not set(required).issubset(records.dtype.names):
        raise CorruptionError("LIDAR_TOP lacks x/y/z/intensity fields")
    points = np.column_stack([records[name] for name in required])
    rate = RAIN_RATES[severity - 1]
    tasks = [
        (point, rate, derived_seed(seed, LIDAR_TOPIC, 0, index))
        for index, point in enumerate(points)
    ]
    transformed = np.asarray(pool.map(
        _legacy_lisa_point, tasks, chunksize=256
    ))
    keep = (transformed[:, 4] != 0) & np.isfinite(transformed[:, :4]).all(1)
    output = records[keep].copy()
    for index, name in enumerate(required):
        output[name] = transformed[keep, index]
    message.height = 1
    message.width = len(output)
    message.row_step = message.point_step * message.width
    message.is_dense = bool(keep.all())
    message.data = array("B", output.tobytes())
    return message, len(records), len(output)


def _corrupt_lidar_rain(message, pool, rate, seed, header_timestamp_ns):
    import numpy as np
    records = _point_records(message)
    required = ("x", "y", "z", "intensity")
    if not set(required).issubset(records.dtype.names):
        raise CorruptionError("LIDAR_TOP lacks x/y/z/intensity fields")
    points = np.column_stack([records[name] for name in required]).astype(
        np.float64
    )
    if not np.isfinite(points).all():
        raise CorruptionError("clean LIDAR_TOP contains a nonfinite point")
    if np.any(points[:, 3] < 0) or np.any(points[:, 3] > 255):
        raise CorruptionError("nuScenes intensity lies outside [0,255]")
    points[:, 3] /= 255.0
    message_seed = derived_seed(seed, LIDAR_TOPIC, header_timestamp_ns)
    point_seeds = [
        derived_seed(message_seed, LIDAR_TOPIC, header_timestamp_ns, index)
        for index in range(len(points))
    ]
    tasks = list(zip(points, [rate] * len(points), point_seeds))
    transformed = np.asarray(pool.map(
        _pylisa_point, tasks, chunksize=256
    ), dtype=np.float64)
    if transformed.shape != (len(points), 5):
        raise CorruptionError("official LISA returned an invalid point array")
    keep = transformed[:, 4] != 0
    if not np.isfinite(transformed[keep, :4]).all():
        raise CorruptionError("official LISA returned a nonfinite point")
    if np.any(transformed[keep, 3] < 0) or np.any(transformed[keep, 3] > 1):
        raise CorruptionError("official LISA returned invalid reflectivity")
    output = records[keep].copy()
    for index, name in enumerate(required[:3]):
        output[name] = transformed[keep, index]
    output["intensity"] = transformed[keep, 3] * 255.0
    message.height = 1
    message.width = len(output)
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = array("B", output.tobytes())
    return message, {
        "source_point_count": len(records),
        "output_point_count": len(output),
        "lost_point_count": int(np.count_nonzero(transformed[:, 4] == 0)),
        "scattered_point_count": int(np.count_nonzero(transformed[:, 4] == 1)),
        "surviving_point_count": int(np.count_nonzero(transformed[:, 4] == 2)),
        "point_seed_first": point_seeds[0] if point_seeds else None,
        "point_seed_last": point_seeds[-1] if point_seeds else None,
        "output_points_finite": True,
    }


def _stream_digest_update(digest, topic, timestamp_ns, payload):
    digest.update(topic.encode())
    digest.update(b"\0")
    digest.update(int(timestamp_ns).to_bytes(8, "big", signed=False))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _quaternion_matrix(rotation):
    import numpy as np
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise CorruptionError("invalid nuScenes quaternion")
    norm = np.linalg.norm(value)
    if norm == 0:
        raise CorruptionError("zero nuScenes quaternion")
    w, x, y, z = value / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ])


def _forward_transform(points, record):
    import numpy as np
    return points @ _quaternion_matrix(record["rotation"]).T + np.asarray(
        record["translation"], dtype=np.float64
    )


def _inverse_transform(points, record):
    import numpy as np
    return (points - np.asarray(
        record["translation"], dtype=np.float64
    )) @ _quaternion_matrix(record["rotation"])


def lidar_to_camera(points, lidar_sample_data, camera_sample_data,
                    calibrated_sensors, ego_poses):
    """Motion-compensate LiDAR points into one camera coordinate frame."""
    points = _forward_transform(
        points,
        calibrated_sensors[lidar_sample_data["calibrated_sensor_token"]],
    )
    points = _forward_transform(
        points, ego_poses[lidar_sample_data["ego_pose_token"]]
    )
    points = _inverse_transform(
        points, ego_poses[camera_sample_data["ego_pose_token"]]
    )
    return _inverse_transform(
        points,
        calibrated_sensors[camera_sample_data["calibrated_sensor_token"]],
    )


def project_depth(points_camera, intrinsic, width, height, max_depth=100.0):
    """Project camera-frame points with a nearest-positive-depth z-buffer."""
    import numpy as np
    points = np.asarray(points_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or intrinsic.shape != (3, 3):
        raise CorruptionError("invalid point or intrinsic matrix shape")
    z = points[:, 2]
    projected = points @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.floor(projected[:, 0] / z + 0.5).astype(np.int64)
        y = np.floor(projected[:, 1] / z + 0.5).astype(np.int64)
    valid = (
        np.isfinite(points).all(axis=1) & (z > 0) & (z < max_depth)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, y[valid] * width + x[valid], z[valid])
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(height, width)


def complete_depth_ip_basic(sparse_depth, max_depth=100.0):
    """Densify metric depth with the deterministic IP-Basic pipeline."""
    import cv2
    import numpy as np
    depth = np.asarray(sparse_depth, dtype=np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if not valid.any():
        raise CorruptionError("sparse camera depth has no valid pixels")
    depth[~valid] = 0
    depth[valid] = max_depth - depth[valid]
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    depth = cv2.dilate(depth, kernel5)
    depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel5)
    empty = depth < 0.1
    dilated = cv2.dilate(depth, np.ones((7, 7), dtype=np.uint8))
    depth[empty] = dilated[empty]

    top = np.argmax(depth > 0.1, axis=0)
    top_values = depth[top, np.arange(depth.shape[1])]
    rows = np.arange(depth.shape[0])[:, None]
    depth = np.where(rows < top[None, :], top_values[None, :], depth)

    kernel31 = np.ones((31, 31), dtype=np.uint8)
    while np.any(depth < 0.1):
        empty = depth < 0.1
        dilated = cv2.dilate(depth, kernel31)
        if not np.any(dilated[empty] >= 0.1):
            raise CorruptionError("31x31 depth fill made no progress")
        depth[empty] = dilated[empty]
    depth = cv2.medianBlur(depth, 5)
    depth = cv2.bilateralFilter(depth, 5, 1.5, 2.0)
    depth = max_depth - depth
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise CorruptionError(
            "dense camera depth is not fully finite/positive"
        )
    return depth.astype(np.float32, copy=False)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_nuscenes_metadata(root, scene_token, scene_name=None):
    root = Path(root).resolve()
    required = (
        "scene.json", "sample.json", "sample_data.json", "sensor.json",
        "calibrated_sensor.json", "ego_pose.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise CorruptionError(f"nuScenes metadata files missing: {missing}")
    scenes = {item["token"]: item for item in _load_json(root / "scene.json")}
    if scene_token not in scenes:
        raise CorruptionError(f"nuScenes scene token not found: {scene_token}")
    scene = scenes[scene_token]
    if scene_name is not None and scene.get("name") != scene_name:
        raise CorruptionError(
            f"scene name/token mismatch: {scene_name} != {scene.get('name')}"
        )
    samples = {
        item["token"]: item for item in _load_json(root / "sample.json")
    }
    sample_tokens = set()
    token = scene["first_sample_token"]
    while token:
        if token in sample_tokens or token not in samples:
            raise CorruptionError("invalid nuScenes scene sample chain")
        sample_tokens.add(token)
        token = samples[token]["next"]
    if len(sample_tokens) != scene["nbr_samples"]:
        raise CorruptionError("nuScenes scene sample count differs")

    sensors = {
        item["token"]: item for item in _load_json(root / "sensor.json")
    }
    calibrated = {
        item["token"]: item
        for item in _load_json(root / "calibrated_sensor.json")
    }
    ego_poses = {
        item["token"]: item for item in _load_json(root / "ego_pose.json")
    }
    by_channel = {"CAM_FRONT": {}, "LIDAR_TOP": {}}
    for item in _load_json(root / "sample_data.json"):
        if item["sample_token"] not in sample_tokens:
            continue
        calibration = calibrated[item["calibrated_sensor_token"]]
        channel = sensors[calibration["sensor_token"]]["channel"]
        if channel not in by_channel:
            continue
        timestamp_ns = int(item["timestamp"]) * 1000
        if timestamp_ns in by_channel[channel]:
            raise CorruptionError(
                f"duplicate {channel} metadata timestamp {timestamp_ns}"
            )
        if item["ego_pose_token"] not in ego_poses:
            raise CorruptionError(
                f"missing ego pose for sample_data {item['token']}"
            )
        by_channel[channel][timestamp_ns] = item
    return {
        "root": root,
        "channels": by_channel,
        "calibrated_sensors": calibrated,
        "ego_poses": ego_poses,
    }


def _read_full_bag(source):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = reader.get_all_topics_and_types()
    messages, sensor_messages = [], {}
    source_digest, preserved = hashlib.sha256(), hashlib.sha256()
    first_ns = last_ns = None
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        timestamp_ns = int(timestamp_ns)
        first_ns = timestamp_ns if first_ns is None else first_ns
        last_ns = timestamp_ns
        messages.append((topic, raw, timestamp_ns))
        _stream_digest_update(source_digest, topic, timestamp_ns, raw)
        if topic == CAMERA_TOPIC:
            message = deserialize_message(raw, CompressedImage)
            key = (topic, timestamp_ns)
            if key in sensor_messages:
                raise CorruptionError(
                    f"duplicate selected bag timestamp: {key}"
                )
            sensor_messages[key] = message
        elif topic == LIDAR_TOPIC:
            message = deserialize_message(raw, PointCloud2)
            key = (topic, timestamp_ns)
            if key in sensor_messages:
                raise CorruptionError(
                    f"duplicate selected bag timestamp: {key}"
                )
            sensor_messages[key] = message
        else:
            _stream_digest_update(preserved, topic, timestamp_ns, raw)
    if first_ns is None:
        raise CorruptionError("source MCAP is empty")
    return {
        "messages": messages,
        "sensor_messages": sensor_messages,
        "topics": topics,
        "first_ns": first_ns,
        "last_ns": last_ns,
        "source_digest": source_digest.hexdigest(),
        "preserved_digest": preserved.hexdigest(),
    }


def _astra_provenance(astra_root):
    astra_root = Path(astra_root).resolve()
    simulator_root = astra_root / "3rdparty/weather-particle-simulator"
    marker = astra_root / "3rdparty/rainstreakdb.archive.sha256"
    astra_commit = _git_output(astra_root, "rev-parse", "HEAD")
    simulator_commit = _git_output(simulator_root, "rev-parse", "HEAD")
    archive_hash = marker.read_text(encoding="utf-8").strip()
    if astra_commit != ASTRA_COMMIT:
        raise CorruptionError(f"unexpected Astra commit: {astra_commit}")
    if simulator_commit != WEATHER_SIMULATOR_COMMIT:
        raise CorruptionError(
            f"unexpected weather simulator commit: {simulator_commit}"
        )
    if archive_hash != RAIN_STREAK_ARCHIVE_SHA256:
        raise CorruptionError(f"unexpected rain-streak hash: {archive_hash}")
    return {
        "astra_repository": "https://github.com/astra-vision/rain-rendering",
        "astra_commit": astra_commit,
        "astra_dirty": bool(_git_output(astra_root, "status", "--porcelain")),
        "weather_simulator_repository": (
            "https://github.com/astra-vision/weather-particle-simulator"
        ),
        "weather_simulator_commit": simulator_commit,
        "weather_simulator_dirty": bool(_git_output(
            simulator_root, "status", "--porcelain"
        )),
        "rain_streak_archive_sha256": archive_hash,
        "compatibility_patches": [
            "fail-fast renderer/simulator exceptions",
            "caller-supplied simulator and per-frame seeds",
        ],
    }


def _run_adapter(stage, spec_path, astra_root, legacy=False, log_path=None):
    command = [
        sys.executable, "-m", "closeloop_experiments.astra_adapter",
        stage, str(spec_path),
    ]
    environment = None
    if legacy:
        import os
        environment = os.environ.copy()
        prefix = "/opt/astra-legacy/lib:/opt/astra-legacy/lib64"
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{prefix}:{current}" if current else prefix
        )
    if log_path is None:
        result = subprocess.run(
            command, cwd=astra_root, env=environment,
            check=False, capture_output=True, text=True,
        )
        if result.returncode:
            raise CorruptionError(
                f"Astra {stage} failed:\n{result.stdout}\n{result.stderr}"
            )
        return None
    log = Path(log_path).open("w", encoding="utf-8")
    return subprocess.Popen(
        command, cwd=astra_root, stdout=log, stderr=subprocess.STDOUT,
        text=True,
    ), log


def _run_astra(workspace, astra_root, astra_cache, rate, duration_seconds,
               frame_seeds, workers, seed):
    astra_root = Path(astra_root).resolve()
    cache = Path(astra_cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    rate = float(rate)
    settings = {
        "cam_hz": ASTRA_CAMERA["nominal_hz"],
        "cam_CCD_WH": [ASTRA_CAMERA["width"], ASTRA_CAMERA["height"]],
        "cam_CCD_pixsize": ASTRA_CAMERA["pixel_pitch_um"],
        "cam_WH": [ASTRA_CAMERA["width"], ASTRA_CAMERA["height"]],
        "cam_focal": ASTRA_CAMERA["focal_length_mm"],
        "cam_gain": ASTRA_CAMERA["gain"],
        "cam_f_number": ASTRA_CAMERA["f_number"],
        "cam_focus_plane": 6.0,
        "cam_exposure": ASTRA_CAMERA["exposure_ms"],
        "cam_pos": [1.5, 1.5, 0.3],
        "cam_lookat": [1.5, 1.5, -1.0],
        "cam_up": [0.0, 1.0, 0.0],
        "depth_scale": 1,
        "render_scale": 1,
        "sim_hz": 2000,
        "sim_mode": "normal",
        "sim_duration": duration_seconds,
        "sim_steps": {},
        "seed": seed,
    }
    particle_spec = workspace / "particles.json"
    particle_spec.write_text(json.dumps({
        "astra_root": str(astra_root), "cache": str(cache),
        "settings": settings, "rain_rate": rate,
    }), encoding="utf-8")
    rate_dir = cache / "rain" / f"{rate}mm"
    existed = rate_dir.exists()
    _run_adapter("particles", particle_spec, astra_root, legacy=True)
    options_path = rate_dir / "sim_options.json"
    if not options_path.is_file():
        raise CorruptionError(
            f"Astra simulator options missing: {options_path}"
        )
    actual_options = _load_json(options_path)
    if any(
            actual_options.get(key) != value
            for key, value in settings.items()):
        raise CorruptionError("Astra particle cache settings differ")
    particle_files = sorted(rate_dir.glob("*_camera0.xml"))
    if len(particle_files) != 1:
        raise CorruptionError(
            f"expected one Astra particle XML, found {len(particle_files)}"
        )
    particle_file = particle_files[0]
    particle_frames = len(ElementTree.parse(particle_file).getroot())
    if particle_frames < len(frame_seeds):
        raise CorruptionError(
            f"Astra produced {particle_frames} particle frames for "
            f"{len(frame_seeds)} images"
        )
    if not existed:
        (rate_dir / "automate_log.txt").unlink(missing_ok=True)

    sequence = ASTRA_SEQUENCE
    output_root = workspace / "output"
    base_spec = {
        "astra_root": str(astra_root),
        "conflict_strategy": "overwrite",
        "rendering_strategy": None,
        "output": str(output_root),
        "dataset": "customdb",
        "dataset_root": str(workspace / "source"),
        "images": {sequence: str(workspace / "source" / sequence / "rgb")},
        "sequences": [sequence],
        "depth": {sequence: str(workspace / "source" / sequence / "depth")},
        "particles": {sequence: [str(particle_file)]},
        "weather": [{"weather": "rain", "fallrate": rate}],
        "texture": str(
            astra_root / "3rdparty/rainstreakdb/env_light_database/size32"
        ),
        "norm_coeff": str(
            astra_root
            / "3rdparty/rainstreakdb/env_light_database/txt"
            / "normalized_env_max.txt"
        ),
        "save_envmap": False,
        "settings": settings,
        "calib": {sequence: None},
        "noise_scale": 0.0,
        "noise_std": 0.0,
        "opacity_attenuation": 1.0,
        "frame_start": 0,
        "frame_end": None,
        "frame_step": 1,
        "frame_seeds": frame_seeds,
        "verbose": False,
    }
    processes = []
    worker_count = max(1, min(int(workers), len(frame_seeds)))
    for worker in range(worker_count):
        frames = list(range(worker, len(frame_seeds), worker_count))
        spec = dict(base_spec, frames=frames)
        spec_path = workspace / f"renderer-{worker:02d}.json"
        log_path = workspace / f"renderer-{worker:02d}.log"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        process, log = _run_adapter(
            "renderer", spec_path, astra_root, log_path=log_path
        )
        processes.append((process, log, log_path))
    failures = []
    for process, log, log_path in processes:
        returncode = process.wait()
        log.close()
        if returncode:
            failures.append(log_path.read_text(encoding="utf-8")[-8000:])
    if failures:
        raise CorruptionError("Astra renderer failed:\n" + "\n".join(failures))

    rendered_dir = (
        output_root / "customdb" / sequence / "rain" / f"{rate}mm"
        / "rainy_image"
    )
    expected = [rendered_dir / f"frame_{index:06d}.png"
                for index in range(len(frame_seeds))]
    if any(not path.is_file() for path in expected):
        missing = [str(path) for path in expected if not path.is_file()]
        raise CorruptionError(f"Astra renderer outputs missing: {missing[:3]}")
    if len(list(rendered_dir.glob("*.png"))) != len(expected):
        raise CorruptionError(
            "Astra renderer produced an unexpected frame count"
        )
    return expected, {
        "particle_cache": str(cache),
        "particle_file_sha256": _sha256_file(particle_file),
        "particle_frame_count": particle_frames,
        "rendered_frame_count": len(expected),
    }


def _prepare_astra_inputs(workspace, camera_entries, lidar_by_header,
                          camera_pairs, metadata, seed):
    import cv2
    import numpy as np
    rgb_dir = workspace / "source" / ASTRA_SEQUENCE / "rgb"
    depth_dir = workspace / "source" / ASTRA_SEQUENCE / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    calibrated = metadata["calibrated_sensors"]
    ego_poses = metadata["ego_poses"]
    camera_metadata = metadata["channels"]["CAM_FRONT"]
    lidar_metadata = metadata["channels"]["LIDAR_TOP"]
    pair_by_camera = {
        camera: (lidar, delta) for camera, lidar, delta in camera_pairs
    }
    frame_seeds, stats = [], {}
    intrinsic = None
    for index, (header_ns, message) in enumerate(sorted(camera_entries)):
        lidar_header_ns, delta_ns = pair_by_camera[header_ns]
        camera_sd = camera_metadata[header_ns]
        lidar_sd = lidar_metadata[lidar_header_ns]
        camera_calibration = calibrated[
            camera_sd["calibrated_sensor_token"]
        ]
        current_intrinsic = np.asarray(
            camera_calibration["camera_intrinsic"], dtype=np.float64
        )
        if intrinsic is None:
            intrinsic = current_intrinsic
        elif not np.array_equal(intrinsic, current_intrinsic):
            raise CorruptionError("CAM_FRONT intrinsic changed within scene")
        lidar_records = _point_records(lidar_by_header[lidar_header_ns])
        if not {"x", "y", "z"}.issubset(lidar_records.dtype.names):
            raise CorruptionError("LIDAR_TOP lacks x/y/z fields")
        points = np.column_stack([
            lidar_records["x"], lidar_records["y"], lidar_records["z"]
        ]).astype(np.float64)
        points_camera = lidar_to_camera(
            points, lidar_sd, camera_sd, calibrated, ego_poses
        )
        sparse = project_depth(
            points_camera, current_intrinsic,
            ASTRA_CAMERA["width"], ASTRA_CAMERA["height"],
            DEPTH_COMPLETION["max_depth_m"],
        )
        dense = complete_depth_ip_basic(
            sparse, DEPTH_COMPLETION["max_depth_m"]
        )
        image = _decode_camera(message)
        image_path = rgb_dir / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(image_path), image):
            raise CorruptionError(f"could not write Astra image {image_path}")
        np.save(depth_dir / f"frame_{index:06d}.npy", dense)
        message_seed = derived_seed(seed, CAMERA_TOPIC, header_ns)
        frame_seeds.append(message_seed)
        sparse_count = int(np.count_nonzero(sparse))
        dense_count = int(np.count_nonzero(dense))
        stats[header_ns] = {
            "paired_lidar_header_timestamp_ns": lidar_header_ns,
            "lidar_pairing_delta_ns": delta_ns,
            "sparse_depth_valid_pixels": sparse_count,
            "sparse_depth_coverage": sparse_count / sparse.size,
            "dense_depth_valid_pixels": dense_count,
            "dense_depth_coverage": dense_count / dense.size,
            "depth_payload_sha256": _sha256_bytes(dense.tobytes()),
        }
    ASTRA_CAMERA["pixel_pitch_um"] = (
        1000.0 * ASTRA_CAMERA["focal_length_mm"] / intrinsic[0, 0]
    )
    ASTRA_CAMERA["intrinsic"] = intrinsic.tolist()
    return frame_seeds, stats


def _write_full_rain_bag(source, output, bag, rendered, camera_stats,
                         metadata, scene_name, scene_token,
                         rate, seed, lisa_root, workers, astra_provenance,
                         astra_stats):
    import rosbag2_py
    from rclpy.serialization import serialize_message

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.stage-", dir=output.parent
    ))
    working = stage / output.name
    writer = None
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(
        workers, initializer=_pylisa_initializer,
        initargs=(str(Path(lisa_root).resolve()), float(rate)),
    )
    records = []
    output_digest = hashlib.sha256()
    camera_messages = sorted(
        (_header_timestamp_ns(message), message)
        for (topic, _timestamp), message in bag["sensor_messages"].items()
        if topic == CAMERA_TOPIC
    )
    camera_rendered = {
        header: path for (header, _message), path in zip(
            camera_messages, rendered
        )
    }
    try:
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(working), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        for topic_metadata in bag["topics"]:
            writer.create_topic(topic_metadata)
        for topic, raw, timestamp_ns in bag["messages"]:
            output_raw = raw
            if topic in (CAMERA_TOPIC, LIDAR_TOPIC):
                message = bag["sensor_messages"][(topic, timestamp_ns)]
                header_ns = _header_timestamp_ns(message)
                message_seed = derived_seed(seed, topic, header_ns)
                source_count = output_count = None
                extra = {}
                channel = "CAM_FRONT" if topic == CAMERA_TOPIC else "LIDAR_TOP"
                sample_data = metadata["channels"][channel][header_ns]
                if topic == CAMERA_TOPIC:
                    rendered_path = camera_rendered[header_ns]
                    message = _encode_astra_camera(message, rendered_path)
                    extra.update(camera_stats[header_ns])
                    extra["renderer_payload_sha256"] = _sha256_file(
                        rendered_path
                    )
                else:
                    message, lidar_stats = _corrupt_lidar_rain(
                        message, pool, float(rate), seed, header_ns
                    )
                    source_count = lidar_stats.pop("source_point_count")
                    output_count = lidar_stats.pop("output_point_count")
                    extra.update(lidar_stats)
                output_raw = serialize_message(message)
                if _sha256_bytes(raw) == _sha256_bytes(output_raw):
                    raise CorruptionError(
                        "rain left selected payload unchanged: "
                        f"{topic} {header_ns}"
                    )
                records.append({
                    "topic": topic,
                    "bag_timestamp_ns": timestamp_ns,
                    "output_bag_timestamp_ns": timestamp_ns,
                    "header_timestamp_ns": header_ns,
                    "output_header_timestamp_ns": _header_timestamp_ns(
                        message
                    ),
                    "message_seed": message_seed,
                    "source_payload_sha256": _sha256_bytes(raw),
                    "output_payload_sha256": _sha256_bytes(output_raw),
                    "source_point_count": source_count,
                    "output_point_count": output_count,
                    "metadata_sample_data_token": sample_data["token"],
                    "metadata_ego_pose_token": sample_data["ego_pose_token"],
                    "metadata_calibrated_sensor_token": sample_data[
                        "calibrated_sensor_token"
                    ],
                    **extra,
                })
            _stream_digest_update(
                output_digest, topic, timestamp_ns, output_raw
            )
            writer.write(topic, output_raw, timestamp_ns)
        del writer
        writer = None
        pool.close()
        pool.join()
        pool = None

        duration_ns = bag["last_ns"] - bag["first_ns"]
        camera_records = [r for r in records if r["topic"] == CAMERA_TOPIC]
        lidar_records = [r for r in records if r["topic"] == LIDAR_TOPIC]
        all_deltas = [r["lidar_pairing_delta_ns"] for r in camera_records]
        depth_lidars = {
            r["paired_lidar_header_timestamp_ns"] for r in camera_records
        }
        manifest = {
            "schema_version": 1,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "duration_ns": duration_ns,
            "duration_seconds": duration_ns / 1_000_000_000,
            "source_message_count": len(bag["messages"]),
            "window_start_bag_timestamp_ns": bag["first_ns"],
            "window_end_bag_timestamp_ns": bag["last_ns"],
            "source_bag": str(source),
            "source_bag_sha256": _sha256_file(source),
            "output_bag_directory": str(output),
            "output_bag_files": [
                {
                    "path": str(path.relative_to(working)),
                    "sha256": _sha256_file(path),
                }
                for path in sorted(working.rglob("*.mcap"))
            ],
            "corruption": {
                "type": "rain", "severity": None,
                "rain_rate_mm_per_hour": float(rate),
                "rate_path_component": rain_rate_slug(rate),
                "seed": seed, "full_duration": True,
            },
            "generator": {
                **astra_provenance,
                "camera_implementation": "Astra physics-based rain renderer",
                "camera_settings": ASTRA_CAMERA,
                "depth_completion": DEPTH_COMPLETION,
                "lidar_repository": "repository LISA/pylisa",
                "lidar_implementation": (
                    "pylisa.Lisa(atm_model=rain, mode=strongest)"
                ),
                "lisa_defaults": {
                    "refractive_index": 1.328,
                    "wavelength_nm": 905,
                    "maximum_range_m": 200,
                    "minimum_range_m": 1.5,
                    "beam_divergence_rad": 0.003,
                    "minimum_drop_diameter_mm": 0.05,
                    "range_accuracy_m": 0.09,
                    "return_mode": "strongest",
                    "input_intensity_range": [0, 255],
                    "simulation_intensity_range": [0, 1],
                },
                "randomization": (
                    "sha256-derived per-message/per-point seeds; identical "
                    "streams at every rate"
                ),
                **astra_stats,
            },
            "source_stream_sha256": bag["source_digest"],
            "output_stream_sha256": output_digest.hexdigest(),
            "preserved_stream_source_sha256": bag["preserved_digest"],
            "preserved_stream_output_sha256": bag["preserved_digest"],
            "pairing": {
                "basis": "sensor header timestamp",
                "tolerance_ns": MAX_PAIR_DELTA_NS,
                "camera_messages": len(camera_records),
                "lidar_messages": len(lidar_records),
                "paired_camera_messages": len(camera_records),
                "depth_pair_count": len(camera_records),
                "unique_depth_source_lidar_messages": len(depth_lidars),
                "maximum_delta_ns": max(all_deltas),
            },
            "sensor_records": records,
        }
        errors = validate_manifest(manifest)
        if errors:
            raise CorruptionError("; ".join(errors))
        (working / "input_variation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        working.rename(output)
        return output / "input_variation_manifest.json"
    finally:
        if writer is not None:
            del writer
        if pool is not None:
            pool.terminate()
            pool.join()
        shutil.rmtree(stage, ignore_errors=True)


def _transform_full_rain(source, output, scene_name, scene_token, rate, seed,
                         astra_root, lisa_root, metadata_root, astra_cache,
                         workers):
    source, output = Path(source).resolve(), Path(output).resolve()
    astra_cache = resolve_astra_cache(scene_name, astra_cache)
    if output.exists():
        raise CorruptionError(f"immutable output already exists: {output}")
    if not source.is_file():
        raise CorruptionError(f"source MCAP file not found: {source}")
    rain_rate_slug(rate)
    provenance = _astra_provenance(astra_root)
    if not (Path(lisa_root) / "pylisa/lisa.py").is_file():
        raise CorruptionError(f"official LISA root is invalid: {lisa_root}")
    bag = _read_full_bag(source)
    duration_ns = bag["last_ns"] - bag["first_ns"]
    if duration_ns <= 0:
        raise CorruptionError(
            f"source bag duration must be positive: {duration_ns} ns"
        )
    camera_entries, lidar_by_header = [], {}
    for (topic, _timestamp), message in bag["sensor_messages"].items():
        header_ns = _header_timestamp_ns(message)
        if topic == CAMERA_TOPIC:
            camera_entries.append((header_ns, message))
        else:
            if header_ns in lidar_by_header:
                raise CorruptionError(
                    f"duplicate LIDAR_TOP header timestamp {header_ns}"
                )
            lidar_by_header[header_ns] = message
    camera_headers = [header for header, _message in camera_entries]
    lidar_headers = list(lidar_by_header)
    if not camera_headers or not lidar_headers:
        raise CorruptionError("source bag lacks CAM_FRONT or LIDAR_TOP data")
    metadata = _load_nuscenes_metadata(
        metadata_root, scene_token, scene_name
    )
    if sorted(camera_headers) != sorted(
            metadata["channels"]["CAM_FRONT"]):
        raise CorruptionError(
            "CAM_FRONT headers do not exactly match metadata"
        )
    if sorted(lidar_headers) != sorted(
            metadata["channels"]["LIDAR_TOP"]):
        raise CorruptionError(
            "LIDAR_TOP headers do not exactly match metadata"
        )
    camera_pairs = pair_camera_lidar_timestamps(
        camera_headers, lidar_headers, MAX_PAIR_DELTA_NS
    )
    with tempfile.TemporaryDirectory(prefix="pperf-astra-") as workspace_name:
        workspace = Path(workspace_name)
        frame_seeds, camera_stats = _prepare_astra_inputs(
            workspace, camera_entries, lidar_by_header, camera_pairs,
            metadata, seed,
        )
        rendered, astra_stats = _run_astra(
            workspace, astra_root, astra_cache, rate,
            duration_ns / 1_000_000_000, frame_seeds, workers, seed,
        )
        return _write_full_rain_bag(
            source, output, bag, rendered, camera_stats, metadata,
            scene_name, scene_token, rate, seed, lisa_root,
            workers, provenance, astra_stats,
        )


def _transform_full_clean(source, output, scene_name, scene_token, seed):
    """Copy one complete MCAP byte-for-byte with an integrity manifest."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise CorruptionError(f"immutable output already exists: {output}")
    if not source.is_file():
        raise CorruptionError(f"source MCAP file not found: {source}")
    bag = _read_full_bag(source)
    camera_headers = sorted(
        _header_timestamp_ns(message)
        for (topic, _timestamp), message in bag["sensor_messages"].items()
        if topic == CAMERA_TOPIC
    )
    lidar_headers = sorted(
        _header_timestamp_ns(message)
        for (topic, _timestamp), message in bag["sensor_messages"].items()
        if topic == LIDAR_TOPIC
    )
    camera_pairs = pair_camera_lidar_timestamps(
        camera_headers, lidar_headers, MAX_PAIR_DELTA_NS
    )
    pair_by_camera = {
        camera: (lidar, delta) for camera, lidar, delta in camera_pairs
    }
    records = []
    for topic, raw, timestamp_ns in bag["messages"]:
        if topic not in (CAMERA_TOPIC, LIDAR_TOPIC):
            continue
        message = bag["sensor_messages"][(topic, timestamp_ns)]
        header_ns = _header_timestamp_ns(message)
        record = {
            "topic": topic,
            "bag_timestamp_ns": timestamp_ns,
            "output_bag_timestamp_ns": timestamp_ns,
            "header_timestamp_ns": header_ns,
            "output_header_timestamp_ns": header_ns,
            "message_seed": derived_seed(seed, topic, header_ns),
            "source_payload_sha256": _sha256_bytes(raw),
            "output_payload_sha256": _sha256_bytes(raw),
        }
        if topic == CAMERA_TOPIC:
            lidar_header, delta = pair_by_camera[header_ns]
            record.update({
                "paired_lidar_header_timestamp_ns": lidar_header,
                "lidar_pairing_delta_ns": delta,
            })
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.stage-", dir=output.parent
    ))
    working = stage / output.name
    try:
        working.mkdir()
        output_mcap = working / source.name
        shutil.copyfile(source, output_mcap)
        duration_ns = bag["last_ns"] - bag["first_ns"]
        manifest = {
            "schema_version": 1,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "duration_ns": duration_ns,
            "duration_seconds": duration_ns / 1_000_000_000,
            "source_message_count": len(bag["messages"]),
            "output_message_count": len(bag["messages"]),
            "window_start_bag_timestamp_ns": bag["first_ns"],
            "window_end_bag_timestamp_ns": bag["last_ns"],
            "source_bag": str(source),
            "source_bag_sha256": _sha256_file(source),
            "output_bag_directory": str(output),
            "output_bag_files": [{
                "path": output_mcap.name,
                "sha256": _sha256_file(output_mcap),
            }],
            "corruption": {
                "type": "clean", "severity": 0, "seed": seed,
                "full_duration": True,
            },
            "generator": {
                "camera_implementation": "identity",
                "lidar_implementation": "identity",
                "randomization": "none",
            },
            "source_stream_sha256": bag["source_digest"],
            "output_stream_sha256": bag["source_digest"],
            "preserved_stream_source_sha256": bag["preserved_digest"],
            "preserved_stream_output_sha256": bag["preserved_digest"],
            "pairing": {
                "basis": "sensor header timestamp",
                "tolerance_ns": MAX_PAIR_DELTA_NS,
                "camera_messages": len(camera_headers),
                "lidar_messages": len(lidar_headers),
                "paired_camera_messages": len(camera_pairs),
                "depth_pair_count": len(camera_pairs),
                "unique_depth_source_lidar_messages": len({
                    lidar for _camera, lidar, _delta in camera_pairs
                }),
                "maximum_delta_ns": max(
                    delta for _camera, _lidar, delta in camera_pairs
                ),
            },
            "sensor_records": records,
        }
        errors = validate_manifest(manifest)
        if errors:
            raise CorruptionError("; ".join(errors))
        (working / "input_variation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        working.rename(output)
        return output / "input_variation_manifest.json"
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _fixed_assets(image_path, lidar_path):
    """Read and validate the exact camera and 34,752x5 LiDAR payloads."""
    import cv2
    import numpy as np

    image_path, lidar_path = Path(image_path), Path(lidar_path)
    if not image_path.is_file() or not lidar_path.is_file():
        raise CorruptionError("fixed warmup image and LiDAR files are required")
    image = image_path.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (
            ASTRA_CAMERA["height"], ASTRA_CAMERA["width"]):
        raise CorruptionError("fixed warmup image is not a 1600x900 image")
    lidar = lidar_path.read_bytes()
    values = np.frombuffer(lidar, dtype="<f4")
    if (values.size != FIXED_POINT_COUNT * FIXED_FEATURE_COUNT
            or not np.isfinite(values).all()):
        raise CorruptionError(
            "fixed warmup LiDAR must be finite 34,752x5 float32"
        )
    return image, lidar


def _validate_fixed_pointcloud(message):
    fields = sorted(message.fields, key=lambda field: field.offset)
    if (message.is_bigendian or message.point_step != FIXED_POINT_STEP
            or len(fields) != FIXED_FEATURE_COUNT
            or any(field.offset != index * 4 or field.datatype != 7
                   or field.count != 1
                   for index, field in enumerate(fields))):
        raise CorruptionError(
            "LIDAR_TOP must use five contiguous little-endian float32 fields"
        )


def _transform_full_fixed(source, output, scene_name, scene_token, seed,
                          warmup_image, warmup_lidar):
    """Repeat fixed sensor contents while retaining the source bag envelope."""
    import rosbag2_py
    from rclpy.serialization import serialize_message

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise CorruptionError(f"immutable output already exists: {output}")
    if not source.is_file():
        raise CorruptionError(f"source MCAP file not found: {source}")
    image, lidar = _fixed_assets(warmup_image, warmup_lidar)
    image_hash, lidar_hash = _sha256_bytes(image), _sha256_bytes(lidar)
    bag = _read_full_bag(source)
    camera_headers = sorted(
        _header_timestamp_ns(message)
        for (topic, _timestamp), message in bag["sensor_messages"].items()
        if topic == CAMERA_TOPIC
    )
    lidar_headers = sorted(
        _header_timestamp_ns(message)
        for (topic, _timestamp), message in bag["sensor_messages"].items()
        if topic == LIDAR_TOPIC
    )
    if not camera_headers or not lidar_headers:
        raise CorruptionError("source bag lacks CAM_FRONT or LIDAR_TOP data")
    camera_pairs = pair_camera_lidar_timestamps(
        camera_headers, lidar_headers, MAX_PAIR_DELTA_NS
    )
    pair_by_camera = {
        camera: (paired_lidar, delta)
        for camera, paired_lidar, delta in camera_pairs
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.stage-", dir=output.parent
    ))
    working = stage / output.name
    writer = None
    records = []
    output_digest = hashlib.sha256()
    try:
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(working), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        for metadata in bag["topics"]:
            writer.create_topic(metadata)
        for topic, raw, timestamp_ns in bag["messages"]:
            output_raw = raw
            if topic in (CAMERA_TOPIC, LIDAR_TOPIC):
                message = bag["sensor_messages"][(topic, timestamp_ns)]
                header_ns = _header_timestamp_ns(message)
                content = bytes(message.data)
                record = {
                    "topic": topic,
                    "bag_timestamp_ns": timestamp_ns,
                    "output_bag_timestamp_ns": timestamp_ns,
                    "header_timestamp_ns": header_ns,
                    "output_header_timestamp_ns": header_ns,
                    "message_seed": derived_seed(seed, topic, header_ns),
                    "source_payload_sha256": _sha256_bytes(raw),
                    "source_content_sha256": _sha256_bytes(content),
                }
                if topic == CAMERA_TOPIC:
                    message.data = array("B", image)
                    paired_lidar, delta = pair_by_camera[header_ns]
                    record.update({
                        "paired_lidar_header_timestamp_ns": paired_lidar,
                        "lidar_pairing_delta_ns": delta,
                        "fixed_content_sha256": image_hash,
                        "output_content_sha256": image_hash,
                    })
                else:
                    _validate_fixed_pointcloud(message)
                    message.height = 1
                    message.width = FIXED_POINT_COUNT
                    message.row_step = FIXED_POINT_COUNT * FIXED_POINT_STEP
                    message.is_dense = True
                    message.data = array("B", lidar)
                    record.update({
                        "fixed_content_sha256": lidar_hash,
                        "output_content_sha256": lidar_hash,
                        "output_point_count": FIXED_POINT_COUNT,
                        "output_height": message.height,
                        "output_width": message.width,
                        "output_point_step": message.point_step,
                        "output_row_step": message.row_step,
                        "output_points_finite": True,
                    })
                output_raw = serialize_message(message)
                record["output_payload_sha256"] = _sha256_bytes(output_raw)
                records.append(record)
            _stream_digest_update(output_digest, topic, timestamp_ns,
                                  output_raw)
            writer.write(topic, output_raw, timestamp_ns)
        del writer
        writer = None

        generated = list(working.glob("*.mcap"))
        if len(generated) != 1:
            raise CorruptionError(
                f"fixed writer produced {len(generated)} MCAP files"
            )
        output_mcap = working / source.name
        generated[0].rename(output_mcap)
        metadata_path = working / "metadata.yaml"
        metadata = metadata_path.read_text(encoding="utf-8")
        if generated[0].name not in metadata:
            raise CorruptionError("fixed bag metadata lacks its MCAP filename")
        metadata_path.write_text(
            metadata.replace(generated[0].name, output_mcap.name),
            encoding="utf-8",
        )

        duration_ns = bag["last_ns"] - bag["first_ns"]
        manifest = {
            "schema_version": 1,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "duration_ns": duration_ns,
            "duration_seconds": duration_ns / 1_000_000_000,
            "source_message_count": len(bag["messages"]),
            "output_message_count": len(bag["messages"]),
            "window_start_bag_timestamp_ns": bag["first_ns"],
            "window_end_bag_timestamp_ns": bag["last_ns"],
            "source_bag": str(source),
            "source_bag_sha256": _sha256_file(source),
            "output_bag_directory": str(output),
            "output_bag_files": [{
                "path": str(path.relative_to(working)),
                "sha256": _sha256_file(path),
            } for path in sorted(working.rglob("*.mcap"))],
            "corruption": {
                "type": "fixed", "severity": 0, "seed": seed,
                "full_duration": True,
            },
            "generator": {
                "camera_implementation": "fixed byte payload",
                "lidar_implementation": "fixed 34752x5 float32 payload",
                "randomization": "none",
            },
            "fixed_assets": {
                "camera": {
                    "path": str(Path(warmup_image).resolve()),
                    "bytes": len(image), "sha256": image_hash,
                },
                "lidar": {
                    "path": str(Path(warmup_lidar).resolve()),
                    "bytes": len(lidar), "sha256": lidar_hash,
                    "point_count": FIXED_POINT_COUNT,
                    "feature_count": FIXED_FEATURE_COUNT,
                    "dtype": "float32", "finite": True,
                },
            },
            "source_stream_sha256": bag["source_digest"],
            "output_stream_sha256": output_digest.hexdigest(),
            "preserved_stream_source_sha256": bag["preserved_digest"],
            "preserved_stream_output_sha256": bag["preserved_digest"],
            "pairing": {
                "basis": "sensor header timestamp",
                "tolerance_ns": MAX_PAIR_DELTA_NS,
                "camera_messages": len(camera_headers),
                "lidar_messages": len(lidar_headers),
                "paired_camera_messages": len(camera_pairs),
                "depth_pair_count": len(camera_pairs),
                "unique_depth_source_lidar_messages": len({
                    paired_lidar
                    for _camera, paired_lidar, _delta in camera_pairs
                }),
                "maximum_delta_ns": max(
                    delta for _camera, _lidar, delta in camera_pairs
                ),
            },
            "sensor_records": records,
        }
        errors = validate_manifest(manifest)
        if errors:
            raise CorruptionError("; ".join(errors))
        (working / "input_variation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        working.rename(output)
        return output / "input_variation_manifest.json"
    finally:
        if writer is not None:
            del writer
        shutil.rmtree(stage, ignore_errors=True)


def _transform_legacy(source, output, scene_name, scene_token, kind,
                      severity, seed, corruptions_root, workers):
    """Retain the original clean/snow ten-second behavior."""
    source, output = Path(source).resolve(), Path(output).resolve()
    root = Path(corruptions_root).resolve() if corruptions_root else None
    if output.exists():
        raise CorruptionError(f"immutable output already exists: {output}")
    if kind not in ("clean", "snow"):
        raise CorruptionError(f"unsupported legacy corruption type: {kind}")
    if kind == "snow" and (severity not in range(1, 6) or root is None):
        raise CorruptionError(
            "snow requires severity 1-5 and corruptions root"
        )
    severity = 0 if kind == "clean" else severity

    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import CompressedImage, PointCloud2

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    for metadata in reader.get_all_topics_and_types():
        writer.create_topic(metadata)

    camera_module = _load_camera_module(root) if kind == "snow" else None
    source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    preserved_source, preserved_output = hashlib.sha256(), hashlib.sha256()
    records, start_ns, end_ns = [], None, None
    mie_cache = _ensure_lisa_cache(root, kind)
    context = multiprocessing.get_context("spawn")
    pool = (
        context.Pool(
            workers, initializer=_legacy_lisa_initializer,
            initargs=(str(root), kind),
        ) if kind == "snow" else None
    )
    try:
        while reader.has_next():
            topic, raw, timestamp_ns = reader.read_next()
            if start_ns is None:
                start_ns = int(timestamp_ns)
                end_ns = start_ns + 10_000_000_000
            if int(timestamp_ns) > end_ns:
                break
            output_raw = raw
            if topic in (CAMERA_TOPIC, LIDAR_TOPIC):
                message_seed = derived_seed(seed, topic, timestamp_ns)
                source_count = output_count = None
                if topic == CAMERA_TOPIC:
                    message = deserialize_message(raw, CompressedImage)
                    source_header = _header_timestamp_ns(message)
                    if kind == "snow":
                        message = _corrupt_camera(
                            message,
                            camera_module.ImageAddSnow(severity, message_seed),
                        )
                else:
                    message = deserialize_message(raw, PointCloud2)
                    source_header = _header_timestamp_ns(message)
                    source_count = int(message.width) * int(message.height)
                    if kind == "snow":
                        message, source_count, output_count = \
                            _corrupt_lidar_snow(
                                message, pool, severity, message_seed
                            )
                    else:
                        output_count = source_count
                output_header = _header_timestamp_ns(message)
                output_raw = (
                    serialize_message(message) if kind == "snow" else raw
                )
                records.append({
                    "topic": topic,
                    "bag_timestamp_ns": int(timestamp_ns),
                    "output_bag_timestamp_ns": int(timestamp_ns),
                    "header_timestamp_ns": source_header,
                    "output_header_timestamp_ns": output_header,
                    "message_seed": message_seed,
                    "source_payload_sha256": _sha256_bytes(raw),
                    "output_payload_sha256": _sha256_bytes(output_raw),
                    "source_point_count": source_count,
                    "output_point_count": output_count,
                })
            else:
                _stream_digest_update(
                    preserved_source, topic, timestamp_ns, raw
                )
                _stream_digest_update(
                    preserved_output, topic, timestamp_ns, output_raw
                )
            _stream_digest_update(source_digest, topic, timestamp_ns, raw)
            _stream_digest_update(
                output_digest, topic, timestamp_ns, output_raw
            )
            writer.write(topic, output_raw, timestamp_ns)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        del writer

    camera_times = [record["bag_timestamp_ns"] for record in records
                    if record["topic"] == CAMERA_TOPIC]
    lidar_times = [record["bag_timestamp_ns"] for record in records
                   if record["topic"] == LIDAR_TOPIC]
    pairs = paired_timestamps(camera_times, lidar_times, 100_000_000)
    manifest = {
        "schema_version": 1,
        "scene_name": scene_name,
        "scene_token": scene_token,
        "duration_seconds": 10.0,
        "window_start_bag_timestamp_ns": start_ns,
        "window_end_bag_timestamp_ns": end_ns,
        "source_bag": str(source),
        "source_bag_sha256": _sha256_file(source),
        "output_bag_directory": str(output),
        "output_bag_files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": _sha256_file(path),
            }
            for path in sorted(output.rglob("*.mcap"))
        ],
        "corruption": {"type": kind, "severity": severity, "seed": seed},
        "generator": {
            "repository": (
                "https://github.com/thu-ml/3D_Corruptions_AD"
                if kind == "snow" else None
            ),
            "commit": _git_output(root, "rev-parse", "HEAD")
            if kind == "snow" else None,
            "dirty": bool(_git_output(root, "status", "--porcelain"))
            if kind == "snow" else False,
            "camera_implementation": (
                "Camera_corruptions.ImageAddSnow" if kind == "snow"
                else "identity"
            ),
            "lidar_implementation": (
                "utils.lisa.monte_carlo_lisa(mode=gunn)"
                if kind == "snow" else "identity"
            ),
            "randomization": "sha256-derived per-message/per-point seed",
            "mie_cache_sha256": _sha256_file(mie_cache) if mie_cache else None,
        },
        "source_stream_sha256": source_digest.hexdigest(),
        "output_stream_sha256": output_digest.hexdigest(),
        "preserved_stream_source_sha256": preserved_source.hexdigest(),
        "preserved_stream_output_sha256": preserved_output.hexdigest(),
        "pairing": {
            "tolerance_ns": 100_000_000,
            "camera_messages": len(camera_times),
            "lidar_messages": len(lidar_times),
            "paired_lidar_messages": len(pairs),
        },
        "sensor_records": records,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise CorruptionError("; ".join(errors))
    manifest_path = output / "input_variation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def transform_bag(source, output, scene_name, scene_token, kind,
                  severity=None, seed=0, corruptions_root=None, workers=None,
                  rain_rate=None, astra_root="/opt/rain-rendering",
                  lisa_root="/mmdetection3d_ros2/LISA",
                  nuscenes_metadata_root=(
                      "/mmdetection3d_ros2/data/nuscenes/v1.0-trainval"
                  ),
                  astra_cache=None, full_duration=False, warmup_image=None,
                  warmup_lidar=None):
    """Transform one source MCAP using the selected corruption path."""
    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    if kind == "fixed":
        if (not full_duration or severity is not None or rain_rate is not None
                or corruptions_root is not None):
            raise CorruptionError(
                "fixed requires --full-duration and no weather arguments"
            )
        if warmup_image is None or warmup_lidar is None:
            raise CorruptionError(
                "fixed requires --warmup-image and --warmup-lidar"
            )
        return _transform_full_fixed(
            source, output, scene_name, scene_token, seed,
            warmup_image, warmup_lidar,
        )
    if kind == "rain":
        if rain_rate is None or not full_duration:
            raise CorruptionError(
                "rain requires --rain-rate and --full-duration"
            )
        if severity is not None or corruptions_root is not None:
            raise CorruptionError(
                "--severity/--corruptions-root apply only to snow"
            )
        return _transform_full_rain(
            source, output, scene_name, scene_token, rain_rate, seed,
            astra_root, lisa_root, nuscenes_metadata_root, astra_cache,
            workers,
        )
    if kind == "clean" and full_duration:
        if (rain_rate is not None or severity is not None
                or corruptions_root is not None):
            raise CorruptionError(
                "full clean data does not accept corruption arguments"
            )
        return _transform_full_clean(
            source, output, scene_name, scene_token, seed
        )
    if rain_rate is not None or full_duration:
        raise CorruptionError("rain-only arguments require --type rain")
    return _transform_legacy(
        source, output, scene_name, scene_token, kind, severity, seed,
        corruptions_root, workers,
    )


def main(argv=None):
    """Transform one source MCAP and print its validated manifest path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument(
        "--type", choices=("clean", "fixed", *WEATHER_TYPES), required=True
    )
    parser.add_argument("--severity", type=int)
    parser.add_argument("--rain-rate", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--corruptions-root")
    parser.add_argument("--astra-root", default="/opt/rain-rendering")
    parser.add_argument("--lisa-root", default="/mmdetection3d_ros2/LISA")
    parser.add_argument(
        "--nuscenes-metadata-root",
        default="/mmdetection3d_ros2/data/nuscenes/v1.0-trainval",
    )
    parser.add_argument(
        "--astra-cache",
        help=(
            "particle cache root; defaults to "
            "/mmdetection3d_ros2/data/astra_particles/SCENE_NAME"
        ),
    )
    parser.add_argument("--full-duration", action="store_true")
    parser.add_argument("--warmup-image")
    parser.add_argument("--warmup-lidar")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    try:
        path = transform_bag(
            args.source, args.output, args.scene_name, args.scene_token,
            args.type, severity=args.severity, seed=args.seed,
            corruptions_root=args.corruptions_root, workers=args.workers,
            rain_rate=args.rain_rate, astra_root=args.astra_root,
            lisa_root=args.lisa_root,
            nuscenes_metadata_root=args.nuscenes_metadata_root,
            astra_cache=args.astra_cache, full_duration=args.full_duration,
            warmup_image=args.warmup_image, warmup_lidar=args.warmup_lidar,
        )
    except (CorruptionError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
