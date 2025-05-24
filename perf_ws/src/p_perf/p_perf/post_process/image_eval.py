import numpy as np
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
import json

from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box

from p_perf.utils import interpolate_gt
from p_perf.config.constant import image_classes, nusc

from typing import List, Tuple, Union
from collections import defaultdict

from mmdet.datasets.coco import CocoDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

# PREDICTION PROCESSING
def image_output_to_coco(detection, image_id, score_thresh=0.5):
    """
    Convert image-based detection results to COCO-style format.

    Args:
        pred_instance (InstanceData): Contains `bboxes`, `labels`, and `scores`.
        image_id (int or str): Unique image ID in the COCO dataset.
        category_id_map (dict): Mapping from internal label IDs to COCO category IDs.
        score_thresh (float): Minimum score threshold to keep predictions.

    Returns:
        List[dict]: List of COCO-style prediction dicts for this image.
    """
    bboxes = detection.bboxes.cpu().numpy()
    labels = detection.labels.cpu().numpy()
    labels = _mmdet_to_nusc_labels(labels)
    scores = detection.scores.cpu().numpy()

    coco_results = []

    for bbox, label, score in zip(bboxes, labels, scores):
        # Apply score threshold
        if score < score_thresh:
            continue
        # Convert [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        result = {
            "image_id": image_id,
            "category_id": int(label),
            "bbox": coco_bbox,
            "score": float(score)
        }
        coco_results.append(result)

    return coco_results


def _mmdet_to_nusc_labels(mmdet_labels):
    # MMDetection COCO-style class names
    mmdet_classes = CocoDataset.METAINFO['classes']

    # Direct class name mappings from COCO to NuScenes (no approximation)
    coco_to_nusc = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'bicycle': 'bicycle',
        'motorcycle': 'motorcycle',
        'person': 'pedestrian',
    }

    # Convert: MMDet label index → NuScenes class index
    result = []
    for label_id in mmdet_labels:
        class_name = mmdet_classes[label_id]
        nusc_name = coco_to_nusc.get(class_name)
        if nusc_name in image_classes:
            nusc_index = image_classes.index(nusc_name) + 1
            result.append(nusc_index)

    return np.array(result)


# GROUNDTRUTH processing
def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900),
                        min_iob: float = 0.5) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas,
    return None if no intersection or the intersection area is too small.

    Args:
        corner_coords (List): List of [x, y] points from 3D projection.
        imsize (Tuple): (width, height) of the image canvas.
        min_iob (float): Minimum visible portion required (Intersection-over-BBox).

    Returns:
        Tuple[float, float, float, float] or None: Clipped 2D box in xyxy or None if invalid.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    image_canvas = box(0, 0, imsize[0], imsize[1])

    if not polygon_from_2d_box.is_valid or not polygon_from_2d_box.intersects(image_canvas):
        return None

    intersection = polygon_from_2d_box.intersection(image_canvas)

    # Compute Intersection over Original Box Area
    original_area = polygon_from_2d_box.area
    intersect_area = intersection.area
    iob = intersect_area / original_area if original_area > 0 else 0

    if iob < min_iob:
        return None

    # Get bounding box of the visible (intersected) portion
    intersection_coords = np.array(intersection.exterior.coords)
    min_x = min(intersection_coords[:, 0])
    min_y = min(intersection_coords[:, 1])
    max_x = max(intersection_coords[:, 0])
    max_y = max(intersection_coords[:, 1])

    return min_x, min_y, max_x, max_y


def get_2d_boxes(sample_data_token: str, visibilities: List[str], visibility=True):
    """
    Project 3D boxes from interpolate_gt() into 2D and return records in COCO-style format.
    
    Args:
        sample_data_token (str): Token for the camera image.
        visibilities (List[str]): List of visibility tokens to filter.
        box_cls: Box class (e.g., DetectionBox or NuScenesBox).
        visibility (bool): Whether to apply visibility filtering.

    Returns:
        (list of projected 2D boxes and list of labels (str detection name))
    """
    # Get the sample data and related calibration/pose info
    sd_rec = nusc.get('sample_data', sample_data_token)
    assert sd_rec['sensor_modality'] == 'camera', 'get_2d_boxes only works for camera data!'

    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get interpolated boxes instead of raw annotations
    boxes_3d, instances = interpolate_gt(nusc, sample_data_token, visibility=visibility, visibilities=visibilities)

    bboxes = []
    labels = []

    for box in boxes_3d:
        if box.detection_name not in image_classes:
            continue
        box = Box(box.translation, box.size, Quaternion(box.rotation), name=box.detection_name, token=box.sample_token)

        # Transform from global -> ego
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Transform from ego -> sensor
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Get corners and filter out those behind camera
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        if corners_3d.shape[1] == 0:
            continue  # All corners are behind the camera

        # Project to image plane
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Filter to valid image bounds
        final_coords = post_process_coords(corner_coords)
        if final_coords is None:
            continue
        min_x, min_y, max_x, max_y = final_coords
        bboxes.append([min_x, min_y, max_x, max_y])
        labels.append(box.name)

    bboxes = _xyxy_to_coco(bboxes)
    return bboxes, labels, instances


def _xyxy_to_coco(bboxes_xyxy):
    bboxes_xyxy = np.asarray(bboxes_xyxy)
    x1 = bboxes_xyxy[:, 0]
    y1 = bboxes_xyxy[:, 1]
    x2 = bboxes_xyxy[:, 2]
    y2 = bboxes_xyxy[:, 3]

    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=1)


def generate_coco_gt(sample_data_tokens, json_path: str, image_size=(1600, 900), visibilities=['3', '4']):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Populate the categories section
    for i, name in enumerate(image_classes):
        coco["categories"].append({
            "id": i + 1,  # COCO-style IDs start from 1
            "name": name,
            "supercategory": "object"
        })

    annotation_id = 0

    for image_id, token in enumerate(sample_data_tokens):
        # Get image metadata from NuScenes
        sd_rec = nusc.get('sample_data', token)
        coco["images"].append({
            "id": image_id,
            "file_name": sd_rec["filename"],
            "width": image_size[0],
            "height": image_size[1],
            "token": token
        })

        bboxes, labels, instances = get_2d_boxes(token, visibilities)

        for bbox, label, instance in zip(bboxes, labels, instances):
            if label not in image_classes:
                continue  # skip unknown label

            x, y, w, h = bbox
            area = w * h
            category_id = image_classes.index(label) + 1  # consistent with prediction

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
                "instance_token": instance
            })
            annotation_id += 1

    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"COCO ground truth saved to: {json_path}")


def change_pred_imageid(pred_json_path: str, gt_json_path: str):
    """
    Convert prediction 'image_id' fields from token strings to integer image IDs,
    based on the 'token' field in COCO GT JSON.

    Args:
        pred_json_path (str): Path to prediction JSON (image_id = token).
        gt_json_path (str): Path to ground truth JSON (contains token → image_id).
        output_path (str): Path to save the updated prediction JSON.
    """

    # Load GT JSON and build token → image_id map
    with open(gt_json_path, 'r') as f:
        gt = json.load(f)

    token_to_image_id = {img['token']: img['id'] for img in gt['images'] if 'token' in img}

    # Load prediction JSON
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)

    updated_preds = []
    for pred in preds:
        token = pred['image_id']  # originally a token string
        if token not in token_to_image_id:
            print(f"Warning: token {token} not found in GT images. Skipping.")
            continue
        pred['image_id'] = token_to_image_id[token]
        updated_preds.append(pred)

    # Save updated predictions
    with open(pred_json_path, 'w') as f:
        json.dump(updated_preds, f, indent=2)




class image_evaluater():
    def __init__(self, pred_json, gt_json, nusc, output_dir, index):
        '''        
        :param result_path: Path to the .json result file provided by the user.
        '''
        self.nusc = nusc
        self.pred_json = pred_json
        self.gt_json = gt_json
        self.output_dir = output_dir
        self.index = index
        self.ap_path = f"{output_dir}/image_ap_{index}.csv"
        self.instance_path = f"{output_dir}/image_instance_{index}.csv"
    
    def mAP_evaluate(self):
        # Load GT and predictions
        coco_gt = COCO(self.gt_json)
        coco_dt = coco_gt.loadRes(self.pred_json)

        # Initialize evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def get_instance_hit(self, iou_type='bbox', iou_thresh=0.5):
        """
        Identifies which GT annotations were matched (TP) using COCO evaluation.

        Args:
            gt_json_path (str): Path to ground truth JSON.
            pred_json_path (str): Path to predictions JSON.
            iou_type (str): Evaluation type ('bbox', 'segm', etc.).
            iou_thresh (float): IoU threshold for match (e.g., 0.5).

        Returns:
            Set[int]: A set of GT annotation IDs that were successfully matched.
        """
        coco_gt = COCO(self.gt_json)
        coco_dt = coco_gt.loadRes(self.pred_json)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        coco_eval.params.iouThrs = [iou_thresh]
        coco_eval.evaluate()

        # Build GT ann_id → instance_token mapping
        ann_id_to_instance = {
            ann["id"]: ann["instance_token"]
            for ann in coco_gt.dataset["annotations"]
            if "instance_token" in ann
        }

        instance_hits = defaultdict(int)

        for eval_img in coco_eval.evalImgs:
            if eval_img is None:
                continue

            dt_matches = eval_img['dtMatches'][0]  # IoU = 0.5 (first row), shape: (num_detections,)
            for matched_gt_id in dt_matches:
                if matched_gt_id > 0:
                    instance_token = ann_id_to_instance.get(int(matched_gt_id))
                    if instance_token is not None:
                        instance_hits[instance_token] += 1

        with open(self.instance_path, 'w') as f:
            json.dump(dict(instance_hits), f, indent=2)