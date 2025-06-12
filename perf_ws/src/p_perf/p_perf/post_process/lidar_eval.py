from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, attr_acc, cummean
import numpy as np
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetricData
)
from nuscenes.eval.detection.algo import calc_ap
import json
import csv
from collections import defaultdict
from pyquaternion import Quaternion

from p_perf.config.constant import nus_lidar_classes, dist_ths
from p_perf.utils import interpolate_gt, get_offset_sd_token
from mmdet3d.structures import LiDARInstance3DBoxes

# HELPER FUNCTION FOR EVALUATION PIPELINE ==> LIDAR
def lidar_output_to_nusc_box(detection, token, score_thresh=0.2, mode='kitti'):
    
    bbox3d = detection.bboxes_3d.to('cpu')
    scores = detection.scores_3d.cpu().numpy()
    labels = detection.labels_3d.cpu().numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        if 'kitti' in mode:
            center_nusc = np.stack([-box_gravity_center[:, 1],
                                    box_gravity_center[:, 0],
                                    box_gravity_center[:, 2]], axis=1)

            # Transform dims: [w, l, h] → [l, w, h]
            dims_nusc = box_dims[:, [1, 0, 2]]

            for i in range(len(bbox3d)):
                if scores[i] < score_thresh:
                    continue

                yaw_kitti = box_yaw[i]
                # Rotate yaw -90 degrees to align KITTI to nuScenes frame
                yaw_nusc = yaw_kitti - np.pi / 2

                quat = Quaternion(axis=[0, 0, 1], radians=yaw_nusc)
                box = NuScenesBox(
                    center_nusc[i],
                    dims_nusc[i],
                    quat,
                    label=labels[i],
                    score=scores[i],
                    token=token
                )
                box_list.append(box)
        elif 'nus' in mode:
            # Already in nuScenes format
            nus_box_dims = box_dims[:, [1, 0, 2]]
            for i in range(len(bbox3d)):
                if scores[i] < score_thresh:
                    continue
                quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
                box = NuScenesBox(
                    box_gravity_center[i],
                    nus_box_dims[i],
                    quat,
                    label=labels[i],
                    score=scores[i],
                    token=token
                )
                box_list.append(box)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    else:
        raise NotImplementedError(
            f'Do not support converting {type(bbox3d)} bboxes to NuScenesBoxes.')

    return box_list

def lidar_nusc_box_to_global(
        nusc,
        sample_data_token: str,
        boxes):
    """
    Convert predicted NuScenesBoxes from LiDAR to global coordinates using sample_data_token.

    Args:
        nusc: NuScene instance
        sample_data_token (str): Token for the sample_data (e.g. from file name).
        boxes (List[NuScenesBox]): Predicted bounding boxes in LiDAR coordinates.

    Returns:
        List[NuScenesBox]: Boxes transformed into global coordinates and filtered.
    """
    # Step 1: Get sensor calibration and ego poses
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Construct transformation matrices
    lidar2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
    lidar2ego_trans = np.array(cs_record['translation'])

    ego2global_rot = Quaternion(pose_record['rotation']).rotation_matrix
    ego2global_trans = np.array(pose_record['translation'])

    # Transform boxes
    box_list = []
    for box in boxes:
        # LiDAR -> Ego
        box.rotate(Quaternion(matrix=lidar2ego_rot))
        box.translate(lidar2ego_trans)

        # Ego -> Global
        box.rotate(Quaternion(matrix=ego2global_rot))
        box.translate(ego2global_trans)

        box_list.append(box)

    return box_list



class lidar_evaluater():
    def __init__(self, prediction_json, nusc, output_dir, index, lidar_classes):
        '''        
        :param result_path: Path to the .json result file provided by the user.
        '''
        self.nusc = nusc
        self.result_path = prediction_json
        self.output_dir = output_dir
        self.index = index
        self.delay_path = f"{output_dir}/delays_{index}.csv"
        self.lidar_classes = lidar_classes


    def accumulate(self,
                pred_boxes: EvalBoxes,
                class_name: str,
                dist_fcn,
                dist_th: float,
                mode: str,
                verbose: bool = False
                ):
        pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
        pred_confs = [box.detection_score for box in pred_boxes_list]

        # Sort by confidence
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        tp, fp, conf = [], [], []
        match_data = {k: [] for k in ['trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err', 'conf']}
        taken = set()
        interpolated_cache = {}
        instance_hits = {}

        # Precompute interpolated GTs per sample_data_token
        for token in pred_boxes.sample_tokens:
            if mode == 'stream':
                sd_offset = get_offset_sd_token(self.nusc, token, 'lidar', self.delay_path)
            else:
                sd_offset = token
            boxes, instance_tokens = interpolate_gt(self.nusc, token, sd_offset, False, [])
            interpolated_cache[token] = (boxes, instance_tokens)

        # Compute total GTs for this class (avoid double counting)
        npos = sum(
            sum(1 for box in boxes if box.detection_name == class_name)
            for boxes, _ in interpolated_cache.values()
        )

        if verbose:
            print(f"GT count for class {class_name}: {npos}")

        # Matching
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            token = pred_box.sample_token
            interpolated_gts, instance_tokens = interpolated_cache[token]

            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(interpolated_gts):
                if gt_box.detection_name == class_name and (token, gt_idx) not in taken:
                    this_dist = dist_fcn(gt_box, pred_box)
                    if this_dist < min_dist:
                        min_dist = this_dist
                        match_gt_idx = gt_idx

            is_match = min_dist < dist_th

            if is_match:
                taken.add((token, match_gt_idx))
                tp.append(1)
                fp.append(0)
                conf.append(pred_box.detection_score)

                gt_box_match = interpolated_gts[match_gt_idx]
                instance_token = instance_tokens[match_gt_idx]

                # Track instance hit
                if instance_token is not None:
                    instance_hits[instance_token] = instance_hits.get(instance_token, 0) + 1

                match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
                match_data['vel_err'].append(0.1)
                match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period))
                match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
                match_data['conf'].append(pred_box.detection_score)
            else:
                tp.append(0)
                fp.append(1)
                conf.append(pred_box.detection_score)

        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_predictions(), instance_hits

        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf = np.array(conf)

        prec = tp / (fp + tp)
        rec = tp / float(npos)

        rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp

        for key in match_data:
            if key == 'conf':
                continue
            tmp = cummean(np.array(match_data[key]))
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

        return DetectionMetricData(
            recall=rec,
            precision=prec,
            confidence=conf,
            trans_err=match_data['trans_err'],
            vel_err=match_data['vel_err'],
            scale_err=match_data['scale_err'],
            orient_err=match_data['orient_err'],
            attr_err=match_data['attr_err']
        ), instance_hits


    def load_prediction_of_sample_tokens(self,
                                        sample_tokens, all, box_cls = DetectionBox):
        """
        Loads object predictions from file.

        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The deserialized results and meta data.
        """

        # Load from file and check that the format is correct.
        with open(self.result_path) as f:
            data = json.load(f)

        assert isinstance(data['results'], dict), 'Error: results must be a dict.'

        # Filter by sample tokens.
        if all:
            results_of_split = data['results']
        else:
            results_of_split = {sample_token: data['results'][sample_token] for sample_token in sample_tokens}

        # Deserialize results and get meta data.
        boxes_of_split = EvalBoxes.deserialize(results_of_split, box_cls)
        return boxes_of_split
    

    def evaluate(self, pred_boxes, mode):
        metrics = {}
        metric_data_list = DetectionMetricDataList()
        all_instance_hits = {dist_th: defaultdict(int) for dist_th in dist_ths}

        for class_name in self.lidar_classes:
            for dist_th in dist_ths:
                md, instance_hits = self.accumulate(pred_boxes, class_name, center_distance, dist_th, mode)
                
                # Record AP
                metric_data_list.set(class_name, dist_th, md)
                ap = calc_ap(md, 0.1, 0.1)
                metrics[(class_name, dist_th)] = ap

                # Aggregate instance hits per threshold
                for inst_token, count in instance_hits.items():
                    all_instance_hits[dist_th][inst_token] += count

        ap_path = f'{self.output_dir}/lidar_{mode}_ap_{self.index}.csv'
        with open(ap_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class_name', 'dist_th', 'ap'])  # header

            for (class_name, dist_th), ap in metrics.items():
                writer.writerow([class_name, dist_th, ap])

        # Show instance hit stats for each distance threshold
        for dist_th, hits_dict in all_instance_hits.items():
            if dist_th != 4:
                continue
            print(f"\n[Threshold {dist_th}] Unique instances matched: {len(hits_dict)}")
            file_path = f"{self.output_dir}/instance_{mode}_{self.index}.json"
            with open(file_path, 'w') as json_file:
                json.dump(hits_dict, json_file, indent=4)


    
    
