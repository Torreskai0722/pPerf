from p_perf.pPerf_inferencer import pPerf2dDetInferencer, pPerf2dSegInferencer, pPerf3dSegInferencer
from p_perf.pPerf import pPerf
import cv2
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector
from mmdet3d.apis import LidarDet3DInferencer, LidarSeg3DInferencer
from mmengine.dataset import Compose

from p_perf.general_utils import convert_to_kitti_ros, visualize_segmentation_with_legend

WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
WARM_IMAGE = cv2.imread(WARM_IMAGE_PATH)
WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)

def test_pPerf2dDetInferencer():
    model_name = 'dino-4scale_r50_8xb2-12e_coco'
    inferencer = pPerf2dDetInferencer(model_name)
    inferencer.show_progress = False
    results = inferencer(dict(img=WARM_IMAGE))

def test_pPerf3dSegInferencer():
    model_name = 'minkunet18_w20_torchsparse_8xb2-amp-15e_semantickitti'
    inferencer = pPerf3dSegInferencer(model_name, weights='/mmdetection3d_ros2/minkunet_w20_8xb2-15e_semantickitti_20230309_160718-c3b92e6e.pth')
    
    # Convert numpy array to structured array with named fields
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('ring', np.float32)
    ])
    structured_pcd = np.array([tuple(row) for row in WARM_PCD], dtype=dtype)
    
    kitti_points = convert_to_kitti_ros(structured_pcd)
    
    # Convert structured array to regular numpy array for MMDetection3D
    kitti_array = np.column_stack((
        kitti_points['x'],
        kitti_points['y'], 
        kitti_points['z'],
        kitti_points['intensity']
    ))
    
    results = inferencer(dict(points=kitti_array))
    print(results)

def test_pPerf2dSegInferencer():
    model = 'deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280'
    inferencer = pPerf2dSegInferencer(model, mode='sem_seg')
    inferencer.model.eval()

    results = inferencer(dict(img=WARM_IMAGE))
    print(results.keys())
    print(results)
    inferencer.show_result(WARM_IMAGE, results)

if __name__ == '__main__':
    test_pPerf2dDetInferencer()
    # test_bdd100k_det_inferencer()
    # test_pPerf3dSegInferencer()
    # test_pPerf2dSegInferencer()