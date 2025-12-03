from p_perf.pPerf_inferencer import pPerf2dDetInferencer, bddDetInferencer, bddSegInferencer, pPerf3dSegInferencer
import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet3d.apis import LidarDet3DInferencer, LidarSeg3DInferencer
from mmengine.dataset import Compose

from p_perf.general_utils import convert_to_kitti_ros

WARM_IMAGE_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg'
WARM_IMAGE = cv2.imread(WARM_IMAGE_PATH)
WARM_PCD_PATH = '/mmdetection3d_ros2/perf_ws/src/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin'
WARM_PCD = np.fromfile(WARM_PCD_PATH, dtype=np.float32).reshape(-1, 5)

def test_pPerf2dDetInferencer():
    model_name = 'faster-rcnn_r50_fpn_1x_coco'
    inferencer = pPerf2dDetInferencer(model_name)
    inferencer.show_progress = False
    results = inferencer(dict(img=WARM_IMAGE))
    print(results)


if __name__ == "__main__":
    test_pPerf2dDetInferencer()