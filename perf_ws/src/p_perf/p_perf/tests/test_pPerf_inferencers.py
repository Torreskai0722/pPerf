from p_perf.pPerf_inferencer import pPerf2dDetInferencer, bddDetInferencer, bddSegInferencer, pPerf3dSegInferencer
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
    model_name = 'faster-rcnn_r50_fpn_1x_coco'
    inferencer = pPerf2dDetInferencer(model_name)
    inferencer.show_progress = False
    results = inferencer(dict(img=WARM_IMAGE))
    print(results)

def test_bdd100k_det_inferencer():
    model_name = 'faster_rcnn_r50_fpn_dconv_1x_det_bdd100k'
    inferencer = bddDetInferencer(model_name)
    inferencer.show_progress = False
    results = inferencer(dict(img=WARM_IMAGE))
    print(results)
    inferencer.show_result(WARM_IMAGE, results)

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

def test_bddSegInferencer(n_runs=10, splits_h=2, splits_w=2):
    # BDD100K class names
    BDD_CLASS_NAMES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle'
    ]
    
    modes = ['sem_seg']
    models = ['deeplabv3+_r50-d8_512x1024_40k_sem_seg_bdd100k']
    for mode, model in zip(modes, models):
        inferencer = bddSegInferencer(model, mode=mode)
        inferencer.model.eval()
        
        module = inferencer.model.decode_head.bottleneck_spatial
        module.splits_h = splits_h
        module.splits_w = splits_w
        
        # Initialize pPerf profiler
        profiler = pPerf(model, inferencer, depth=1, mode='image')
        
        # Warm up and register hooks
        profiler.warm_up(dict(img=WARM_IMAGE))
        profiler.register_hooks(dict(img=WARM_IMAGE))
        
        # Run inference with NVTX annotation
        last_result = None
        for i in range(n_runs):
            torch.cuda.nvtx.range_push(f"{model}.e2e")
            try:
                results = profiler.run_inference(dict(img=WARM_IMAGE), input_name=f"input_{i}")
                # print(results)
                # Handle both list and single DetDataSample results
                if isinstance(results, list):
                    last_result = results[0] if results else None
                else:
                    last_result = results
            except Exception as e:
                print(f"Error processing {model}: {e}")

            torch.cuda.nvtx.range_pop()
        
        # Save visualization of the last result
        if last_result is not None:
            output_path = f'seg_vis_h{splits_h}_w{splits_w}.png'
            print(f"\n✅ Saving visualization to: {output_path}")
            visualize_segmentation_with_legend(
                last_result,
                output_path=output_path,
                alpha=0.5,
                palette='bdd',
                class_names=BDD_CLASS_NAMES
            )


if __name__ == '__main__':
    # test_pPerf2dDetInferencer()
    # test_bdd100k_det_inferencer()
    # test_pPerf3dSegInferencer()
    test_bddSegInferencer(1, 1, 1)
    test_bddSegInferencer(1, 2, 2)
    test_bddSegInferencer(1, 2, 1)
    test_bddSegInferencer(1, 4, 1)