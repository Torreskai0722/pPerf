import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from nuscenes.nuscenes import NuScenes
from p_perf.utils import load_sweep_sd, get_offset_sd_token

import dino_package.datasets.transforms as T
from dino_package.util.slconfig import SLConfig
from dino_package.models.registry import MODULE_BUILD_FUNCS
import cv2

import warnings
warnings.filterwarnings("ignore")

# ---- Set once globally ----
TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

IMAGE_CLASSES = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'person']


def load_model(config_path, ckpt_path, device='cuda'):

    def build_model_main(args):
        # we use register to maintain models from catdet6 on.
        assert args.modelname in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)
        return model, criterion, postprocessors
    
    args = SLConfig.fromfile(config_path)
    args.device = device
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, postprocessors

def infer_image(image_path, model, postprocessors, id2name, threshold=0.5, device='cuda'):
    """
    Runs DINO inference on a single image.

    Returns:
        boxes_px: [N, 4] array of [x1, y1, x2, y2] in pixels
        labels: list of label names
        scores: list of confidence scores
    """
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    img_tensor, _ = TRANSFORM(image, None)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor[None])
        outputs = postprocessors['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).to(device))[0]

    # Step 1: Raw predictions
    scores_all = outputs['scores']
    boxes_all = outputs['boxes'].cpu().numpy()
    labels_all = outputs['labels'].cpu().numpy()
    scores_all = scores_all.cpu().numpy()

    # Step 2: Convert label indices to class names using id2name
    label_names_all = [id2name[int(l)] for l in labels_all]

    # Step 3: Keep only predictions passing both threshold and class filter
    filtered = [
        (box, label_name, score)
        for box, label_name, score in zip(boxes_all, label_names_all, scores_all)
        if score > threshold and label_name in IMAGE_CLASSES
    ]

    # Step 4: Unpack filtered results
    if filtered:
        boxes, labels, scores = zip(*filtered)
        boxes = list(boxes)
        boxes = np.array(boxes)
        labels = list(labels)
        scores = list(scores)
    else:
        boxes, labels, scores = [], [], []
        return boxes, labels, scores

    # Scale boxes to original image size
    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H

    return boxes, labels, scores


def visualization(image, boxes, labels, scores, save_path=None):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (str): Original image file path
        boxes (array): Nx4 array of boxes in [x1, y1, x2, y2] (pixels).
        labels (list of str): List of class names.
        scores (list of float): List of confidence scores.
        save_path (str, optional): If provided, saves the image to this path.

    Returns:
        PIL.Image: The image with boxes drawn.
    """
    image = Image.open(image).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        text = f"{label}: {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Draw background for text
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill='red')
        draw.text((x1, y1), text, fill='white', font=font)

    if save_path:
        image.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    return image


def generate_pseudo_coco_gt(nusc, sample_data_tokens, model, postprocessors, id2name, delay_csv_path, json_path: str, 
                            image_size=(1600, 900), threshold=0.5, streaming=True, model_name='image_model'):
    """
    Generate COCO-format ground truth using model predictions (pseudo-GT) from NuScenes sample_data_tokens.

    Args:
        sample_data_tokens: list of camera sample_data tokens (e.g., CAM_FRONT)
        model: DINO model
        postprocessors: DINO postprocessors
        id2name: dict mapping category ID to label name
        delay_csv_path: a csv file that contains of all the required information regarding a token's coom delay, processing dealy,
        json_path: output path for COCO-format annotation
        nusc: NuScenes instance
        image_size: expected image size (default for CAM_FRONT: 1600x900)
        threshold: confidence threshold for filtering predictions
        streaming: decide whether to use streaming mAP
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build COCO categories
    for i, name in enumerate(IMAGE_CLASSES):
        coco["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "object"
        })

    # detector = Pseudo_Detectron2Detector()

    annotation_id = 0

    for image_id, token in enumerate(sample_data_tokens):
        # Get file path from NuScenes
        sd_rec = nusc.get('sample_data', token)
        sd_offset_token = get_offset_sd_token(nusc, token, model_name, 'image', delay_csv_path)
        sd_offset = nusc.get('sample_data', sd_offset_token)
        if streaming:
            img_path = os.path.join(nusc.dataroot, sd_offset['filename'])
        else:
            img_path = os.path.join(nusc.dataroot, sd_rec['filename'])

        # Register the image in COCO
        coco["images"].append({
            "id": image_id,
            "file_name": sd_offset["filename"] if streaming else sd_rec["filename"],
            "width": image_size[0],
            "height": image_size[1],
            "token": token,
            "offset_token": sd_offset_token
        })

        # Run inference
        result = infer_image(img_path, model, postprocessors, id2name, threshold=threshold)
        boxes, labels, _ = result
        
        # boxes, labels = detector.inference(img_path) 

        for bbox, label in zip(boxes, labels):
            if label not in IMAGE_CLASSES:
                continue  # Skip labels not in allowed list

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            area = w * h
            category_id = IMAGE_CLASSES.index(label) + 1

            coco["annotations"].append({
                "id": int(annotation_id),
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0
            })
            annotation_id += 1

    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Pseudo COCO GT saved {json_path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    config_dir = '/mmdetection3d_ros2/DINO/dino_package/config'
    data_dir = '/mmdetection3d_ros2/data/Nuscenes/sweeps/CAM_FRONT'
    config_path = f'{config_dir}/DINO/DINO_4scale_swin.py'
    ckpt_path = f'{config_dir}/ckpts/checkpoint0029_4scale_swin.pth'
    id2name_path = '/mmdetection3d_ros2/DINO/dino_package/util/coco_id2name.json'

    with open(id2name_path) as f:
        id2name = {int(k): v for k, v in json.load(f).items()}
    
    DATA_ROOT = '/mmdetection3d_ros2/data/Nuscenes'
    nusc = NuScenes(
                version='v1.0-mini',
                dataroot=DATA_ROOT 
            )

    model, postprocessors = load_model(config_path, ckpt_path)
    
    scene = nusc.scene[0]
    sd_tokens = load_sweep_sd(nusc, scene, 'CAM_FRONT')
    
    generate_pseudo_coco_gt(nusc, sd_tokens, model, postprocessors, id2name, 'pseudo.json')
        