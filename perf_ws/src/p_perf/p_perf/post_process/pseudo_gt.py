import os
import json
import torch
from PIL import Image
import dino_package.datasets.transforms as T

from dino_package.util.slconfig import SLConfig
from PIL import ImageDraw, ImageFont

# ---- Set once globally ----
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian']


def load_model(config_path, ckpt_path, device='cuda'):

    def build_model_main(args):
        # we use register to maintain models from catdet6 on.
        from dino_package.models.registry import MODULE_BUILD_FUNCS
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
    img_tensor, _ = transform(image, None)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor[None])
        outputs = postprocessors['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).to(device))[0]

    # Filter by threshold
    scores = outputs['scores']
    keep = scores > threshold
    boxes = outputs['boxes'][keep].cpu().numpy()
    labels = outputs['labels'][keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    # Scale boxes to original image size
    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H

    return boxes, [id2name[int(l)] for l in labels], scores


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

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    config_dir = '/mmdetection3d_ros2/DINO/dino_package/config'
    data_dir = '/mmdetection3d_ros2/data/nuscenes/sweeps/CAM_FRONT'
    config_path = f'{config_dir}/DINO/DINO_4scale_swin.py'
    ckpt_path = f'{config_dir}/ckpts/checkpoint0029_4scale_swin.pth'
    id2name_path = '/mmdetection3d_ros2/DINO/dino_package/util/coco_id2name.json'

    with open(id2name_path) as f:
        id2name = {int(k): v for k, v in json.load(f).items()}

    model, postprocessors = load_model(config_path, ckpt_path)

    # -- run inference on a list of images --
    image_list = [
        f'{data_dir}/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg',
        f'{data_dir}/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603662404.jpg',
        f'{data_dir}/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603762404.jpg'
    ]

    for img_path in image_list:
        boxes, labels, scores = infer_image(img_path, model, postprocessors, id2name)
        vis_image = visualization(img_path, boxes, labels, scores,
                                     save_path=f'/mmdetection3d_ros2/{os.path.basename(img_path)}')
        