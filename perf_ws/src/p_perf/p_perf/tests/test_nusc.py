import json
from p_perf.nuscenes_instance import get_nuscenes_instance
from p_perf.utils import load_sweep_sd
import os

# def get_scene_name(scene_token):
#     """Get scene name from scene token without loading entire NuScenes dataset"""
#     scene_file = '/mmdetection3d_ros2/data/scene.json'
#     with open(scene_file, 'r') as f:
#         scenes = json.load(f)
#         for scene in scenes:
#             if scene['token'] == scene_token:
#                 return scene['name']
#     raise ValueError(f"Scene token {scene_token} not found")

# scene_tokens = ['73030fb67d3c46cfb5e590168088ae39', 'c3e0e9f6ee8d4170a3d22a6179f1ca3a', 'e7ef871f77f44331aefdebc24ec034b7', '25496f19ffd14bd088cb430bfc01a4d7', '6f5133fe62b240e797bac25aeff8b531', '7deb4760e2244f32b57f9d631b535b66']

# for scene_token in scene_tokens:
#     scene_name = get_scene_name(scene_token)
#     print(f"Scene token: {scene_token}, Scene name: {scene_name}")

nusc = get_nuscenes_instance(version='v1.0-trainval', dataroot='/mmdetection3d_ros2/data/nuscenes')

scene_token = '2f0e54af35964a3fb347359836bec035_rainrate25'
scene = nusc.get('scene', scene_token)
print(scene['name'])
tokens = load_sweep_sd(nusc, scene, sensor_channel='LIDAR_TOP')

count = 0
for token in tokens:
    sd = nusc.get('sample_data', token)
    if os.path.exists(f"{nusc.dataroot}/{sd['filename']}"):
        print(sd['filename'])
        count += 1
    else:
        print(f"File not found: {sd['filename']}")

print(f"Total files found: {count}")