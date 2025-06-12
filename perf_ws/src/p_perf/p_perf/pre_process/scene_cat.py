# ENVIRONMENT
# Crowded city
# Scene 429: Jaywalkers, bus, many peds, cross intersection
# Scene 27: Taxi parked on road, many peds, jaywalkers, lane change, crosswalk activity

# Highspeed highway:
# Scene 194: High speed, car overtaking, construction sign
# Scene 612: Peds on sidewalk, children, cross intersection, high speed, bus stop, parked bicycles

# Surburban:
# Scene 709: Cleaner cleaning street, nature, parked car, trash can, parked vehicles, parked bicycle, parking lot, bump, parked bicycle
# Scene 143: Many trees, narrow road

# WEATHER
# Clear
# 201	Many peds, parked bicycles, jaywalkers, arrive at intersection
# 97	Many peds, parked buses and cars

# Rainy
# 389	Rain, many peds, jaywalker, bicycle rack, parked cars
# 364	Rain, many trucks, peds, parking lot

# LIGHTING
# Day:
# 314	Very dense traffic, congestion, overtake car, peds
# 37	Busy street, parked cars, ped at bus stop

# Night:
# 784	Night, motorcycles, dense traffic, congestion
# 768	Night, bus, peds wearing dark clothes, low traffic


# import json
# from nuscenes.nuscenes import NuScenes

# DATA_ROOT = '/mnt/nas/Nuscenes'
# nusc = NuScenes(
#     version='v1.0-trainval',
#     dataroot=DATA_ROOT,
#     verbose=True
# )
# OUTPUT_JSON = 'scene_category_with_real_tokens.json'

# # Scene categorization
# scene_categories = {
#     "ENVIRONMENT": {
#         "Crowded city": [429, 27],
#         "Highspeed highway": [194, 612],
#         "Surburban": [709, 143]
#     },
#     "WEATHER": {
#         "Clear": [201, 97],
#         "Rainy": [389, 364]
#     },
#     "LIGHTING": {
#         "Day": [314, 37],
#         "Night": [784, 768]
#     }
# }

# # Extract real tokens and populate structure
# combined_data = {}
# for category, groups in scene_categories.items():
#     combined_data[category] = {}
#     for label, scenes in groups.items():
#         combined_data[category][label] = []
#         for sid in scenes:
#             if sid < len(nusc.scene):
#                 token = nusc.scene[sid]['token']
#                 combined_data[category][label].append({
#                     "scene_id": sid,
#                     "token": token
#                 })

# # Save result
# with open(OUTPUT_JSON, "w") as f:
#     json.dump(combined_data, f, indent=4)

# print(f"Saved to {OUTPUT_JSON}")

from nuscenes import NuScenes

DATA_ROOT = '/mmdetection3d_ros2/data/nuscenes'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=DATA_ROOT,
    verbose=True
)

# Print token and description for each scene
for scene in nusc.scene:
    token = scene['token']
    description = scene['name']
    print(f"{token}: {description}")

print(nusc.scene)
