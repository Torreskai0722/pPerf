from utils import *

EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}

def get_pose_stamped(stamp):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.pose.orientation.w = 1.0
    return msg

def get_coordinate(ref_lat, ref_lon, bearing, dist):
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(math.sin(lat) * math.cos(angular_distance) + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing))
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(location, pose):
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.

    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).
    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), f"Error: The given location: {location}, has no available reference."

    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
    return lat, lon

def get_transform(data):
    t = Transform()
    t.translation.x = data['translation'][0]
    t.translation.y = data['translation'][1]
    t.translation.z = data['translation'][2]
    
    t.rotation.w = data['rotation'][0]
    t.rotation.x = data['rotation'][1]
    t.rotation.y = data['rotation'][2]
    t.rotation.z = data['rotation'][3]
    
    return t

def get_gps(location, ego_pose, stamp):
    gps = NavSatFix()
    gps.header.frame_id = 'base_link'
    gps.header.stamp = stamp
    gps.status.status = 1
    gps.status.service = 1
    lat, lon = derive_latlon(location, ego_pose)
    gps.latitude = lat
    gps.longitude = lon
    gps.altitude = get_transform(ego_pose).translation.z
    return gps

def write_occupancy_grid(bag, nusc_map, ego_pose, stamp):
    translation = ego_pose['translation']
    rotation = Quaternion(ego_pose['rotation'])
    yaw = quaternion_yaw(rotation) / np.pi * 180
    patch_box = (translation[0], translation[1], 32, 32)
    canvas_size = (patch_box[2] * 10, patch_box[3] * 10)

    drivable_area = nusc_map.get_map_mask(patch_box, yaw, ['drivable_area'], canvas_size)[0]
    drivable_area = (drivable_area * 100).astype(np.int8)

    msg = OccupancyGrid()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = 0.1
    msg.info.width = drivable_area.shape[1]
    msg.info.height = drivable_area.shape[0]
    msg.info.origin.position.x = -16.0
    msg.info.origin.position.y = -16.0
    msg.info.origin.orientation.w = 1.0
    msg.data = drivable_area.flatten().tolist()

    bag.write('/drivable_area', serialize_message(msg), to_nano(stamp))

def load_bitmap(dataroot: str, map_name: str, layer_name: str) -> np.ndarray:
    """render bitmap map layers. Currently these are:
    - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
    - basemap: The HD lidar basemap used for localization and as general context.

    :param dataroot: Path of the nuScenes dataset.
    :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
        'boston-seaport'.
    :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
    """
    # Load bitmap.
    if layer_name == "basemap":
        map_path = os.path.join(dataroot, "maps", "basemap", map_name + ".png")
    elif layer_name == "semantic_prior":
        map_hashes = {
            "singapore-onenorth": "53992ee3023e5494b90c316c183be829",
            "singapore-hollandvillage": "37819e65e09e5547b8a3ceaefba56bb2",
            "singapore-queenstown": "93406b464a165eaba6d9de76ca09f5da",
            "boston-seaport": "36092f0b03a857c6a3403e25b4b7aab3",
        }
        map_hash = map_hashes[map_name]
        map_path = os.path.join(dataroot, "maps", map_hash + ".png")
    else:
        raise Exception("Error: Invalid bitmap layer: %s" % layer_name)

    # Convert to numpy.
    if os.path.exists(map_path):
        image = np.array(Image.open(map_path).convert("L"))
    else:
        raise Exception("Error: Cannot find %s %s! Please make sure that the map is correctly installed." % (layer_name, map_path))

    # Invert semantic prior colors.
    if layer_name == "semantic_prior":
        image = image.max() - image

    return image

def scene_bounding_box(nusc, scene, nusc_map, padding=75.0):
    box = [np.inf, np.inf, -np.inf, -np.inf]
    cur_sample = nusc.get('sample', scene['first_sample_token'])
    while cur_sample is not None:
        sample_lidar = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_lidar['ego_pose_token'])
        x, y = ego_pose['translation'][:2]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        cur_sample = nusc.get('sample', cur_sample['next']) if cur_sample.get('next') != '' else None
    box[0] = max(box[0] - padding, 0.0)
    box[1] = max(box[1] - padding, 0.0)
    box[2] = min(box[2] + padding, nusc_map.canvas_edge[0]) - box[0]
    box[3] = min(box[3] + padding, nusc_map.canvas_edge[1]) - box[1]
    return box

def get_scene_map(nusc, scene, nusc_map, image, stamp):
    x, y, w, h = scene_bounding_box(nusc, scene, nusc_map)
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    img = np.flipud(image)[img_y:img_y+img_h, img_x:img_x+img_w]
    img = (img * (100.0 / 255.0)).astype(np.int8)

    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = 0.1
    msg.info.width = img_w
    msg.info.height = img_h
    msg.info.origin.position.x = x
    msg.info.origin.position.y = y
    msg.info.origin.orientation.w = 1.0
    msg.data = img.flatten().tolist()

    return msg

def get_centerline_markers(nusc, scene, nusc_map, stamp):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained_pose_lists = []
    for pose_list in pose_lists:
        new_pose_list = []
        for pose in pose_list:
            if rectContains(bbox, pose):
                new_pose_list.append(pose)
        if len(new_pose_list) > 0:
            contained_pose_lists.append(new_pose_list)
    
    msg = MarkerArray()
    for i, pose_list in enumerate(contained_pose_lists):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'centerline'
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.frame_locked = True
        marker.scale.x = 0.1
        marker.color.r = 51.0 / 255.0
        marker.color.g = 160.0 / 255.0
        marker.color.b = 44.0 / 255.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for pose in pose_list:
            p = Point()
            p.x = pose[0]
            p.y = pose[1]
            p.z = 0.0
            marker.points.append(p)
        msg.markers.append(marker)

    return msg