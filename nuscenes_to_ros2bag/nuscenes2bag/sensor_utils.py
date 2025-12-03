from utils import * 

def get_radar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    print(pc_filename)
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = numpy_pc2.array_to_pointcloud2(pc.pc_data)
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    return msg

def get_lidar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    pc_filesize = os.stat(pc_filename).st_size

    with open(pc_filename, 'rb') as pc_file:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = get_time(sample_data)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = len(msg.fields) * 4 # 4 bytes per field
        msg.row_step = pc_filesize
        msg.width = round(pc_filesize / msg.point_step)
        msg.height = 1 # unordered
        msg.data = pc_file.read()
        return msg

def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data['filename']
    msg = CompressedImage()
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    msg.format = "jpeg"
    with open(jpg_filename, 'rb') as jpg_file:
        msg.data = jpg_file.read()
    return msg

def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    msg_info = CameraInfo()
    msg_info.header.frame_id = frame_id
    msg_info.header.stamp = get_time(sample_data)
    msg_info.height = sample_data['height']
    msg_info.width = sample_data['width']
    msg_info.k[0] = calib['camera_intrinsic'][0][0]
    msg_info.k[1] = calib['camera_intrinsic'][0][1]
    msg_info.k[2] = calib['camera_intrinsic'][0][2]
    msg_info.k[3] = calib['camera_intrinsic'][1][0]
    msg_info.k[4] = calib['camera_intrinsic'][1][1]
    msg_info.k[5] = calib['camera_intrinsic'][1][2]
    msg_info.k[6] = calib['camera_intrinsic'][2][0]
    msg_info.k[7] = calib['camera_intrinsic'][2][1]
    msg_info.k[8] = calib['camera_intrinsic'][2][2]
    
    msg_info.r[0] = 1
    msg_info.r[3] = 1
    msg_info.r[6] = 1
    
    msg_info.p[0] = msg_info.k[0]
    msg_info.p[1] = msg_info.k[1]
    msg_info.p[2] = msg_info.k[2]
    msg_info.p[3] = 0
    msg_info.p[4] = msg_info.k[3]
    msg_info.p[5] = msg_info.k[4]
    msg_info.p[6] = msg_info.k[5]
    msg_info.p[7] = 0
    msg_info.p[8] = 0
    msg_info.p[9] = 0
    msg_info.p[10] = 1
    msg_info.p[11] = 0
    return msg_info

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

def get_tfs(nusc, sample):
    sample_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_lidar['ego_pose_token'])
    stamp = get_time(ego_pose)

    transforms = []

    # create ego transform
    ego_tf = TransformStamped()
    ego_tf.header.frame_id = 'map'
    ego_tf.header.stamp = stamp
    ego_tf.child_frame_id = 'base_link'
    ego_tf.transform = get_transform(ego_pose)
    transforms.append(ego_tf)

    for (sensor_id, sample_token) in sample['data'].items():
        sample_data = nusc.get('sample_data', sample_token)

        # create sensor transform
        sensor_tf = TransformStamped()
        sensor_tf.header.frame_id = 'base_link'
        sensor_tf.header.stamp = stamp
        sensor_tf.child_frame_id = sensor_id
        sensor_tf.transform = get_transform(
            nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token']))
        transforms.append(sensor_tf)

    return transforms

def get_tfmessage(nusc, sample):
    # get transforms for the current sample
    tf_array = TFMessage()
    tf_array.transforms = get_tfs(nusc, sample)

    # add transforms from the next sample to enable interpolation
    next_sample = nusc.get('sample', sample['next']) if sample.get('next') != '' else None
    if next_sample is not None:
        tf_array.transforms += get_tfs(nusc, next_sample)

    return tf_array