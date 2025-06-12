from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sensor_publish_freq_lidar", default_value="20"),
        DeclareLaunchArgument("sensor_publish_freq_image", default_value="12"),
        DeclareLaunchArgument("scene", default_value="None"),
        DeclareLaunchArgument("sensor_expected_models", default_value="2"),
        DeclareLaunchArgument("idx", default_value="0"),
        
        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),
        DeclareLaunchArgument("image_queue", default_value="1"),

        DeclareLaunchArgument("lidar_model_name", default_value="pv_rcnn_8xb2-80e_kitti-3d-3class"),
        DeclareLaunchArgument("lidar_model_mode", default_value="nus"),
        DeclareLaunchArgument("lidar_model_thresh", default_value="0.2"),
        DeclareLaunchArgument("lidar_depth", default_value="0"),
        DeclareLaunchArgument("lidar_queue", default_value="1"),

        DeclareLaunchArgument("data_dir", default_value="0"),
        DeclareLaunchArgument("bag_dir", default_value="/mmdetection3d_ros2/data/bag"),

        Node(
            package="p_perf",
            executable="sensor_replay_node",
            name="sensor_publisher",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "scene": LaunchConfiguration("scene"),
                "expected_models": LaunchConfiguration("sensor_expected_models"),
                "index": LaunchConfiguration("idx"),
                "bag_dir": LaunchConfiguration("bag_dir"),
                "data_dir": LaunchConfiguration("data_dir"),
            }]
        ),

        Node(
            package="p_perf",
            executable="inference_node",
            name="image_inference_node",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "depth": LaunchConfiguration("image_depth"),
                "model_name": LaunchConfiguration("image_model_name"),
                "mode": "image",
                "scene": LaunchConfiguration("scene"),
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "publisher",
                "image_queue": LaunchConfiguration("image_queue")              
            }]
        ),

        Node(
            package="p_perf",
            executable="inference_node",
            name="multi_modal_inference_node",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "depth": LaunchConfiguration("lidar_depth"),
                "model_name": LaunchConfiguration("lidar_model_name"),
                "mode": "multi-modal",
                "scene": LaunchConfiguration("scene"),
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "publisher",
                "lidar_model_mode": LaunchConfiguration("lidar_model_mode"),
                "lidar_model_thresh": LaunchConfiguration("lidar_model_thresh"),
                "lidar_queue": LaunchConfiguration("lidar_queue")       
            }]
        ),
    ])
