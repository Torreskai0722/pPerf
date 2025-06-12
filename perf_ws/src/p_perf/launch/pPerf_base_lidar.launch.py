from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sensor_publish_freq_lidar", default_value="20"),
        DeclareLaunchArgument("sensor_publish_freq_image", default_value="12"),
        DeclareLaunchArgument("scene", default_value="None"),
        DeclareLaunchArgument("sensor_expected_models", default_value="1"),
        DeclareLaunchArgument("idx", default_value="0"),

        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),

        DeclareLaunchArgument("lidar_model_name", default_value="pv_rcnn_8xb2-80e_kitti-3d-3class"),
        DeclareLaunchArgument("lidar_model_mode", default_value="nus"),
        DeclareLaunchArgument("lidar_model_thresh", default_value="0.2"),
        DeclareLaunchArgument("lidar_depth", default_value="0"),

        DeclareLaunchArgument("data_dir", default_value="0"),

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
            name="lidar_inference_node",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "scene": LaunchConfiguration("scene"),
                "depth": LaunchConfiguration("lidar_depth"),
                "model_name": LaunchConfiguration("lidar_model_name"),
                "mode": "lidar",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "bag"
            }]
        ),

    ])
