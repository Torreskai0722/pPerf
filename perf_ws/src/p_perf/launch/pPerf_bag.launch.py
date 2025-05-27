from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value="1"),
        DeclareLaunchArgument("sensor_expected_models", default_value="2"),
        DeclareLaunchArgument("idx", default_value="0"),
        DeclareLaunchArgument("input_type", default_value="publisher"),
        DeclareLaunchArgument("bag_dir", default_value="/mmdetection3d_ros2/data/bag"),

        DeclareLaunchArgument("image_sample_freq", default_value="10"),
        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),

        DeclareLaunchArgument("lidar_sample_freq", default_value="10"),
        DeclareLaunchArgument("lidar_model_name", default_value="pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d"),
        DeclareLaunchArgument("lidar_depth", default_value="0"),

        DeclareLaunchArgument("data_dir", default_value="/mmdetection3d_ros2/outputs/bag"),

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
                "sample_freq": LaunchConfiguration("image_sample_freq"),
                "depth": LaunchConfiguration("image_depth"),
                "model_name": LaunchConfiguration("image_model_name"),
                "mode": "image",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "bag"
            }]
        ),

        Node(
            package="p_perf",
            executable="inference_node",
            name="lidar_inference_node",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "sample_freq": LaunchConfiguration("lidar_sample_freq"),
                "depth": LaunchConfiguration("lidar_depth"),
                "model_name": LaunchConfiguration("lidar_model_name"),
                "mode": "lidar",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "bag"
            }]
        ),
    ])
