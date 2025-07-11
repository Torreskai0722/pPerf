from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Common parameters
        DeclareLaunchArgument("scene", default_value="1"),
        DeclareLaunchArgument("sensor_expected_models", default_value="2"),
        DeclareLaunchArgument("idx", default_value="0"),
        DeclareLaunchArgument("input_type", default_value="publisher"),
        DeclareLaunchArgument("bag_dir", default_value="/mmdetection3d_ros2/data/bag"),
        DeclareLaunchArgument("data_dir", default_value="/mmdetection3d_ros2/outputs/bag"),

        # Model parameters
        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),
        DeclareLaunchArgument("lidar_model_name", default_value="pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d"),
        DeclareLaunchArgument("lidar_depth", default_value="0"),
        DeclareLaunchArgument("lidar_model_mode", default_value="nus"),
        DeclareLaunchArgument("lidar_model_thresh", default_value="0.5"),
        DeclareLaunchArgument("mm_model", default_value=""),
        DeclareLaunchArgument("delay_csv", default_value="delay_log.csv"),
        DeclareLaunchArgument("lidar_stream_priority", default_value="0"),
        DeclareLaunchArgument("image_stream_priority", default_value="0"),

        # Sensor replayer node
        Node(
            package="p_perf",
            executable="sensor_replay_node",
            name="sensor_publisher",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "scene": LaunchConfiguration("scene"),
                "expected_models": 1,
                "index": LaunchConfiguration("idx"),
                "bag_dir": LaunchConfiguration("bag_dir"),
                "data_dir": LaunchConfiguration("data_dir"),
                "is_ms_mode": True
            }]
        ),

        # Inferencer MS node
        Node(
            package="p_perf",
            executable="inferencer_ms_node",
            name="inferencer_ms_node",
            output="screen",
            parameters=[{
                'use_sim_time': True,
                "scene": LaunchConfiguration("scene"),
                "input_type": "bag",
                "image_models_str": LaunchConfiguration("image_model_name"),
                "lidar_models_str": LaunchConfiguration("lidar_model_name"),
                "lidar_model_mode": LaunchConfiguration("lidar_model_mode"),
                "lidar_model_thresh": LaunchConfiguration("lidar_model_thresh"),
                "depth": LaunchConfiguration("image_depth"),
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "lidar_stream_priority": LaunchConfiguration("lidar_stream_priority"),
                "image_stream_priority": LaunchConfiguration("image_stream_priority")
            }]
        ),
    ]) 