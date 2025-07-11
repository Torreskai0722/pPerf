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
        DeclareLaunchArgument("data_dir", default_value="0"),

        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),

        

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
                "scene": LaunchConfiguration("scene"),
                "depth": LaunchConfiguration("image_depth"),
                "model_name": LaunchConfiguration("image_model_name"),
                "mode": "image",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "bag"
            }]
        ),

    ])
