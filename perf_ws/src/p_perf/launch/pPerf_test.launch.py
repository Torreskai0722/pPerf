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

        DeclareLaunchArgument("image_sample_freq", default_value="10"),
        DeclareLaunchArgument("image_model_name", default_value="yolov3_d53_320_273e_coco"),
        DeclareLaunchArgument("image_depth", default_value="0"),

        DeclareLaunchArgument("lidar_sample_freq", default_value="10"),
        DeclareLaunchArgument("lidar_model_name", default_value="pv_rcnn_8xb2-80e_kitti-3d-3class"),
        DeclareLaunchArgument("lidar_model_mode", default_value="nus"),
        DeclareLaunchArgument("lidar_model_thresh", default_value="0.2"),
        DeclareLaunchArgument("lidar_depth", default_value="0"),

        DeclareLaunchArgument("data_dir", default_value="0"),

        Node(
            package="p_perf",
            executable="sensor_publish_node",
            name="sensor_publisher",
            output="screen",
            parameters=[{
                "publish_freq_lidar": LaunchConfiguration("sensor_publish_freq_lidar"),
                "publish_freq_image": LaunchConfiguration("sensor_publish_freq_image"),
                "scene": LaunchConfiguration("scene"),
                "expected_models": LaunchConfiguration("sensor_expected_models"),
                "index": LaunchConfiguration("idx"),
                "data_dir": LaunchConfiguration("data_dir"),
                "lidar_model_mode": LaunchConfiguration("lidar_model_mode")
            }]
        ),

        Node(
            package="p_perf",
            executable="inference_node",
            name="image_inference_node",
            output="screen",
            parameters=[{
                "sample_freq": LaunchConfiguration("image_sample_freq"),
                "depth": LaunchConfiguration("image_depth"),
                "model_name": LaunchConfiguration("image_model_name"),
                "mode": "image",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "publisher"              
            }]
        ),

        Node(
            package="p_perf",
            executable="inference_node",
            name="lidar_inference_node",
            output="screen",
            parameters=[{
                "sample_freq": LaunchConfiguration("lidar_sample_freq"),
                "depth": LaunchConfiguration("lidar_depth"),
                "model_name": LaunchConfiguration("lidar_model_name"),
                "mode": "lidar",
                "data_dir": LaunchConfiguration("data_dir"),
                "index": LaunchConfiguration("idx"),
                "input_type": "publisher",
                "lidar_model_mode": LaunchConfiguration("lidar_model_mode"),
                "lidar_model_thresh": LaunchConfiguration("lidar_model_thresh") 
            }]
        ),
    ])
