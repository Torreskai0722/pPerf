#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os

def generate_launch_description():
    """Launch file for resource manager demo with inference nodes."""
    
    # Launch arguments
    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='/tmp/perf_ws_data',
        description='Directory to store profiling data'
    )
    
    index_arg = DeclareLaunchArgument(
        'index',
        default_value='0',
        description='Index for this test run'
    )
    
    scene_arg = DeclareLaunchArgument(
        'scene',
        default_value='2f0e54af35964a3fb347359836bec035',
        description='Scene token to process'
    )
    
    lidar_model_arg = DeclareLaunchArgument(
        'lidar_model_name',
        default_value='pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
        description='LiDAR detection model name'
    )
    
    image_model_arg = DeclareLaunchArgument(
        'image_model_name',
        default_value='faster_rcnn_r50_fpn_1x_coco',
        description='Image detection model name'
    )
    
    depth_arg = DeclareLaunchArgument(
        'depth',
        default_value='0',
        description='Profiling depth for NVTX annotation'
    )
    
    enable_resource_manager_arg = DeclareLaunchArgument(
        'enable_resource_manager',
        default_value='true',
        description='Whether to enable the resource manager'
    )
    
    # Get launch configurations
    data_dir = LaunchConfiguration('data_dir')
    index = LaunchConfiguration('index')
    scene = LaunchConfiguration('scene')
    lidar_model_name = LaunchConfiguration('lidar_model_name')
    image_model_name = LaunchConfiguration('image_model_name')
    depth = LaunchConfiguration('depth')
    enable_resource_manager = LaunchConfiguration('enable_resource_manager')
    
    # Resource manager node
    resource_manager = Node(
        package='p_perf',
        executable='resource_managing.py',
        name='resource_manager',
        parameters=[{
            'data_dir': data_dir,
            'index': index,
            'perf_output_dir': [data_dir, '/perf_data'],
            'monitor_interval': 0.1
        }],
        condition=IfCondition(enable_resource_manager),
        output='screen'
    )
    
    # LiDAR inference node
    lidar_inferencer = Node(
        package='p_perf',
        executable='det_inferencer.py',
        name='lidar_inferencer',
        parameters=[{
            'mode': 'lidar',
            'model_name': lidar_model_name,
            'depth': depth,
            'index': index,
            'data_dir': data_dir,
            'input_type': 'publisher',
            'lidar_model_mode': 'nus',
            'lidar_queue': 1,
            'image_queue': 1
        }],
        output='screen'
    )
    
    # Image inference node
    image_inferencer = Node(
        package='p_perf',
        executable='det_inferencer.py',
        name='image_inferencer',
        parameters=[{
            'mode': 'image',
            'model_name': image_model_name,
            'depth': depth,
            'index': index,
            'data_dir': data_dir,
            'input_type': 'publisher',
            'lidar_model_mode': 'nus',
            'lidar_queue': 1,
            'image_queue': 1
        }],
        output='screen'
    )
    
    # Sensor publisher node
    sensor_publisher = Node(
        package='p_perf',
        executable='sensor_publisher.py',
        name='sensor_publisher',
        parameters=[{
            'index': index,
            'data_dir': data_dir,
            'scene': scene,
            'sensor_expected_models': 2,
            'publish_freq_lidar': 10.0,
            'publish_freq_image': 10.0
        }],
        output='screen'
    )
    
    return LaunchDescription([
        data_dir_arg,
        index_arg,
        scene_arg,
        lidar_model_arg,
        image_model_arg,
        depth_arg,
        enable_resource_manager_arg,
        
        # Start resource manager first
        resource_manager,
        
        # Start inference nodes
        lidar_inferencer,
        image_inferencer,
        
        # Start sensor publisher last
        sensor_publisher
    ]) 