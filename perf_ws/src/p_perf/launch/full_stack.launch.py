#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def load_nodes(context, *args, **kwargs):
    """Dynamically load nodes from YAML configuration file"""
    # Get the config file path from launch arguments
    config_file = LaunchConfiguration('config_file').perform(context)
    
    if not config_file:
        print("Error: config_file parameter is required and must be an absolute path")
        return []
    
    try:
        # Load configuration from YAML
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✓ Loaded configuration from: {config_file}")
    except FileNotFoundError:
        print(f"Error: YAML config file not found at {config_file}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []
    
    nodes = []
    
    def create_environment(node_config):
        """Create environment dictionary with CUDA MPS settings for processing nodes"""
        env = {
            'CUDA_MPS_PIPE_DIRECTORY': '/tmp/nvidia-mps',
            'CUDA_MPS_LOG_DIRECTORY': '/tmp/nvidia-log'
        }
        
        # Check if MPS thread percentage is configured for this node
        if 'cuda_mps_thread_percentage' in node_config:
            percentage = node_config['cuda_mps_thread_percentage']
            env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
            print(f"  ✓ Setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={percentage}")
        
        return env
    
    # Launch sensor_replayer if configured
    if 'sensor_replayer' in config:
        sensor_config = config['sensor_replayer']
        sensor_node = Node(
            package='p_perf',
            executable='sensor_replay_node', 
            name='sensor_replayer',
            parameters=[sensor_config],
            output='screen'
        )
        nodes.append(sensor_node)
        print(f"✓ Added sensor_replayer with parameters: {sensor_config}")
    
    # Launch detection inferencer nodes if configured
    if 'det_inferencers' in config:
        for i, det_config in enumerate(config['det_inferencers']):
            # Create a copy of config without non-ROS parameters for ROS parameters
            ros_params = {k: v for k, v in det_config.items() 
                          if k not in ['inferencer_index', 'cuda_mps_thread_percentage']}
            
            # Create environment variables for MPS
            env_vars = create_environment(det_config)
            
            # Create node parameters
            node_params = {
                'package': 'p_perf',
                'executable': 'det_inference_node',
                'name': f'det_inferencer_{det_config.get("inferencer_index")}_{det_config.get("mode")}',
                'parameters': [ros_params],
                'output': 'screen'
            }
            
            # Add environment variables if any are set
            if env_vars:
                node_params['additional_env'] = env_vars
            
            det_node = Node(**node_params)
            nodes.append(det_node)
            print(f"✓ Added det_inferencer_{det_config.get('inferencer_index', i)} with parameters: {ros_params}")
    
    # Launch segmentation inferencer nodes if configured
    if 'seg_inferencers' in config:
        for i, seg_config in enumerate(config['seg_inferencers']):
            # Create a copy of config without non-ROS parameters for ROS parameters
            ros_params = {k: v for k, v in seg_config.items() 
                          if k not in ['inferencer_index', 'cuda_mps_thread_percentage']}
            
            # Create environment variables for MPS
            env_vars = create_environment(seg_config)
            
            # Create node parameters
            node_params = {
                'package': 'p_perf',
                'executable': 'seg_inference_node',
                'name': f'seg_inferencer_{seg_config.get("inferencer_index")}_{seg_config.get("mode")}',
                'parameters': [ros_params],
                'output': 'screen'
            }
            
            # Add environment variables if any are set
            if env_vars:
                node_params['additional_env'] = env_vars
            
            seg_node = Node(**node_params)
            nodes.append(seg_node)
            print(f"✓ Added seg_inferencer_{seg_config.get('inferencer_index', i)} with parameters: {ros_params}")
    
    # Add log info about total nodes
    total_nodes = (
        (1 if 'sensor_replayer' in config else 0) +
        len(config.get('det_inferencers', [])) +
        len(config.get('seg_inferencers', []))
    )
    print(f"✓ Total nodes to launch: {total_nodes}")
    
    return nodes


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Absolute path to YAML configuration file'
    )
    
    # Initialize launch description
    ld = LaunchDescription()
    ld.add_action(config_file_arg)
    
    # Add log info about config file
    ld.add_action(LogInfo(msg=f"Loading configuration from: {LaunchConfiguration('config_file')}"))
    
    # Use OpaqueFunction to dynamically load nodes from YAML config
    ld.add_action(OpaqueFunction(function=load_nodes))
    
    return ld
