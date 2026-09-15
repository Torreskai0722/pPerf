"""Launch one process per configured model plus the replay coordinator."""

from closeloop_testbed.config import load_runtime_config
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from closeloop_testbed.process_env import fma_environment, model_environment


def _build(context):
    config_path = LaunchConfiguration("config_file").perform(context)
    run_directory = LaunchConfiguration("run_directory").perform(context)
    config = load_runtime_config(config_path).data
    actions = []
    for model in config["models"]:
        environment = model_environment(config, model, run_directory)
        cta_profile = bool(model.get("nvbit_cta_profile"))
        node = Node(
            package="closeloop_testbed", executable="model_node",
            name=model["node_name"], output="screen",
            additional_env=environment,
            sigterm_timeout="120" if cta_profile else "5",
            sigkill_timeout="30" if cta_profile else "5",
            arguments=["--config", config_path, "--model-id", model["id"],
                       "--run-directory", run_directory])
        actions.append(TimerAction(
            period=float(model["launch_offset_seconds"]), actions=[node]
        ))
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(
                reason=f"model {model['id']} exited"))])))
    if "synthetic_fma" in config:
        fma = Node(
            package="closeloop_testbed", executable="fma_workload_node",
            name="closeloop_fma_workload", output="screen",
            additional_env=fma_environment(config, run_directory),
            sigterm_timeout="30", sigkill_timeout="10",
            arguments=["--config", config_path,
                       "--run-directory", run_directory],
        )
        actions.append(fma)
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=fma,
            on_exit=[EmitEvent(event=Shutdown(
                reason="FMA workload exited"))])))
    paired = all("paired_trial" in model for model in config["models"])
    input_scope = "input" in config["recording"]["scopes"]
    if not paired and input_scope:
        relay = Node(
            package="closeloop_testbed", executable="relay_node",
            name="closeloop_communication_relay", output="screen",
            arguments=["--config", config_path,
                       "--run-directory", run_directory])
        actions.append(relay)
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=relay,
            on_exit=[EmitEvent(event=Shutdown(
                reason="communication relay exited"))])))
    replayer = Node(
        package="closeloop_testbed", executable="replayer_node",
        name="closeloop_replayer", output="screen",
        arguments=["--config", config_path,
                   "--run-directory", run_directory])
    actions.append(replayer)
    actions.append(RegisterEventHandler(OnProcessExit(
        target_action=replayer,
        on_exit=[EmitEvent(event=Shutdown(
            reason="replay coordinator exited"))])))
    return actions


def generate_launch_description():
    """Return the dynamic testbed launch description."""
    return LaunchDescription([
        DeclareLaunchArgument("config_file"),
        DeclareLaunchArgument("run_directory", default_value="/tmp/closeloop"),
        OpaqueFunction(function=_build),
    ])
