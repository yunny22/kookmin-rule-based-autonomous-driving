"""Start the final Xbin-to-RULE boundary without vehicle motor output.

This public launcher is intentionally narrower than the competition launcher:
it demonstrates the final Xbin metric-path and fused-controller connection
only. Traffic, shortcut, cone and vehicle-avoidance candidates must be
provided by their authorised local ROS 2 interfaces before the full mission
arbitrator is run.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument(
                "model_path",
                default_value="/absolute/path/to/local/xbin_centerline_model.pt",
            ),
            DeclareLaunchArgument(
                "direct_centerline_topic",
                default_value="/perception/xbin_direct_centerline",
            ),
            DeclareLaunchArgument(
                "rule_candidate_topic",
                default_value="/hybrid/rule_candidate",
            ),
            DeclareLaunchArgument("lookahead_distance_m", default_value="1.0"),
            DeclareLaunchArgument("command_rate_hz", default_value="10.0"),
            Node(
                package="lane_seg_control",
                executable="lraspp_inference_node",
                name="xbin_centerline_perception",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "enable_rectify": False,
                        "direct_canonical_enabled": False,
                        "direct_centerline_enabled": True,
                        "direct_centerline_topic": LaunchConfiguration(
                            "direct_centerline_topic"
                        ),
                    }
                ],
            ),
            Node(
                package="xycar_rule_drive",
                executable="canonical_stanley_pursuit_driver",
                name="xbin_fused_rule_controller",
                output="screen",
                parameters=[
                    {
                        "drive_enabled": False,
                        "external_path_enabled": True,
                        "external_path_topic": LaunchConfiguration(
                            "direct_centerline_topic"
                        ),
                        "shadow_motor_topic": LaunchConfiguration(
                            "rule_candidate_topic"
                        ),
                        "lookahead_distance_m": _float("lookahead_distance_m"),
                        "command_rate_hz": _float("command_rate_hz"),
                    }
                ],
            ),
        ]
    )
