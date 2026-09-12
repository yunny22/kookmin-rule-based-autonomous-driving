"""Public Rule-only composition derived from the audited final launch branch.

It deliberately excludes RL, Hybrid, model binaries, and vehicle calibration.
Motor output stays disabled unless an operator explicitly changes the parameter
after supplying a locally reviewed configuration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _as_bool(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def generate_launch_description() -> LaunchDescription:
    perception_launch = PathJoinSubstitution(
        [
            FindPackageShare("lane_seg_control"),
            "launch",
            "lane_seg_lraspp_canonical_only.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("use_compressed_image", default_value="false"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(perception_launch),
                launch_arguments={
                    "model_path": LaunchConfiguration("model_path"),
                    "image_topic": LaunchConfiguration("image_topic"),
                    "use_compressed_image": LaunchConfiguration(
                        "use_compressed_image"
                    ),
                    "enable_rectify": "false",
                    "camera_yaml": "",
                    "direct_canonical_enabled": "true",
                    "publish_intermediate_topics": "false",
                }.items(),
            ),
            Node(
                package="xycar_rule_drive",
                executable="canonical_stanley_pursuit_driver",
                name="canonical_stanley_pursuit_driver",
                output="screen",
                parameters=[
                    {
                        "drive_enabled": _as_bool("drive_enabled"),
                        "command_on_canonical": True,
                    }
                ],
            ),
        ]
    )
