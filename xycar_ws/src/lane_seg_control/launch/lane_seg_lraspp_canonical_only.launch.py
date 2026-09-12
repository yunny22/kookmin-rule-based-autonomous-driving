from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _as_bool(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _as_float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _as_int(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    direct_canonical = LaunchConfiguration("direct_canonical_enabled")
    processed_image_topic = "/lane_seg/source_image"
    white_topic = "/lane_seg/white_boundary_mask"
    yellow_topic = "/lane_seg/yellow_centerline_mask"

    geometry_parameters = {
        "src_tl_x_ratio": 472.0 / 1280.0,
        "src_tl_y_ratio": 494.0 / 1024.0,
        "src_tr_x_ratio": 906.0 / 1280.0,
        "src_tr_y_ratio": 486.0 / 1024.0,
        "src_br_x_ratio": 1272.0 / 1280.0,
        "src_br_y_ratio": 612.0 / 1024.0,
        "src_bl_x_ratio": 46.0 / 1280.0,
        "src_bl_y_ratio": 622.0 / 1024.0,
        "dst_left_ratio": _as_float("dst_left_ratio"),
        "dst_right_ratio": _as_float("dst_right_ratio"),
        "dst_top_y_ratio": 0.0,
        "dst_bottom_y_ratio": 479.0 / 660.0,
        "bev_width": 640,
        "bev_height": 660,
        "bev_valid_lateral_margin_px": 0,
        "bev_valid_erode_px": 0,
        "bev_clip_to_source_polygon": False,
        "lateral_m_per_px": 1.4 / 640.0,
        "forward_m_per_px": 1.5 / 660.0,
        "canonical_width": 256,
        "canonical_height": 144,
        "canonical_lateral_range_m": 1.4,
        "canonical_forward_range_m": 1.5,
        "canonical_white_fit_enabled": _as_bool(
            "canonical_white_fit_enabled"
        ),
        "canonical_white_fit_window_count": _as_int(
            "canonical_white_fit_window_count"
        ),
        "canonical_white_fit_margin_px": _as_int(
            "canonical_white_fit_margin_px"
        ),
        "canonical_white_fit_min_pixels": _as_int(
            "canonical_white_fit_min_pixels"
        ),
        "canonical_white_fit_min_centers": _as_int(
            "canonical_white_fit_min_centers"
        ),
        "canonical_white_fit_min_span_px": _as_int(
            "canonical_white_fit_min_span_px"
        ),
        "canonical_white_fit_residual_px": _as_float(
            "canonical_white_fit_residual_px"
        ),
        "canonical_white_fit_line_width_px": _as_int(
            "canonical_white_fit_line_width_px"
        ),
        "canonical_yellow_divider_enabled": _as_bool(
            "canonical_yellow_divider_enabled"
        ),
        "canonical_yellow_divider_min_pixels": _as_int(
            "canonical_yellow_divider_min_pixels"
        ),
        "canonical_yellow_divider_residual_px": _as_float(
            "canonical_yellow_divider_residual_px"
        ),
        "canonical_yellow_divider_line_width_px": _as_int(
            "canonical_yellow_divider_line_width_px"
        ),
        "canonical_yellow_normalize_enabled": _as_bool(
            "canonical_yellow_normalize_enabled"
        ),
        "canonical_yellow_normalize_line_width_px": _as_int(
            "canonical_yellow_normalize_line_width_px"
        ),
    }

    arguments = [
        DeclareLaunchArgument(
            "model_path",
            description="Absolute path to a local LR-ASPP TorchScript model.",
        ),
        DeclareLaunchArgument(
            "image_topic", default_value="/camera/image_raw"
        ),
        DeclareLaunchArgument("use_compressed_image", default_value="false"),
        DeclareLaunchArgument("enable_rectify", default_value="false"),
        DeclareLaunchArgument(
            "camera_yaml",
            default_value="",
        ),
        DeclareLaunchArgument("rect_balance", default_value="0.3"),
        DeclareLaunchArgument("max_input_age_sec", default_value="0.35"),
        DeclareLaunchArgument(
            "direct_canonical_enabled", default_value="true"
        ),
        DeclareLaunchArgument(
            "publish_intermediate_topics", default_value="false"
        ),
        DeclareLaunchArgument("white_confidence", default_value="0.50"),
        DeclareLaunchArgument("yellow_confidence", default_value="0.50"),
        DeclareLaunchArgument("cpu_threads", default_value="4"),
        DeclareLaunchArgument("opencv_threads", default_value="1"),
        DeclareLaunchArgument("pipeline_qos_depth", default_value="1"),
        DeclareLaunchArgument("debug_rate_hz", default_value="1.0"),
        DeclareLaunchArgument("max_output_rate_hz", default_value="0.0"),
        DeclareLaunchArgument(
            "output_native_resolution", default_value="false"
        ),
        DeclareLaunchArgument(
            "dst_left_ratio", default_value=str(80.0 / 640.0)
        ),
        DeclareLaunchArgument(
            "dst_right_ratio", default_value=str(560.0 / 640.0)
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_enabled", default_value="true"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_window_count", default_value="9"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_margin_px", default_value="24"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_min_pixels", default_value="4"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_min_centers", default_value="2"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_min_span_px", default_value="8"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_residual_px", default_value="6.0"
        ),
        DeclareLaunchArgument(
            "canonical_white_fit_line_width_px", default_value="5"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_divider_enabled", default_value="true"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_divider_min_pixels", default_value="3"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_divider_residual_px", default_value="6.0"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_divider_line_width_px", default_value="5"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_normalize_enabled", default_value="false"
        ),
        DeclareLaunchArgument(
            "canonical_yellow_normalize_line_width_px", default_value="5"
        ),
    ]

    inference = Node(
        package="lane_seg_control",
        executable="lane_seg_lraspp_inference_node",
        name="lane_seg_lraspp_inference",
        output="screen",
        additional_env={
            "OMP_NUM_THREADS": LaunchConfiguration("cpu_threads"),
            "OMP_WAIT_POLICY": "PASSIVE",
            "MKL_NUM_THREADS": LaunchConfiguration("cpu_threads"),
            "OPENBLAS_NUM_THREADS": "1",
        },
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "image_topic": image_topic,
                "use_compressed_image": _as_bool("use_compressed_image"),
                "enable_rectify": _as_bool("enable_rectify"),
                "camera_yaml": LaunchConfiguration("camera_yaml"),
                "rect_balance": _as_float("rect_balance"),
                "max_input_age_sec": _as_float("max_input_age_sec"),
                "processed_image_topic": processed_image_topic,
                "white_mask_topic": white_topic,
                "yellow_mask_topic": yellow_topic,
                "input_width": 256,
                "input_height": 144,
                "white_class_id": 1,
                "yellow_class_id": 2,
                "white_confidence": _as_float("white_confidence"),
                "yellow_confidence": _as_float("yellow_confidence"),
                "cpu_threads": _as_int("cpu_threads"),
                "opencv_threads": _as_int("opencv_threads"),
                "output_qos_depth": _as_int("pipeline_qos_depth"),
                "debug_rate_hz": _as_float("debug_rate_hz"),
                "max_output_rate_hz": _as_float("max_output_rate_hz"),
                "output_native_resolution": _as_bool(
                    "output_native_resolution"
                ),
                "direct_canonical_enabled": _as_bool(
                    "direct_canonical_enabled"
                ),
                "publish_intermediate_topics": _as_bool(
                    "publish_intermediate_topics"
                ),
                **geometry_parameters,
            }
        ],
    )

    adapter = Node(
        package="lane_seg_control",
        executable="lane_seg_canonical_adapter",
        name="lane_seg_canonical_adapter",
        output="screen",
        condition=UnlessCondition(direct_canonical),
        parameters=[
            {
                "image_topic": processed_image_topic,
                "white_mask_topic": white_topic,
                "yellow_mask_topic": yellow_topic,
                "input_is_bev": False,
                "sync_qos_depth": _as_int("pipeline_qos_depth"),
                "debug_rate_hz": _as_float("debug_rate_hz"),
                **geometry_parameters,
            }
        ],
    )
    return LaunchDescription([*arguments, inference, adapter])
