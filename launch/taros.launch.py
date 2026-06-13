# filter_support_radius_m - some points are lost some obtained (more blob like)
# filter_support_ratio - put 0. to use all inpainted points

"""Launch terrain_toolkit_ros with Livox-tuned defaults."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:

    args = [
        # ROS / sensor
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/all_lidar/filtered",
            description="PointCloud2 input topic",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="Use simulation time"
        ),
        DeclareLaunchArgument(
            "map_frame", default_value="os_sensor", description="Map TF frame (NOT USED!)"
        ),
        DeclareLaunchArgument(
            "robot_frame", default_value="base_link", description="Robot TF frame"
        ),
        DeclareLaunchArgument(
            "square_half_size", default_value="25.0", description="Half-side of square ROI (m)"
        ),
        # Grid
        DeclareLaunchArgument("resolution", default_value="0.4", description="Grid cell size (m)"),
        DeclareLaunchArgument(
            "x_range", default_value="25.0", description="Grid half-extent in x (m)"
        ),
        DeclareLaunchArgument(
            "y_range", default_value="25.0", description="Grid half-extent in y (m)"
        ),
        # Pipeline
        DeclareLaunchArgument(
            "z_max", default_value="2.5", description="Discard points above this height (m)"
        ),
        DeclareLaunchArgument(
            "primary", default_value="max", description="Height reduction: max | mean | min"
        ),
        DeclareLaunchArgument(
            "inpaint", default_value="true", description="Enable multigrid inpainting"
        ),
        DeclareLaunchArgument(
            "inpaint_coarse_iters", default_value="200", description="Inpaint coarse iterations"
        ),
        DeclareLaunchArgument(
            "inpaint_iters_per_level",
            default_value="50",
            description="Inpaint iterations per pyramid level",
        ),
        DeclareLaunchArgument(
            "smooth_sigma", default_value="0.3", description="Gaussian smoothing sigma (m)"
        ),
        # Outlier
        DeclareLaunchArgument(
            "outlier_enable", default_value="false", description="Enable outlier filtering"
        ),
        DeclareLaunchArgument(
            "outlier_type", default_value="ror", description="Outlier algorithm: ror | sor"
        ),
        DeclareLaunchArgument(
            "outlier_search_radius_m",
            default_value="0.25",
            description="Neighbor search radius (m)",
        ),
        DeclareLaunchArgument(
            "outlier_min_neighbors", default_value="10", description="Min neighbors within radius"
        ),
        DeclareLaunchArgument(
            "outlier_std_multiplier",
            default_value="1.0",
            description="SOR std multiplier (unused for ROR)",
        ),
        # Traversability
        DeclareLaunchArgument(
            "trav_enable", default_value="true", description="Compute traversability cost layers"
        ),
        DeclareLaunchArgument(
            "trav_max_slope_deg",
            default_value="60.0",
            description="Slope saturating cost to 1 (deg)",
        ),
        DeclareLaunchArgument(
            "trav_max_step_height_m",
            default_value="0.6",
            description="Upward step saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_max_drop_height_m",
            default_value="0.5",
            description="Downward drop saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_max_roughness_m",
            default_value="0.4",
            description="Roughness saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_step_window_radius_m",
            default_value="0.4",
            description="Morphological window radius for step detection (m)",
        ),
        DeclareLaunchArgument(
            "trav_roughness_window_radius_m",
            default_value="0.4",
            description="Window radius for roughness std-dev (m)",
        ),
        DeclareLaunchArgument(
            "trav_slope_weight", default_value="0.2", description="Slope weight in combined cost"
        ),
        DeclareLaunchArgument(
            "trav_step_weight", default_value="0.6", description="Step weight in combined cost"
        ),
        DeclareLaunchArgument(
            "trav_roughness_weight",
            default_value="0.2",
            description="Roughness weight in combined cost",
        ),
        # Temporal filter
        DeclareLaunchArgument(
            "filter_enable",
            default_value="true",
            description="Enable obstacle inflation + temporal gate",
        ),
        DeclareLaunchArgument(
            "filter_support_radius_m",
            default_value="2.0",
            description="Neighborhood radius for support check (m)",
        ),
        DeclareLaunchArgument(
            "filter_support_ratio",
            default_value="0.05",
            description="Min fraction of measured cells to keep",
        ),
        DeclareLaunchArgument(
            "filter_inflation_sigma_m",
            default_value="1.9",
            description="Gaussian sigma for obstacle dilation (m)",
        ),
        DeclareLaunchArgument(
            "filter_obstacle_threshold",
            default_value="0.7",
            description="Cost threshold for obstacle source",
        ),
        DeclareLaunchArgument(
            "filter_obstacle_growth_threshold",
            default_value="2.0",
            description="Reject frame if obstacle count grows by this factor",
        ),
        DeclareLaunchArgument(
            "filter_rejection_limit_frames",
            default_value="5",
            description="Force-accept after this many consecutive rejections",
        ),
        DeclareLaunchArgument(
            "filter_min_obstacle_baseline",
            default_value="10",
            description="Skip hysteresis until this many obstacles seen",
        ),
    ]

    lc = LaunchConfiguration

    launch_dir = os.path.join(get_package_share_directory("terrain_toolkit_ros"), "launch")

    lidar_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "lidar_tf.launch.py"))
    )

    lidar_body_filter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "lidar_body_filter.launch.py"))
    )

    node = Node(
        package="terrain_toolkit_ros",
        executable="terrain_toolkit_node",
        name="terrain_toolkit",
        output="screen",
        parameters=[
            {
                "use_sim_time": lc("use_sim_time"),
                # ROS / sensor
                "lidar_topic": lc("lidar_topic"),
                "map_frame": lc("map_frame"),
                "robot_frame": lc("robot_frame"),
                "square_half_size": lc("square_half_size"),
                # Grid
                "resolution": lc("resolution"),
                "x_range": lc("x_range"),
                "y_range": lc("y_range"),
                # Pipeline
                "z_max": lc("z_max"),
                "primary": lc("primary"),
                "inpaint": lc("inpaint"),
                "inpaint_coarse_iters": lc("inpaint_coarse_iters"),
                "inpaint_iters_per_level": lc("inpaint_iters_per_level"),
                "smooth_sigma": lc("smooth_sigma"),
                # Outlier
                "outlier_enable": lc("outlier_enable"),
                "outlier_type": lc("outlier_type"),
                "outlier_search_radius_m": lc("outlier_search_radius_m"),
                "outlier_min_neighbors": lc("outlier_min_neighbors"),
                "outlier_std_multiplier": lc("outlier_std_multiplier"),
                # Traversability
                "trav_enable": lc("trav_enable"),
                "trav_max_slope_deg": lc("trav_max_slope_deg"),
                "trav_max_step_height_m": lc("trav_max_step_height_m"),
                "trav_max_drop_height_m": lc("trav_max_drop_height_m"),
                "trav_max_roughness_m": lc("trav_max_roughness_m"),
                "trav_step_window_radius_m": lc("trav_step_window_radius_m"),
                "trav_roughness_window_radius_m": lc("trav_roughness_window_radius_m"),
                "trav_slope_weight": lc("trav_slope_weight"),
                "trav_step_weight": lc("trav_step_weight"),
                "trav_roughness_weight": lc("trav_roughness_weight"),
                # Temporal filter
                "filter_enable": lc("filter_enable"),
                "filter_support_radius_m": lc("filter_support_radius_m"),
                "filter_support_ratio": lc("filter_support_ratio"),
                "filter_inflation_sigma_m": lc("filter_inflation_sigma_m"),
                "filter_obstacle_threshold": lc("filter_obstacle_threshold"),
                "filter_obstacle_growth_threshold": lc("filter_obstacle_growth_threshold"),
                "filter_rejection_limit_frames": lc("filter_rejection_limit_frames"),
                "filter_min_obstacle_baseline": lc("filter_min_obstacle_baseline"),
            }
        ],
    )

    return LaunchDescription(args + [lidar_body_filter, node])
