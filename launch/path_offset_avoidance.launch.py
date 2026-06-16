from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("pfvtr_path_topic", default_value="/pfvtr/repeat/local_trajectory_corrected"),
        DeclareLaunchArgument("terrain_topic", default_value="/terrain_map"),
        DeclareLaunchArgument("odom_topic", default_value="/taros/ekf_odom"),
        DeclareLaunchArgument("output_path_topic", default_value="/taros/avoidance/path"),
        DeclareLaunchArgument("output_frame_id", default_value="base_link"),
        DeclareLaunchArgument("state_topic", default_value="/taros/avoidance/state"),
        DeclareLaunchArgument("obstacle_cost_threshold", default_value="0.7"),
        DeclareLaunchArgument("obstacle_confirm_duration_s", default_value="1.0"),
        DeclareLaunchArgument("obstacle_clear_duration_s", default_value="2.0"),
        DeclareLaunchArgument("wait_duration_s", default_value="10.0"),
        DeclareLaunchArgument("preferred_side", default_value="right"),
        DeclareLaunchArgument("offset_candidates_m", default_value="[2.0, 3.0, 4.0]"),
        DeclareLaunchArgument("stop_before_obstacle_m", default_value="5.0"),
        DeclareLaunchArgument("crab_start_distance_m", default_value="6.0"),
        DeclareLaunchArgument("detour_plan_length_m", default_value="15.0"),
        DeclareLaunchArgument("detour_plan_length_min_m", default_value="6.0"),
        DeclareLaunchArgument("detour_retry_wait_s", default_value="3.0"),
        DeclareLaunchArgument("detour_check_min_forward_m", default_value="1.0"),
        DeclareLaunchArgument("crab_side_untraversable_m", default_value="15.0"),
        DeclareLaunchArgument("lookahead_check_m", default_value="10.0"),
        DeclareLaunchArgument("grid_cell_size_m", default_value="0.3"),
        DeclareLaunchArgument("crab_vx_mps", default_value="0.5"),
        DeclareLaunchArgument("crab_vy_over_vx", default_value="0.5"),
        DeclareLaunchArgument("crab_cmd_vel_topic", default_value="/cmd_vel_key"),
        DeclareLaunchArgument("crab_align_threshold_m", default_value="0.4"),
        DeclareLaunchArgument("max_crab_out_s", default_value="12.0"),
        DeclareLaunchArgument("debug_mode_topic", default_value="/taros/avoidance/debug/mode"),
        DeclareLaunchArgument("mission_state_topic", default_value="/mission/state"),
        DeclareLaunchArgument("enable_avoidance_without_mission", default_value="false"),
    ]

    node = Node(
        package="terrain_toolkit_ros",
        executable="path_offset_avoidance_node",
        name="path_offset_avoidance",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "pfvtr_path_topic": LaunchConfiguration("pfvtr_path_topic"),
                "terrain_topic": LaunchConfiguration("terrain_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "output_path_topic": LaunchConfiguration("output_path_topic"),
                "output_frame_id": LaunchConfiguration("output_frame_id"),
                "state_topic": LaunchConfiguration("state_topic"),
                "obstacle_cost_threshold": LaunchConfiguration("obstacle_cost_threshold"),
                "obstacle_confirm_duration_s": LaunchConfiguration("obstacle_confirm_duration_s"),
                "obstacle_clear_duration_s": LaunchConfiguration("obstacle_clear_duration_s"),
                "wait_duration_s": LaunchConfiguration("wait_duration_s"),
                "preferred_side": LaunchConfiguration("preferred_side"),
                "offset_candidates_m": LaunchConfiguration("offset_candidates_m"),
                "stop_before_obstacle_m": LaunchConfiguration("stop_before_obstacle_m"),
                "crab_start_distance_m": LaunchConfiguration("crab_start_distance_m"),
                "detour_plan_length_m": LaunchConfiguration("detour_plan_length_m"),
                "detour_plan_length_min_m": LaunchConfiguration("detour_plan_length_min_m"),
                "detour_retry_wait_s": LaunchConfiguration("detour_retry_wait_s"),
                "detour_check_min_forward_m": LaunchConfiguration("detour_check_min_forward_m"),
                "crab_side_untraversable_m": LaunchConfiguration("crab_side_untraversable_m"),
                "lookahead_check_m": LaunchConfiguration("lookahead_check_m"),
                "grid_cell_size_m": LaunchConfiguration("grid_cell_size_m"),
                "crab_vx_mps": LaunchConfiguration("crab_vx_mps"),
                "crab_vy_over_vx": LaunchConfiguration("crab_vy_over_vx"),
                "crab_cmd_vel_topic": LaunchConfiguration("crab_cmd_vel_topic"),
                "crab_align_threshold_m": LaunchConfiguration("crab_align_threshold_m"),
                "max_crab_out_s": LaunchConfiguration("max_crab_out_s"),
                "debug_mode_topic": LaunchConfiguration("debug_mode_topic"),
                "mission_state_topic": LaunchConfiguration("mission_state_topic"),
                "repeat_mission_states": ["REPEAT"],
                "enable_avoidance_without_mission": LaunchConfiguration(
                    "enable_avoidance_without_mission"
                ),
            }
        ],
    )

    return LaunchDescription(args + [node])

