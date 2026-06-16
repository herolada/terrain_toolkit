#!/usr/bin/env python3
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String, UInt8


class AvoidState(str, enum.Enum):
    PASSTHROUGH = "passthrough"
    FOLLOW = "follow"
    STOP = "stop"
    WAIT = "wait"
    CRAB_OUT = "crab_out"
    PARALLEL = "parallel"
    CRAB_OUT_MORE = "crab_out_more"
    CRAB_SWITCH_SIDE = "crab_switch_side"


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float
    stamp: rclpy.time.Time


class PathOffsetAvoidanceNode(Node):
    """
    Simple local avoidance node implementing:
      FOLLOW -> STOP -> WAIT -> CRAB_OUT -> PARALLEL (+ optional CRAB_OUT_MORE / CRAB_SWITCH_SIDE).

    This is intentionally lightweight and conservative. It only republishes / modifies
    the PFVTR local_trajectory_corrected path when a traversability obstacle is
    detected ahead.

    Output path is published in base_link (same as PFVTR) so path_mpc can keep
    reference_path_in_robot_frame: true. Committed detours are stored in odom
    internally and converted to base_link on each publish tick.

    Timing (defaults):
      FOLLOW  — full path; obstacle must persist obstacle_confirm_duration_s (1 s)
      STOP    — truncated path; MPC brakes toward stop line
      WAIT    — hold stop line for wait_duration_s (10 s) if obstacle remains
      CRAB_OUT — mux crab cmd_vel to lateral offset lane
      PARALLEL — MPC tracks offset detour path

    Resume FOLLOW from STOP/WAIT/CRAB_OUT only after obstacle_clear_duration_s (2 s)
    with no blocked cells ahead.
    """

    def __init__(self) -> None:
        super().__init__("path_offset_avoidance")

        # Parameters (mostly mirrored from the plan file).
        self.declare_parameter("pfvtr_path_topic", "/pfvtr/repeat/local_trajectory_corrected")
        self.declare_parameter("terrain_topic", "/terrain_map")
        self.declare_parameter("odom_topic", "/taros/ekf_odom")
        self.declare_parameter("output_path_topic", "/taros/avoidance/path")
        self.declare_parameter("output_frame_id", "base_link")
        self.declare_parameter("state_topic", "/taros/avoidance/state")

        self.declare_parameter("obstacle_cost_threshold", 0.7)
        self.declare_parameter(
            "obstacle_confirm_duration_s",
            1.0,
        )  # persist before truncating path (STOP)
        self.declare_parameter(
            "obstacle_clear_duration_s",
            2.0,
        )  # path must stay clear this long before resuming FOLLOW
        self.declare_parameter(
            "wait_duration_s",
            10.0,
        )  # wait at stop line before CRAB_OUT manoeuvre
        self.declare_parameter("preferred_side", "right")  # or "left"
        self.declare_parameter("offset_candidates_m", [2.0, 3.0, 4.0])
        self.declare_parameter("stop_before_obstacle_m", 5.0)
        self.declare_parameter("crab_start_distance_m", 6.0)
        self.declare_parameter("crab_side_untraversable_m", 15.0)
        self.declare_parameter("detour_plan_length_m", 15.0)
        self.declare_parameter("detour_plan_length_min_m", 6.0)
        self.declare_parameter("detour_retry_wait_s", 3.0)
        self.declare_parameter("detour_check_min_forward_m", 1.0)
        self.declare_parameter("lookahead_check_m", 10.0)
        self.declare_parameter("grid_cell_size_m", 0.3)  # for terrain grid quantization

        # Crab command parameters.
        # vx used during crab manoeuvre (m/s); vy is derived from vx * vy_over_vx.
        self.declare_parameter("crab_vx_mps", 0.5)
        # vy / vx ratio during crab (max ~0.5 for 35 deg beta).
        self.declare_parameter("crab_vy_over_vx", 0.5)
        # Hijacks twist_mux via /cmd_vel_key (priority 30, above MPC autonomy_low).
        self.declare_parameter("crab_cmd_vel_topic", "/cmd_vel_key")
        self.declare_parameter("crab_align_threshold_m", 0.4)
        self.declare_parameter("max_crab_out_s", 12.0)
        self.declare_parameter("debug_mode_topic", "/taros/avoidance/debug/mode")

        # Only run obstacle avoidance during elrob mission repeat states.
        self.declare_parameter("mission_state_topic", "/mission/state")
        self.declare_parameter(
            "repeat_mission_states",
            ["REPEAT"],
        )
        self.declare_parameter("enable_avoidance_without_mission", False)

        self.pfvtr_path_topic: str = self.get_parameter("pfvtr_path_topic").get_parameter_value().string_value
        self.terrain_topic: str = self.get_parameter("terrain_topic").get_parameter_value().string_value
        self.odom_topic: str = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.output_path_topic: str = self.get_parameter("output_path_topic").get_parameter_value().string_value
        self.output_frame_id: str = (
            self.get_parameter("output_frame_id").get_parameter_value().string_value or "base_link"
        )
        self.state_topic: str = self.get_parameter("state_topic").get_parameter_value().string_value

        self.obstacle_cost_threshold: float = float(
            self.get_parameter("obstacle_cost_threshold").get_parameter_value().double_value
        )
        self.obstacle_confirm_duration_s: float = float(
            self.get_parameter("obstacle_confirm_duration_s").get_parameter_value().double_value
        )
        self.obstacle_clear_duration_s: float = float(
            self.get_parameter("obstacle_clear_duration_s").get_parameter_value().double_value
        )
        self.wait_duration_s: float = float(
            self.get_parameter("wait_duration_s").get_parameter_value().double_value
        )
        self.preferred_side: str = (
            self.get_parameter("preferred_side").get_parameter_value().string_value.lower() or "right"
        )
        self.offset_candidates_m: List[float] = list(
            self.get_parameter("offset_candidates_m").get_parameter_value().double_array_value
        )
        if not self.offset_candidates_m:
            self.offset_candidates_m = [2.0, 3.0, 4.0]

        self.stop_before_obstacle_m: float = float(
            self.get_parameter("stop_before_obstacle_m").get_parameter_value().double_value
        )
        # Ensure enough longitudinal room for the widest lateral detour offset.
        self._stop_margin_boost_m: float = 0.0
        self.crab_start_distance_m: float = float(
            self.get_parameter("crab_start_distance_m").get_parameter_value().double_value
        )
        self.crab_side_untraversable_m: float = float(
            self.get_parameter("crab_side_untraversable_m").get_parameter_value().double_value
        )
        self.detour_plan_length_m: float = float(
            self.get_parameter("detour_plan_length_m").get_parameter_value().double_value
        )
        self.detour_plan_length_min_m: float = float(
            self.get_parameter("detour_plan_length_min_m").get_parameter_value().double_value
        )
        self.detour_retry_wait_s: float = float(
            self.get_parameter("detour_retry_wait_s").get_parameter_value().double_value
        )
        self.detour_check_min_forward_m: float = float(
            self.get_parameter("detour_check_min_forward_m").get_parameter_value().double_value
        )
        self.lookahead_check_m: float = float(
            self.get_parameter("lookahead_check_m").get_parameter_value().double_value
        )
        self.grid_cell_size_m: float = float(
            self.get_parameter("grid_cell_size_m").get_parameter_value().double_value
        )
        self.crab_vx_mps: float = float(
            self.get_parameter("crab_vx_mps").get_parameter_value().double_value
        )
        self.crab_vy_over_vx: float = float(
            self.get_parameter("crab_vy_over_vx").get_parameter_value().double_value
        )
        self.crab_align_threshold_m: float = float(
            self.get_parameter("crab_align_threshold_m").get_parameter_value().double_value
        )
        self.max_crab_out_s: float = float(
            self.get_parameter("max_crab_out_s").get_parameter_value().double_value
        )
        _crab_cmd_vel_topic: str = (
            self.get_parameter("crab_cmd_vel_topic").get_parameter_value().string_value
        )
        _debug_mode_topic: str = (
            self.get_parameter("debug_mode_topic").get_parameter_value().string_value
        )
        self._mission_state_topic: str = (
            self.get_parameter("mission_state_topic").get_parameter_value().string_value
        )
        self._repeat_mission_states: set[str] = set(
            self.get_parameter("repeat_mission_states").value
        )
        if not self._repeat_mission_states:
            self._repeat_mission_states = {"REPEAT"}
        self._enable_avoidance_without_mission: bool = bool(
            self.get_parameter("enable_avoidance_without_mission").value
        )

        # Subscriptions.
        self._pfvtr_path_sub = self.create_subscription(
            Path, self.pfvtr_path_topic, self._pfvtr_path_cb, 10
        )
        self._terrain_sub = self.create_subscription(
            PointCloud2, self.terrain_topic, self._terrain_cb, 10
        )
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, 10
        )
        self._mission_state_sub = self.create_subscription(
            String, self._mission_state_topic, self._mission_state_cb, 10
        )

        # Publications.
        self._path_pub = self.create_publisher(Path, self.output_path_topic, 10)
        self._state_pub = self.create_publisher(String, self.state_topic, 10)

        # Crab cmd hijacks twist_mux via /cmd_vel_key during CRAB_OUT only.
        # PARALLEL is driven by path_mpc on /taros/avoidance/path (MPC crab gate).
        self._crab_cmd_vel_pub = self.create_publisher(TwistStamped, _crab_cmd_vel_topic, 10)
        self._debug_mode_pub = self.create_publisher(UInt8, _debug_mode_topic, 10)
        self.get_logger().info(
            f"Crab cmd_vel topic={_crab_cmd_vel_topic} (mux key), "
            f"path output={self.output_path_topic} frame={self.output_frame_id}, "
            f"mission gate={sorted(self._repeat_mission_states)} on {self._mission_state_topic}, "
            f"mode debug={_debug_mode_topic}"
        )

        # Timer to run state machine at fixed rate.
        self._timer = self.create_timer(0.1, self._tick)

        # Internal buffers.
        self._latest_pfvtr_path: Optional[Path] = None
        self._latest_terrain_grid: Optional[Dict[Tuple[int, int], float]] = None
        self._latest_terrain_stamp: Optional[rclpy.time.Time] = None
        self._latest_odom: Optional[RobotPose] = None

        # State machine.
        self._state: AvoidState = AvoidState.FOLLOW
        self._wait_start: Optional[rclpy.time.Time] = None
        self._obstacle_first_seen: Optional[rclpy.time.Time] = None
        self._obstacle_cleared_since: Optional[rclpy.time.Time] = None
        self._wait_for_crab_s: float = self.wait_duration_s

        # Detour definition (in odom frame).
        self._detour_odom_path: Optional[np.ndarray] = None  # shape (N, 3) x,y,yaw
        self._detour_side: Optional[int] = None  # -1 right, +1 left (base_link y-left)
        self._detour_offset_m: float = 0.0
        self._detour_progress_m: float = 0.0
        self._crab_out_start: Optional[rclpy.time.Time] = None
        self._mux_hijack_active: bool = False
        self._mission_state: Optional[str] = None
        self._avoidance_was_enabled: bool = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pfvtr_path_cb(self, msg: Path) -> None:
        self._latest_pfvtr_path = msg

    def _terrain_cb(self, msg: PointCloud2) -> None:
        grid = self._build_terrain_grid(msg)
        if grid is None:
            return
        self._latest_terrain_grid = grid
        self._latest_terrain_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

    def _odom_cb(self, msg: Odometry) -> None:
        yaw = self._yaw_from_quat(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._latest_odom = RobotPose(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw,
            stamp=rclpy.time.Time.from_msg(msg.header.stamp),
        )

    def _mission_state_cb(self, msg: String) -> None:
        self._mission_state = str(msg.data).strip()

    def _avoidance_enabled(self) -> bool:
        if self._mission_state is None:
            return self._enable_avoidance_without_mission
        return self._mission_state in self._repeat_mission_states

    def _reset_avoidance_logic(self) -> None:
        self._release_mux_cmd()
        self._state = AvoidState.PASSTHROUGH
        self._wait_start = None
        self._crab_out_start = None
        self._obstacle_first_seen = None
        self._obstacle_cleared_since = None
        self._detour_odom_path = None
        self._detour_side = None
        self._detour_offset_m = 0.0
        self._detour_progress_m = 0.0
        self._stop_margin_boost_m = 0.0
        self._wait_for_crab_s = self.wait_duration_s

    def _now(self) -> rclpy.time.Time:
        if self._latest_odom is not None:
            return self._latest_odom.stamp
        return self.get_clock().now()

    def _update_clear_latch(self, path: Path) -> bool:
        """Return True once the path ahead has been clear for obstacle_clear_duration_s."""
        blocked, _, _ = self._find_blocked_segment(path)
        now = self._now()
        if blocked:
            self._obstacle_cleared_since = None
            return False
        if self._obstacle_cleared_since is None:
            self._obstacle_cleared_since = now
            return False
        elapsed_s = (now - self._obstacle_cleared_since).nanoseconds * 1e-9
        return elapsed_s >= self.obstacle_clear_duration_s

    def _resume_follow(self, path: Path, reason: str) -> None:
        self._release_mux_cmd()
        self._obstacle_first_seen = None
        self._obstacle_cleared_since = None
        self._wait_start = None
        self._crab_out_start = None
        self._detour_odom_path = None
        self._detour_side = None
        self._stop_margin_boost_m = 0.0
        self._wait_for_crab_s = self.wait_duration_s
        self._state = AvoidState.FOLLOW
        self._publish_base_link_path(path)
        self.get_logger().info(reason)

    def _effective_stop_before_obstacle_m(self) -> float:
        return max(
            self.stop_before_obstacle_m,
            max(self.offset_candidates_m) + 1.0,
        ) + self._stop_margin_boost_m

    def _publish_passthrough_path(self) -> None:
        if self._latest_pfvtr_path is None:
            return
        self._publish_base_link_path(self._latest_pfvtr_path)

    # ------------------------------------------------------------------
    # Terrain grid helper
    # ------------------------------------------------------------------

    def _build_terrain_grid(
        self, msg: PointCloud2
    ) -> Optional[Dict[Tuple[int, int], float]]:
        try:
            points = pc2.read_points(
                msg,
                field_names=("x", "y", "traversability"),
                skip_nans=False,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to read terrain_map: {exc}")
            return None

        cell = max(self.grid_cell_size_m, 1e-3)
        grid: Dict[Tuple[int, int], float] = {}
        for p in points:
            try:
                x, y, trav = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                continue
            if not math.isfinite(trav):
                continue
            ix = int(round(x / cell))
            iy = int(round(y / cell))
            key = (ix, iy)
            prev = grid.get(key)
            if prev is None or trav > prev:
                grid[key] = trav
        return grid

    def _query_traversability(
        self, x: float, y: float
    ) -> Optional[float]:
        if self._latest_terrain_grid is None:
            return None
        cell = max(self.grid_cell_size_m, 1e-3)
        ix = int(round(x / cell))
        iy = int(round(y / cell))
        return self._latest_terrain_grid.get((ix, iy))

    # ------------------------------------------------------------------
    # Main state machine tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        enabled = self._avoidance_enabled()
        if enabled != self._avoidance_was_enabled:
            if enabled:
                self._state = AvoidState.FOLLOW
                self.get_logger().info(
                    f"Obstacle avoidance ENABLED (mission={self._mission_state!r})"
                )
            else:
                self.get_logger().info(
                    f"Obstacle avoidance DISABLED (mission={self._mission_state!r})"
                )
                self._reset_avoidance_logic()
            self._avoidance_was_enabled = enabled

        if not enabled:
            self._publish_state()
            self._publish_passthrough_path()
            return

        # Publish state string for debug even if we do nothing else.
        self._publish_state()

        if self._latest_pfvtr_path is None or self._latest_terrain_grid is None:
            return

        if self._latest_odom is None:
            # FOLLOW can passthrough without odom; STOP/WAIT must still run their timers.
            if self._state == AvoidState.FOLLOW:
                self._publish_base_link_path(self._latest_pfvtr_path)
            elif self._state == AvoidState.STOP:
                self._handle_stop()
            elif self._state == AvoidState.WAIT:
                self._handle_wait()
            return

        if self._state == AvoidState.FOLLOW:
            self._handle_follow()
        elif self._state == AvoidState.STOP:
            self._handle_stop()
        elif self._state == AvoidState.WAIT:
            self._handle_wait()
        elif self._state == AvoidState.CRAB_OUT:
            self._handle_crab_out()
        elif self._state in (AvoidState.PARALLEL, AvoidState.CRAB_OUT_MORE, AvoidState.CRAB_SWITCH_SIDE):
            self._handle_parallel()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_follow(self) -> None:
        path = self._latest_pfvtr_path
        if path is None:
            return

        blocked, i_block, _ = self._find_blocked_segment(path)
        if not blocked:
            self._obstacle_first_seen = None
            self._obstacle_cleared_since = None
            self._publish_base_link_path(path)
            return

        now = self._now()
        if self._obstacle_first_seen is None:
            self._obstacle_first_seen = now
            self.get_logger().info(
                f"Obstacle detected ahead; confirming for "
                f"{self.obstacle_confirm_duration_s:.1f} s before truncating path"
            )

        elapsed_confirm_s = (now - self._obstacle_first_seen).nanoseconds * 1e-9
        if elapsed_confirm_s < self.obstacle_confirm_duration_s:
            # Keep full PFVTR path while the detection is being confirmed.
            self._publish_base_link_path(path)
            return

        truncated = self._truncate_path_before_obstacle(path, i_block)
        self._publish_base_link_path(truncated)

        self._state = AvoidState.STOP
        self._wait_start = None
        d_stop = self._path_length_m(truncated)
        d_obs = self._distance_to_pose_index(path, i_block)
        self.get_logger().info(
            f"FOLLOW -> STOP after {elapsed_confirm_s:.1f} s confirm "
            f"(obstacle at {d_obs:.2f} m, truncated end at {d_stop:.2f} m, "
            f"margin={max(0.0, d_obs - d_stop):.2f} m)"
        )

    def _handle_stop(self) -> None:
        path = self._latest_pfvtr_path
        if path is None:
            return

        if self._update_clear_latch(path):
            self._resume_follow(
                path,
                f"STOP -> FOLLOW (path clear for {self.obstacle_clear_duration_s:.1f} s)",
            )
            return

        blocked, i_block, _ = self._find_blocked_segment(path)
        if not blocked:
            # Clearing but not long enough yet — resume full path while timer runs.
            self._publish_base_link_path(path)
            return

        truncated = self._truncate_path_before_obstacle(path, i_block)
        self._publish_base_link_path(truncated)

        # Obstacle still present: leave STOP immediately and hold at the stop line in WAIT.
        if self._wait_start is None:
            self._wait_start = self._now()
        if self._state == AvoidState.STOP:
            self._state = AvoidState.WAIT
            self.get_logger().info(
                f"STOP -> WAIT (manoeuvre timer {self._wait_for_crab_s:.1f} s before CRAB_OUT)"
            )

    def _handle_wait(self) -> None:
        path = self._latest_pfvtr_path
        if path is None:
            return

        if self._update_clear_latch(path):
            self._resume_follow(
                path,
                f"WAIT -> FOLLOW (path clear for {self.obstacle_clear_duration_s:.1f} s)",
            )
            return

        blocked, i_block, _ = self._find_blocked_segment(path)
        if not blocked:
            self._publish_base_link_path(path)
            return

        if self._wait_start is None:
            self._wait_start = self._now()

        now = self._now()
        if (now - self._wait_start) >= Duration(seconds=self._wait_for_crab_s):
            self._state = AvoidState.CRAB_OUT
            self._detour_odom_path = None
            self._crab_out_start = None
            waited_s = (now - self._wait_start).nanoseconds * 1e-9
            self.get_logger().info(
                f"WAIT -> CRAB_OUT after {waited_s:.1f} s "
                f"(manoeuvre_wait={self._wait_for_crab_s:.1f} s)"
            )

        self._publish_base_link_path(self._truncate_path_before_obstacle(path, i_block))

    def _handle_crab_out(self) -> None:
        path = self._latest_pfvtr_path
        odom = self._latest_odom
        if path is None or odom is None:
            return

        blocked, i_block, i_block_end = self._find_blocked_segment(path)
        if not blocked:
            if self._update_clear_latch(path):
                self._resume_follow(
                    path,
                    f"CRAB_OUT -> FOLLOW (path clear for {self.obstacle_clear_duration_s:.1f} s)",
                )
            else:
                self._release_mux_cmd()
                self._publish_base_link_path(path)
            return

        # Phase 1: commit detour geometry (once). Tries both sides, all offsets,
        # and progressively shorter forward plan lengths if the full horizon fails.
        if self._detour_odom_path is None:
            best_detour, best_side, best_offset, best_len, tried = self._find_best_detour(
                path, odom, i_block, i_block_end,
            )

            if best_detour is None:
                self._stop_margin_boost_m = min(3.0, self._stop_margin_boost_m + 1.0)
                self._wait_for_crab_s = self.detour_retry_wait_s
                self._wait_start = None
                self._state = AvoidState.STOP
                d_obs = self._distance_to_pose_index(path, i_block)
                self.get_logger().warn(
                    f"CRAB_OUT: no traversable detour ({tried} combos tried, "
                    f"obstacle at {d_obs:.2f} m); backing stop +{self._stop_margin_boost_m:.1f} m, "
                    f"retry in {self.detour_retry_wait_s:.1f} s"
                )
                return

            self._detour_odom_path = best_detour
            self._detour_side = best_side
            self._detour_offset_m = best_offset
            self._detour_progress_m = 0.0
            self._crab_out_start = odom.stamp
            self._wait_for_crab_s = self.wait_duration_s
            self.get_logger().info(
                f"CRAB_OUT committed detour side={best_side}, offset={best_offset:.2f} m, "
                f"plan_length={best_len:.1f} m"
            )

        # Phase 2: hijack mux with crab cmd_vel until aligned on offset lane.
        if self._detour_side is not None:
            self._publish_crab_cmd(self._detour_side)
        self._publish_detour_window()

        dist_to_detour = self._distance_to_detour_m()
        elapsed_s = 0.0
        if self._crab_out_start is not None:
            elapsed_s = (odom.stamp - self._crab_out_start).nanoseconds * 1e-9

        if dist_to_detour <= self.crab_align_threshold_m or elapsed_s >= self.max_crab_out_s:
            self._release_mux_cmd()
            self._state = AvoidState.PARALLEL
            self.get_logger().info(
                f"CRAB_OUT -> PARALLEL (dist_to_detour={dist_to_detour:.2f} m, "
                f"elapsed={elapsed_s:.1f} s)"
            )

    def _handle_parallel(self) -> None:
        if self._detour_odom_path is None or self._latest_odom is None:
            if self._latest_pfvtr_path is not None:
                self._release_mux_cmd()
                self._state = AvoidState.FOLLOW
                self.get_logger().warn("PARALLEL: lost detour, reverting to FOLLOW")
                self._publish_base_link_path(self._latest_pfvtr_path)
            return

        if self._detour_segment_blocked(self._detour_odom_path):
            self.get_logger().warn("PARALLEL: detour became blocked, going back to STOP")
            self._release_mux_cmd()
            self._detour_odom_path = None
            self._crab_out_start = None
            self._wait_start = None
            self._state = AvoidState.STOP
            if self._latest_pfvtr_path is not None:
                self._handle_stop()
            return

        # MPC follows /taros/avoidance/path; crab gate outputs vy with yawrate=0.
        self._publish_detour_window()

        if self._latest_pfvtr_path is not None:
            blocked, _, _ = self._find_blocked_segment(self._latest_pfvtr_path)
            if not blocked:
                self.get_logger().info("PARALLEL -> FOLLOW (teach path clear)")
                self._release_mux_cmd()
                self._detour_odom_path = None
                self._detour_side = None
                self._crab_out_start = None
                self._state = AvoidState.FOLLOW
                self._publish_base_link_path(self._latest_pfvtr_path)

    # ------------------------------------------------------------------
    # Helper logic
    # ------------------------------------------------------------------

    def _find_blocked_segment(self, path: Path) -> Tuple[bool, int, int]:
        """Return (blocked, i_block_start, i_block_end) in PFVTR base_link frame."""
        if self._latest_terrain_grid is None:
            return False, -1, -1

        poses = path.poses
        if not poses:
            return False, -1, -1

        blocked_start = -1
        blocked_end = -1

        dist_acc = 0.0
        prev_xy: Optional[Tuple[float, float]] = None

        for i, ps in enumerate(poses):
            x = float(ps.pose.position.x)
            y = float(ps.pose.position.y)
            if prev_xy is not None:
                dx = x - prev_xy[0]
                dy = y - prev_xy[1]
                dist_acc += math.hypot(dx, dy)
            prev_xy = (x, y)
            if dist_acc > self.lookahead_check_m:
                break

            cost = self._query_traversability(x, y)
            if cost is None:
                continue
            if cost > self.obstacle_cost_threshold:
                if blocked_start < 0:
                    blocked_start = i
                blocked_end = i
            elif blocked_start >= 0:
                # Once we leave a blocked run, stop scanning.
                break

        if blocked_start < 0:
            return False, -1, -1
        if blocked_end < blocked_start:
            blocked_end = blocked_start
        return True, blocked_start, blocked_end

    @staticmethod
    def _path_cumulative_distances(poses: List[PoseStamped]) -> List[float]:
        if not poses:
            return []
        cum = [0.0]
        for i in range(1, len(poses)):
            dx = float(poses[i].pose.position.x) - float(poses[i - 1].pose.position.x)
            dy = float(poses[i].pose.position.y) - float(poses[i - 1].pose.position.y)
            cum.append(cum[-1] + math.hypot(dx, dy))
        return cum

    def _distance_to_pose_index(self, path: Path, index: int) -> float:
        cum = self._path_cumulative_distances(path.poses)
        if not cum:
            return 0.0
        idx = max(0, min(int(index), len(cum) - 1))
        return float(cum[idx])

    def _path_length_m(self, path: Path) -> float:
        cum = self._path_cumulative_distances(path.poses)
        return float(cum[-1]) if cum else 0.0

    def _stop_index_before_obstacle(self, path: Path, i_block: int) -> int:
        """Inclusive path index where MPC should stop, stop_before_obstacle_m before obstacle."""
        poses = path.poses
        if not poses:
            return 0
        if i_block < 0:
            return max(0, len(poses) - 1)

        cum = self._path_cumulative_distances(poses)
        i_block = min(i_block, len(cum) - 1)
        target_d = max(0.0, cum[i_block] - self._effective_stop_before_obstacle_m())

        stop_index = 0
        for i, d in enumerate(cum):
            if d <= target_d + 1e-9:
                stop_index = i
            else:
                break
        return stop_index

    def _truncate_path_before_obstacle(self, path: Path, i_block: int) -> Path:
        stop_index = self._stop_index_before_obstacle(path, i_block)
        truncated = Path()
        truncated.header = path.header
        truncated.poses = path.poses[: stop_index + 1]
        return truncated

    def _index_at_arc_distance_before(self, path: Path, i_ref: int, back_m: float) -> int:
        poses = path.poses
        if not poses:
            return 0
        cum = self._path_cumulative_distances(poses)
        i_ref = min(max(0, i_ref), len(cum) - 1)
        target_d = max(0.0, cum[i_ref] - max(0.0, back_m))
        idx = 0
        for i, d in enumerate(cum):
            if d <= target_d + 1e-9:
                idx = i
            else:
                break
        return idx

    def _detour_plan_length_candidates(self) -> List[float]:
        lengths: List[float] = []
        length = max(self.detour_plan_length_min_m, self.detour_plan_length_m)
        floor = max(2.0, self.detour_plan_length_min_m)
        while length >= floor - 1e-9:
            lengths.append(float(length))
            if length <= floor + 1e-9:
                break
            length = max(floor, length * 0.65)
        return lengths

    def _find_best_detour(
        self,
        path: Path,
        odom: RobotPose,
        i_block: int,
        i_block_end: int,
    ) -> Tuple[Optional[np.ndarray], Optional[int], float, float, int]:
        side_signs: List[int] = [-1, 1] if self.preferred_side == "right" else [1, -1]
        tried = 0
        for plan_len in self._detour_plan_length_candidates():
            for side in side_signs:
                for offset in self.offset_candidates_m:
                    tried += 1
                    detour = self._build_detour_odom(
                        path,
                        odom,
                        side_sign=side,
                        offset_m=offset,
                        i_block=i_block,
                        i_block_end=i_block_end,
                        max_length_m=plan_len,
                    )
                    if detour is None or len(detour) < 2:
                        continue
                    if self._detour_segment_blocked(detour, max_check_length_m=plan_len):
                        continue
                    return detour, side, float(offset), plan_len, tried
        return None, None, 0.0, 0.0, tried

    def _build_detour_odom(
        self,
        path: Path,
        odom: RobotPose,
        side_sign: int,
        offset_m: float,
        i_block: int,
        i_block_end: int,
        max_length_m: float,
    ) -> Optional[np.ndarray]:
        """Build a simple odom-frame detour path by laterally offsetting PFVTR in base_link."""
        poses = path.poses
        if not poses:
            return None

        # Begin offset lane before the obstacle so crab has room to start.
        start_index = self._index_at_arc_distance_before(
            path, i_block, self.crab_start_distance_m,
        )

        detour_points: List[Tuple[float, float, float]] = []
        dist_from_start = 0.0
        prev_xy_b: Optional[Tuple[float, float]] = None

        for i in range(start_index, len(poses)):
            ps = poses[i]
            x_b = float(ps.pose.position.x)
            y_b = float(ps.pose.position.y)

            if prev_xy_b is not None:
                dist_from_start += math.hypot(x_b - prev_xy_b[0], y_b - prev_xy_b[1])
            prev_xy_b = (x_b, y_b)
            if dist_from_start > max_length_m:
                break

            # Estimate path heading from neighbor (or from orientation if available).
            if i < len(poses) - 1:
                nxt = poses[i + 1].pose.position
                dx = float(nxt.x) - x_b
                dy = float(nxt.y) - y_b
                if dx == 0.0 and dy == 0.0:
                    yaw_b = 0.0
                else:
                    yaw_b = math.atan2(dy, dx)
            else:
                yaw_b = 0.0

            # Lateral offset in base_link (x forward, y left): right=-y, left=+y.
            side = float(side_sign)
            off = float(offset_m)
            x_off = x_b
            y_off = y_b + side * off

            # Transform base_link -> odom using current odom pose.
            cos_r = math.cos(odom.yaw)
            sin_r = math.sin(odom.yaw)
            x_o = odom.x + cos_r * x_off - sin_r * y_off
            y_o = odom.y + sin_r * x_off + cos_r * y_off
            yaw_o = yaw_b + odom.yaw
            detour_points.append((x_o, y_o, yaw_o))

        if len(detour_points) < 2:
            return None
        return np.asarray(detour_points, dtype=float)

    def _detour_segment_blocked(
        self, detour_odom: np.ndarray, max_check_length_m: Optional[float] = None,
    ) -> bool:
        """Check whether any point along odom detour segment is blocked in base_link."""
        if self._latest_odom is None or self._latest_terrain_grid is None:
            return False

        check_len = max_check_length_m if max_check_length_m is not None else self.lookahead_check_m

        cos_r = math.cos(-self._latest_odom.yaw)
        sin_r = math.sin(-self._latest_odom.yaw)
        ox = self._latest_odom.x
        oy = self._latest_odom.y

        dist_acc = 0.0
        prev_xy: Optional[Tuple[float, float]] = None
        checked = 0

        for i in range(detour_odom.shape[0]):
            x_o, y_o, _ = detour_odom[i]
            dx = x_o - ox
            dy = y_o - oy
            x_b = cos_r * dx - sin_r * dy
            y_b = sin_r * dx + cos_r * dy

            if x_b < self.detour_check_min_forward_m:
                continue

            if prev_xy is not None:
                dist_acc += math.hypot(x_b - prev_xy[0], y_b - prev_xy[1])
            prev_xy = (x_b, y_b)
            if dist_acc > check_len:
                break

            checked += 1
            cost = self._query_traversability(x_b, y_b)
            if cost is not None and cost > self.obstacle_cost_threshold:
                return True

        return checked < 1

    def _distance_to_detour_m(self) -> float:
        """Min distance from robot (base_link origin) to committed detour path."""
        if self._detour_odom_path is None or self._latest_odom is None:
            return float("inf")

        cos_r = math.cos(-self._latest_odom.yaw)
        sin_r = math.sin(-self._latest_odom.yaw)
        ox = self._latest_odom.x
        oy = self._latest_odom.y

        min_d = float("inf")
        for i in range(self._detour_odom_path.shape[0]):
            x_o, y_o, _ = self._detour_odom_path[i]
            dx = x_o - ox
            dy = y_o - oy
            x_b = cos_r * dx - sin_r * dy
            y_b = sin_r * dx + cos_r * dy
            min_d = min(min_d, math.hypot(x_b, y_b))
        return float(min_d)

    def _publish_path(self, path: Path) -> None:
        try:
            self._path_pub.publish(path)
        except Exception as exc:
            self.get_logger().error(f"Failed to publish avoidance path: {exc}")

    def _publish_base_link_path(self, path_robot: Path) -> None:
        """Publish a path already expressed in the robot/base_link frame."""
        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.output_frame_id
        for p in path_robot.poses:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose = p.pose
            out.poses.append(ps)
        self._publish_path(out)

    def _odom_xyyaw_to_base_link(
        self, x_o: float, y_o: float, yaw_o: float, odom: RobotPose
    ) -> Tuple[float, float, float]:
        cos_r = math.cos(-odom.yaw)
        sin_r = math.sin(-odom.yaw)
        dx = x_o - odom.x
        dy = y_o - odom.y
        x_b = cos_r * dx - sin_r * dy
        y_b = sin_r * dx + cos_r * dy
        yaw_b = yaw_o - odom.yaw
        return x_b, y_b, yaw_b

    def _publish_crab_cmd(self, side_sign: int) -> None:
        """Publish crab TwistStamped to /cmd_vel_key (twist_mux priority 30).

        side_sign matches detour offset: -1 = right (-y), +1 = left (+y) in base_link.
        Positive vy strafes left (ROS convention); negative vy strafes right.
        """
        vx = self.crab_vx_mps
        vy = float(side_sign) * vx * self.crab_vy_over_vx

        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = "base_link"
        ts.twist.linear.x = float(vx)
        ts.twist.linear.y = float(vy)
        ts.twist.angular.z = 0.0
        try:
            self._crab_cmd_vel_pub.publish(ts)
            self._mux_hijack_active = True
        except Exception as exc:
            self.get_logger().error(f"Failed to publish crab cmd_vel: {exc}")

        mode_msg = UInt8()
        mode_msg.data = 2  # MODE_CRAB (debug only)
        try:
            self._debug_mode_pub.publish(mode_msg)
        except Exception:
            pass

    def _release_mux_cmd(self) -> None:
        """Stop publishing to /cmd_vel_key so twist_mux falls back to MPC (autonomy_low)."""
        if not self._mux_hijack_active:
            return
        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = "base_link"
        ts.twist.linear.x = 0.0
        ts.twist.linear.y = 0.0
        ts.twist.angular.z = 0.0
        try:
            self._crab_cmd_vel_pub.publish(ts)
        except Exception as exc:
            self.get_logger().error(f"Failed to release mux cmd_vel: {exc}")
        self._mux_hijack_active = False

    def _publish_state(self) -> None:
        msg = String()
        msg.data = str(self._state.value)
        try:
            self._state_pub.publish(msg)
        except Exception:
            pass

    def _publish_detour_window(self, window_length_m: float = 20.0) -> None:
        if self._detour_odom_path is None or self._latest_odom is None:
            return

        # Simple window: start from closest detour point to current odom position.
        dx = self._detour_odom_path[:, 0] - self._latest_odom.x
        dy = self._detour_odom_path[:, 1] - self._latest_odom.y
        d2 = dx * dx + dy * dy
        i0 = int(np.argmin(d2))

        pts: List[Tuple[float, float, float]] = []
        dist_acc = 0.0
        prev_xy: Optional[Tuple[float, float]] = None
        for i in range(i0, self._detour_odom_path.shape[0]):
            x_o, y_o, yaw_o = self._detour_odom_path[i]
            if prev_xy is not None:
                dist_acc += math.hypot(x_o - prev_xy[0], y_o - prev_xy[1])
            prev_xy = (x_o, y_o)
            pts.append((x_o, y_o, yaw_o))
            if dist_acc >= window_length_m:
                break

        if not pts:
            return

        odom = self._latest_odom
        out = Path()
        out.header.frame_id = self.output_frame_id
        now = self.get_clock().now().to_msg()
        out.header.stamp = now
        for x_o, y_o, yaw_o in pts:
            x_b, y_b, yaw_b = self._odom_xyyaw_to_base_link(x_o, y_o, yaw_o, odom)
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x_b)
            ps.pose.position.y = float(y_b)
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(yaw_b * 0.5)
            ps.pose.orientation.w = math.cos(yaw_b * 0.5)
            out.poses.append(ps)

        self._publish_path(out)

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        # Standard yaw extraction for planar robots.
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathOffsetAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

