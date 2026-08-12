#!/usr/bin/env python3
# Copyright (c) 2026 Adam Herold
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental variant of ``path_follower.py`` with curvature-aware speed and
a speed-scaled lookahead, both tunable at runtime.

Differences from ``path_follower.py``:

- The path's curvature over the next ``curvature_lookahead_distance`` metres
  (measured along consecutive path points, from the point nearest the robot)
  is estimated and normalized to ``[0, 1]`` against ``max_curvature``. The
  commanded speed ceiling is linearly interpolated between
  ``max_linear_velocity`` (straight path) and ``min_linear_velocity`` (path at
  or above ``max_curvature``), i.e. ``ceiling = min + (max - min) * (1 -
  curvature)``.
- The lookahead distance is no longer a fixed parameter. Instead
  ``lookahead_time`` (seconds) is multiplied by the last commanded linear
  velocity to get a speed-proportional lookahead distance, floored at
  ``min_lookahead_distance`` so it never collapses to zero while slow/stopped.
- All tuning parameters (speed/lookahead/curvature/PID/timing) can be changed
  live via ``ros2 param set`` while the node is running; wiring-time
  parameters (topics, frames, input mode) still require a restart.

See ``path_follower.py`` for the base controller/architecture description.
"""

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transform support on tf_buffer
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult


def yaw_from_quaternion(q):
    """Extract the yaw (Z rotation) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ControllerTFError(Exception):
    """The robot pose could not be resolved for longer than the TF timeout."""


class InvalidPath(Exception):
    """The requested path is empty or otherwise unusable."""


class PathFollowerDynamic(Node):
    def __init__(self):
        super().__init__('path_follower_dynamic')

        # --- Parameters -----------------------------------------------------
        # Wiring-time parameters: fixed for the node's lifetime (changing them
        # would require re-creating publishers/subscriptions/action servers).
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._map_frame = self.declare_parameter('map_frame', 'odom_odin').value
        self._cmd_topic = self.declare_parameter('cmd_vel_topic', 'cmd_vel').value
        self._lookahead_topic = self.declare_parameter('lookahead_topic', 'lookahead_pose').value
        self._input_mode = self.declare_parameter('input_mode', 'action').value
        self._plan_topic = self.declare_parameter('plan_topic', 'plan').value
        self._action_name = self.declare_parameter('action_name', 'follow_path').value
        self._static_param_names = {
            'base_frame', 'map_frame', 'cmd_vel_topic', 'lookahead_topic',
            'input_mode', 'plan_topic', 'action_name',
        }

        # Tuning parameters: all adjustable at runtime, see _on_set_parameters.
        self._control_hz = self.declare_parameter('control_frequency', 20.0).value
        # Lookahead is speed-proportional: distance = lookahead_time * current
        # linear speed, floored at min_lookahead_distance.
        self._lookahead_time = self.declare_parameter('lookahead_time', 1.0).value
        self._min_lookahead_distance = self.declare_parameter('min_lookahead_distance', 0.3).value
        self._lookahead = self._min_lookahead_distance  # current effective lookahead distance
        # Speed ceiling at zero curvature / floor at max_curvature.
        self._max_linear_velocity = self.declare_parameter('max_linear_velocity', 1.0).value
        self._min_linear_velocity = self.declare_parameter('min_linear_velocity', 0.1).value
        self._max_angular = self.declare_parameter('max_angular_velocity', 1.5).value
        self._goal_tolerance = self.declare_parameter('goal_tolerance', 0.15).value
        # PID gains acting on the heading error to the lookahead point.
        self._kp = self.declare_parameter('kp', 1.5).value
        self._ki = self.declare_parameter('ki', 0.0).value
        self._kd = self.declare_parameter('kd', 0.1).value
        # Curvature is averaged over the next curvature_lookahead_distance
        # metres of path (from the point nearest the robot) and normalized by
        # max_curvature (rad/m) into [0, 1] (clamped).
        self._curvature_lookahead_distance = self.declare_parameter(
            'curvature_lookahead_distance', 2.0).value
        self._max_curvature = self.declare_parameter('max_curvature', 1.0).value
        # How long TF may fail continuously before a goal is aborted (action mode).
        self._tf_timeout = self.declare_parameter('tf_failure_timeout', 1.0).value

        # Maps dynamic parameter name -> attribute name it controls.
        self._dynamic_param_attrs = {
            'control_frequency': '_control_hz',
            'lookahead_time': '_lookahead_time',
            'min_lookahead_distance': '_min_lookahead_distance',
            'max_linear_velocity': '_max_linear_velocity',
            'min_linear_velocity': '_min_linear_velocity',
            'max_angular_velocity': '_max_angular',
            'goal_tolerance': '_goal_tolerance',
            'kp': '_kp',
            'ki': '_ki',
            'kd': '_kd',
            'curvature_lookahead_distance': '_curvature_lookahead_distance',
            'max_curvature': '_max_curvature',
            'tf_failure_timeout': '_tf_timeout',
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

        if self._input_mode not in ('topic', 'action'):
            raise ValueError(
                f"input_mode must be 'topic' or 'action', got '{self._input_mode}'")

        # --- Controller state -------------------------------------------------
        self._integral = 0.0         # PID integral accumulator
        self._prev_error = None      # PID previous error
        self._prev_time = None       # timestamp of previous tick
        self._current_speed = 0.0    # last commanded linear velocity (feedback)

        # --- TF -----------------------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(TwistStamped, self._cmd_topic, 1)
        self._lookahead_pub = self.create_publisher(PoseStamped, self._lookahead_topic, 1)

        if self._input_mode == 'topic':
            self._setup_topic_mode()
        else:
            self._setup_action_mode()

    # -- dynamic reconfiguration --------------------------------------------
    def _on_set_parameters(self, params):
        """Validate and apply runtime parameter changes.

        Wiring-time parameters (topics/frames/input mode) are rejected since
        changing them wouldn't actually rewire publishers/subscriptions; every
        other declared parameter can be changed live.
        """
        pending = {}
        for p in params:
            if p.name in self._static_param_names:
                return SetParametersResult(
                    successful=False,
                    reason=f"'{p.name}' cannot be changed at runtime "
                           f"(fixed at node startup); restart the node instead.")
            if p.name not in self._dynamic_param_attrs:
                continue
            try:
                pending[p.name] = float(p.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f"'{p.name}' must be numeric.")

        def value(name, current):
            return pending.get(name, current)

        if value('control_frequency', self._control_hz) <= 0.0:
            return SetParametersResult(successful=False, reason='control_frequency must be > 0.')
        if value('lookahead_time', self._lookahead_time) < 0.0:
            return SetParametersResult(successful=False, reason='lookahead_time must be >= 0.')
        if value('min_lookahead_distance', self._min_lookahead_distance) < 0.0:
            return SetParametersResult(
                successful=False, reason='min_lookahead_distance must be >= 0.')
        min_lin = value('min_linear_velocity', self._min_linear_velocity)
        max_lin = value('max_linear_velocity', self._max_linear_velocity)
        if min_lin < 0.0 or max_lin < 0.0:
            return SetParametersResult(
                successful=False, reason='linear velocities must be >= 0.')
        if min_lin > max_lin:
            return SetParametersResult(
                successful=False,
                reason='min_linear_velocity must be <= max_linear_velocity.')
        if value('max_angular_velocity', self._max_angular) <= 0.0:
            return SetParametersResult(
                successful=False, reason='max_angular_velocity must be > 0.')
        if value('goal_tolerance', self._goal_tolerance) < 0.0:
            return SetParametersResult(successful=False, reason='goal_tolerance must be >= 0.')
        if value('curvature_lookahead_distance', self._curvature_lookahead_distance) <= 0.0:
            return SetParametersResult(
                successful=False, reason='curvature_lookahead_distance must be > 0.')
        if value('max_curvature', self._max_curvature) <= 0.0:
            return SetParametersResult(successful=False, reason='max_curvature must be > 0.')
        if value('tf_failure_timeout', self._tf_timeout) <= 0.0:
            return SetParametersResult(
                successful=False, reason='tf_failure_timeout must be > 0.')

        control_hz_changed = 'control_frequency' in pending
        for name, val in pending.items():
            setattr(self, self._dynamic_param_attrs[name], val)

        if control_hz_changed and self._input_mode == 'topic':
            self._timer.cancel()
            self._timer = self.create_timer(1.0 / self._control_hz, self._timer_step)

        return SetParametersResult(successful=True)

    # -- mode setup --------------------------------------------------------
    def _setup_topic_mode(self):
        self._path = None
        self._plan_sub = self.create_subscription(
            Path, self._plan_topic, self._on_path, 1)
        self._timer = self.create_timer(1.0 / self._control_hz, self._timer_step)
        self.get_logger().info(
            f"path_follower_dynamic [topic mode]: '{self._plan_topic}' -> '{self._cmd_topic}' "
            f"at {self._control_hz:.0f} Hz, lookahead_time={self._lookahead_time:.2f} s "
            f"(min {self._min_lookahead_distance:.2f} m), "
            f"max_linear={self._max_linear_velocity:.2f} m/s.")

    def _setup_action_mode(self):
        # Serialises goal execution: a preempting goal waits here until the
        # outgoing loop has exited, so only one control loop ever runs.
        self._exec_lock = threading.Lock()
        self._goal_lock = threading.Lock()
        self._active_goal = None
        self._preempt_requested = False

        # Reentrant group + MultiThreadedExecutor so a new goal can be accepted
        # while the current goal's blocking control loop is still running.
        self._action_server = ActionServer(
            self,
            FollowPath,
            self._action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            handle_accepted_callback=self._handle_accepted_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup())

        self.get_logger().info(
            f"path_follower_dynamic [action mode]: serving '{self._action_name}' -> "
            f"'{self._cmd_topic}' at {self._control_hz:.0f} Hz, "
            f"lookahead_time={self._lookahead_time:.2f} s "
            f"(min {self._min_lookahead_distance:.2f} m), "
            f"max_linear={self._max_linear_velocity:.2f} m/s.")

    # -- shared controller core -------------------------------------------
    def _reset_pid(self):
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def _robot_pose(self, frame):
        """Return (x, y, yaw) of the base frame in ``frame``, or None on failure."""
        try:
            tf = self._tf_buffer.lookup_transform(
                frame, self._base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except tf2_ros.TransformException as ex:
            self.get_logger().warning(
                f"TF {frame} <- {self._base_frame} unavailable: {ex}",
                throttle_duration_sec=2.0)
            return None
        t = tf.transform.translation
        return (t.x, t.y, yaw_from_quaternion(tf.transform.rotation))

    def _transform_path(self, path):
        """Transform ``path`` into ``self._map_frame`` once, up front.

        A single TF lookup is applied to every pose rather than re-resolving the
        path's frame on every control cycle, since the path is otherwise static
        data: if it arrived in a moving frame (e.g. ``base_link``), comparing a
        live robot pose against it every cycle would never show any progress.
        """
        if not path.poses:
            return path
        source_frame = path.header.frame_id
        if source_frame == self._map_frame:
            return path
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, source_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except tf2_ros.TransformException as ex:
            raise InvalidPath(
                f"Could not transform path from '{source_frame}' to "
                f"'{self._map_frame}': {ex}")

        new_path = Path()
        new_path.header.stamp = path.header.stamp
        new_path.header.frame_id = self._map_frame
        new_path.poses = [
            tf2_geometry_msgs.do_transform_pose_stamped(ps, tf) for ps in path.poses
        ]
        return new_path

    def _nearest_index(self, poses, rx, ry):
        """Index of the path pose closest to (rx, ry)."""
        nearest_i = 0
        nearest_d2 = float('inf')
        for i, ps in enumerate(poses):
            dx = ps.pose.position.x - rx
            dy = ps.pose.position.y - ry
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_i = i
        return nearest_i

    def _find_lookahead(self, path, rx, ry, nearest_i):
        """Pick a point ~lookahead distance ahead along ``path`` from (rx, ry).

        Walks forward from the nearest point until the lookahead distance is
        reached (or the end of the path), so the controller progresses along
        the path rather than cutting back toward the start.
        """
        poses = path.poses

        target = poses[-1]
        for i in range(nearest_i, len(poses)):
            dx = poses[i].pose.position.x - rx
            dy = poses[i].pose.position.y - ry
            if math.hypot(dx, dy) >= self._lookahead:
                target = poses[i]
                break

        goal = poses[-1].pose.position
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry)
        return target.pose.position.x, target.pose.position.y, dist_to_goal

    def _path_curvature(self, path, nearest_i):
        """Normalized curvature (0-1) of the path ahead of ``nearest_i``.

        Walks forward accumulating arc length until
        ``curvature_lookahead_distance`` is covered (or the path ends), sums
        the absolute heading change between consecutive segments, and divides
        by the arc length walked to get an average curvature in rad/m. That
        is normalized by ``max_curvature`` and clamped to [0, 1].
        """
        poses = path.poses
        points = [poses[nearest_i].pose.position]
        cum_dist = 0.0
        i = nearest_i
        while i < len(poses) - 1 and cum_dist < self._curvature_lookahead_distance:
            p0 = poses[i].pose.position
            p1 = poses[i + 1].pose.position
            cum_dist += math.hypot(p1.x - p0.x, p1.y - p0.y)
            points.append(p1)
            i += 1

        if len(points) < 3:
            return 0.0

        total_turn = 0.0
        arc_len = 0.0
        prev_heading = None
        for j in range(1, len(points)):
            ax = points[j].x - points[j - 1].x
            ay = points[j].y - points[j - 1].y
            seg_len = math.hypot(ax, ay)
            if seg_len < 1e-9:
                continue
            heading = math.atan2(ay, ax)
            if prev_heading is not None:
                total_turn += abs(normalize_angle(heading - prev_heading))
            prev_heading = heading
            arc_len += seg_len

        if arc_len < 1e-6:
            return 0.0

        avg_curvature = total_turn / arc_len  # rad/m
        return max(0.0, min(1.0, avg_curvature / self._max_curvature))

    def _compute_command(self, path, pose):
        """Run one PID step. Returns (TwistStamped, distance_to_goal, goal_reached)."""
        rx, ry, ryaw = pose
        nearest_i = self._nearest_index(path.poses, rx, ry)

        # Speed-proportional lookahead, floored so it never collapses to zero.
        self._lookahead = max(
            self._min_lookahead_distance, self._lookahead_time * self._current_speed)

        # Curvature-scaled speed ceiling: full max_linear_velocity on a
        # straight path, down to min_linear_velocity as curvature saturates.
        curvature = self._path_curvature(path, nearest_i)
        max_linear = (self._min_linear_velocity
                      + (self._max_linear_velocity - self._min_linear_velocity)
                      * (1.0 - curvature))

        px, py, dist_to_goal = self._find_lookahead(path, rx, ry, nearest_i)
        self._publish_lookahead(path.header.frame_id, px, py)

        if dist_to_goal <= self._goal_tolerance:
            return TwistStamped(), dist_to_goal, True

        # Heading error toward the lookahead point, in the robot frame.
        desired_yaw = math.atan2(py - ry, px - rx)
        error = normalize_angle(desired_yaw - ryaw)

        now = self.get_clock().now()
        dt = 1.0 / self._control_hz
        if self._prev_time is not None:
            dt = max((now - self._prev_time).nanoseconds * 1e-9, 1e-6)

        self._integral += error * dt
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt

        angular = (self._kp * error
                   + self._ki * self._integral
                   + self._kd * derivative)
        angular = max(-self._max_angular, min(self._max_angular, angular))

        self._prev_error = error
        self._prev_time = now

        # Full speed when aligned, reduced for large heading error, ramped down
        # as the goal is approached.
        linear = max_linear * max(0.0, math.cos(error))
        linear = min(linear, max_linear * (dist_to_goal / self._lookahead))
        linear = max(0.0, min(max_linear, linear))

        cmd = TwistStamped()
        cmd.header.frame_id = "base_link"
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = linear
        cmd.twist.angular.z = angular
        return cmd, dist_to_goal, False

    def _publish_cmd(self, cmd):
        self._current_speed = cmd.twist.linear.x
        self._cmd_pub.publish(cmd)

    def _publish_stop(self):
        self._publish_cmd(TwistStamped())

    def _publish_lookahead(self, frame_id, x, y):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        self._lookahead_pub.publish(pose)

    # -- topic mode --------------------------------------------------------
    def _on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warning('Received empty path; stopping.')
            self._path = None
            return
        try:
            self._path = self._transform_path(msg)
        except InvalidPath as ex:
            self.get_logger().error(str(ex))
            self._path = None
            return
        self._reset_pid()

    def _timer_step(self):
        if self._path is None:
            return

        pose = self._robot_pose(self._path.header.frame_id)
        if pose is None:
            self._publish_stop()
            return

        cmd, _, reached = self._compute_command(self._path, pose)
        self._publish_cmd(cmd)
        if reached:
            self.get_logger().info('Goal reached; stopping.')
            self._path = None

    # -- action mode -------------------------------------------------------
    def _goal_callback(self, goal_request):
        # Empty paths are accepted so the client gets an INVALID_PATH result
        # rather than a bare rejection.
        return GoalResponse.ACCEPT

    def _handle_accepted_callback(self, goal_handle):
        with self._goal_lock:
            if self._active_goal is not None and self._active_goal.is_active:
                self.get_logger().info('New goal preempting current.')
                self._preempt_requested = True
        # Non-blocking: schedules _execute_callback on the executor. It will
        # block on _exec_lock until the preempted loop has exited.
        goal_handle.execute()

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        with self._exec_lock:
            with self._goal_lock:
                self._active_goal = goal_handle
                self._preempt_requested = False
            try:
                return self._follow_path(goal_handle)
            finally:
                with self._goal_lock:
                    if self._active_goal is goal_handle:
                        self._active_goal = None

    def _follow_path(self, goal_handle):
        result = FollowPath.Result()
        path = goal_handle.request.path
        self.get_logger().info(
            f'Received a goal with {len(path.poses)} poses, begin computing control.')

        tf_fail_start = None
        self._reset_pid()

        try:
            if not path.poses:
                raise InvalidPath('Received goal with empty path.')
            path = self._transform_path(path)

            while rclpy.ok():
                cycle_start = time.monotonic()
                period = 1.0 / self._control_hz  # re-read: control_frequency may change live

                if not goal_handle.is_active:
                    self.get_logger().debug('Goal no longer active. Stopping.')
                    self._publish_stop()
                    return result

                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Cancel requested. Stopping the robot.')
                    self._publish_stop()
                    goal_handle.canceled()
                    result.error_msg = 'Goal canceled.'
                    return result

                with self._goal_lock:
                    preempted = self._preempt_requested
                if preempted:
                    # Don't stop: the incoming goal takes over immediately.
                    goal_handle.abort()
                    result.error_msg = 'Preempted by a new goal.'
                    return result

                pose = self._robot_pose(path.header.frame_id)
                if pose is None:
                    self._publish_stop()
                    if tf_fail_start is None:
                        tf_fail_start = cycle_start
                    elif cycle_start - tf_fail_start > self._tf_timeout:
                        raise ControllerTFError(
                            f'Could not resolve {path.header.frame_id} <- '
                            f'{self._base_frame} for {self._tf_timeout:.1f}s.')
                    dist_to_goal = float('nan')
                    reached = False
                else:
                    tf_fail_start = None
                    cmd, dist_to_goal, reached = self._compute_command(path, pose)
                    self._publish_cmd(cmd)

                feedback = FollowPath.Feedback()
                feedback.distance_to_goal = float(dist_to_goal)
                feedback.speed = float(self._current_speed)
                goal_handle.publish_feedback(feedback)

                if reached:
                    self.get_logger().info('Reached the goal!')
                    self._publish_stop()
                    goal_handle.succeed()
                    return result

                elapsed = time.monotonic() - cycle_start
                if elapsed > period:
                    self.get_logger().warning(
                        f'Control loop missed its desired rate of {self._control_hz:.2f} Hz. '
                        f'Current loop rate is {1.0 / elapsed:.2f} Hz.')
                else:
                    time.sleep(period - elapsed)

            # rclpy shutting down.
            self._publish_stop()
            return result

        except InvalidPath as ex:
            return self._abort(goal_handle, result,
                               FollowPath.Result.INVALID_PATH, str(ex))
        except ControllerTFError as ex:
            return self._abort(goal_handle, result,
                               FollowPath.Result.TF_ERROR, str(ex))
        except Exception as ex:  # noqa: BLE001 - surface anything else as UNKNOWN
            return self._abort(goal_handle, result,
                               FollowPath.Result.UNKNOWN, repr(ex))

    def _abort(self, goal_handle, result, error_code, message):
        self.get_logger().error(message)
        self._publish_stop()
        result.error_code = error_code
        result.error_msg = message
        if goal_handle.is_active:
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerDynamic()
    # MultiThreadedExecutor is required in action mode so a preempting goal can
    # be handled while the current goal's control loop blocks; harmless in
    # topic mode.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
