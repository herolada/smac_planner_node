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

"""Follow a path by emitting ``cmd_vel`` from a simple PID controller.

The controller looks up the robot pose via TF, picks a lookahead point a fixed
distance ahead along the path, and drives toward it. A PID on the heading error
to that lookahead point produces the angular velocity; the linear velocity is a
capped forward command, reduced when the heading error is large or the goal is
near.

Two input modes, selected by the ``input_mode`` parameter:

``topic``
    Subscribe to a ``nav_msgs/Path`` on ``plan`` and run a free-running timer at
    ``control_frequency``. Simplest to drive by hand or from RViz.

``action``
    Serve ``nav2_msgs/action/FollowPath``. The control loop runs inside the goal
    execution, publishes ``distance_to_goal``/``speed`` feedback every cycle, and
    honours cancel and preemption by a newer goal. Mirrors how a Nav2 controller
    server behaves, so it can be driven by a Nav2 behaviour tree.
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
from geometry_msgs.msg import Twist
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path


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


class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # --- Parameters ---------------------------------------------------
        self._control_hz = self.declare_parameter('control_frequency', 20.0).value
        self._lookahead = self.declare_parameter('lookahead_distance', 0.6).value
        self._max_linear = self.declare_parameter('max_linear_velocity', 1.0).value
        self._max_angular = self.declare_parameter('max_angular_velocity', 1.5).value
        self._goal_tolerance = self.declare_parameter('goal_tolerance', 0.15).value
        # PID gains acting on the heading error to the lookahead point.
        self._kp = self.declare_parameter('kp', 1.5).value
        self._ki = self.declare_parameter('ki', 0.0).value
        self._kd = self.declare_parameter('kd', 0.1).value
        # Frame the robot's base lives in (TF lookup target).
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._cmd_topic = self.declare_parameter('cmd_vel_topic', 'cmd_vel').value
        # 'topic' -> subscribe to plan_topic; 'action' -> serve FollowPath.
        self._input_mode = self.declare_parameter('input_mode', 'topic').value
        self._plan_topic = self.declare_parameter('plan_topic', 'plan').value
        self._action_name = self.declare_parameter('action_name', 'follow_path').value
        # How long TF may fail continuously before a goal is aborted (action mode).
        self._tf_timeout = self.declare_parameter('tf_failure_timeout', 1.0).value

        if self._input_mode not in ('topic', 'action'):
            raise ValueError(
                f"input_mode must be 'topic' or 'action', got '{self._input_mode}'")

        # --- Controller state ---------------------------------------------
        self._integral = 0.0         # PID integral accumulator
        self._prev_error = None      # PID previous error
        self._prev_time = None       # timestamp of previous tick
        self._current_speed = 0.0    # last commanded linear velocity (feedback)

        # --- TF -----------------------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 1)

        if self._input_mode == 'topic':
            self._setup_topic_mode()
        else:
            self._setup_action_mode()

    # -- mode setup --------------------------------------------------------
    def _setup_topic_mode(self):
        self._path = None
        self._plan_sub = self.create_subscription(
            Path, self._plan_topic, self._on_path, 1)
        self._timer = self.create_timer(1.0 / self._control_hz, self._timer_step)
        self.get_logger().info(
            f"path_follower [topic mode]: '{self._plan_topic}' -> '{self._cmd_topic}' "
            f"at {self._control_hz:.0f} Hz, lookahead={self._lookahead:.2f} m, "
            f"max_linear={self._max_linear:.2f} m/s.")

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
            f"path_follower [action mode]: serving '{self._action_name}' -> "
            f"'{self._cmd_topic}' at {self._control_hz:.0f} Hz, "
            f"lookahead={self._lookahead:.2f} m, max_linear={self._max_linear:.2f} m/s.")

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

    def _find_lookahead(self, path, rx, ry):
        """Pick a point ~lookahead distance ahead along ``path`` from (rx, ry).

        Finds the path point nearest the robot, then walks forward until the
        lookahead distance is reached (or the end of the path), so the controller
        progresses along the path rather than cutting back toward the start.
        """
        poses = path.poses

        nearest_i = 0
        nearest_d2 = float('inf')
        for i, ps in enumerate(poses):
            dx = ps.pose.position.x - rx
            dy = ps.pose.position.y - ry
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_i = i

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

    def _compute_command(self, path, pose):
        """Run one PID step. Returns (Twist, distance_to_goal, goal_reached)."""
        rx, ry, ryaw = pose
        px, py, dist_to_goal = self._find_lookahead(path, rx, ry)

        if dist_to_goal <= self._goal_tolerance:
            return Twist(), dist_to_goal, True

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
        linear = self._max_linear * max(0.0, math.cos(error))
        linear = min(linear, self._max_linear * (dist_to_goal / self._lookahead))
        linear = max(0.0, min(self._max_linear, linear))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd, dist_to_goal, False

    def _publish_cmd(self, cmd):
        self._current_speed = cmd.linear.x
        self._cmd_pub.publish(cmd)

    def _publish_stop(self):
        self._publish_cmd(Twist())

    # -- topic mode --------------------------------------------------------
    def _on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warning('Received empty path; stopping.')
            self._path = None
            return
        self._path = msg
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

        period = 1.0 / self._control_hz
        tf_fail_start = None
        self._reset_pid()

        try:
            if not path.poses:
                raise InvalidPath('Received goal with empty path.')

            while rclpy.ok():
                cycle_start = time.monotonic()

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
    node = PathFollower()
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
