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

"""Follow a ``plan`` path by emitting ``cmd_vel`` with a simple PID controller.

Subscribes to a ``nav_msgs/Path`` on ``plan`` and, at a fixed rate, looks up the
robot pose via TF, picks a lookahead point a fixed distance ahead along the path,
and drives toward it. A PID controller on the heading error to that lookahead
point produces the angular velocity; the linear velocity is a capped forward
command that is reduced when the heading error is large or the goal is near.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path


def yaw_from_quaternion(q):
    """Extract the yaw (Z rotation) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


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
        self._plan_topic = self.declare_parameter('plan_topic', 'plan').value

        # --- State --------------------------------------------------------
        self._path = None            # latest nav_msgs/Path
        self._integral = 0.0         # PID integral accumulator
        self._prev_error = None      # PID previous error
        self._prev_time = None       # timestamp of previous tick

        # --- TF -----------------------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- ROS interfaces ----------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 1)
        self._plan_sub = self.create_subscription(
            Path, self._plan_topic, self._on_path, 1)

        period = 1.0 / self._control_hz
        self._timer = self.create_timer(period, self._control_step)

        self.get_logger().info(
            f"path_follower up: following '{self._plan_topic}' -> '{self._cmd_topic}' "
            f"at {self._control_hz:.0f} Hz, lookahead={self._lookahead:.2f} m, "
            f"max_linear={self._max_linear:.2f} m/s.")

    # ---------------------------------------------------------------------
    def _on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warning('Received empty path; stopping.')
            self._path = None
            return
        self._path = msg
        # Reset PID state for a fresh path.
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def _robot_pose_in_path_frame(self):
        """Return (x, y, yaw) of the base frame in the path's frame, or None."""
        if self._path is None:
            return None
        frame = self._path.header.frame_id
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
        yaw = yaw_from_quaternion(tf.transform.rotation)
        return (t.x, t.y, yaw)

    def _find_lookahead(self, rx, ry):
        """Pick a point ~lookahead distance ahead along the path from (rx, ry).

        Returns (px, py, dist_to_goal). Finds the path point nearest the robot,
        then walks forward until the lookahead distance is reached (or end of
        path), so the controller progresses along the path rather than cutting
        corners back to the start.
        """
        poses = self._path.poses

        # Nearest path index to the robot.
        nearest_i = 0
        nearest_d2 = float('inf')
        for i, ps in enumerate(poses):
            dx = ps.pose.position.x - rx
            dy = ps.pose.position.y - ry
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_i = i

        # Walk forward from the nearest point to the lookahead distance.
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

    def _control_step(self):
        if self._path is None:
            return

        pose = self._robot_pose_in_path_frame()
        if pose is None:
            self._publish_stop()
            return
        rx, ry, ryaw = pose

        px, py, dist_to_goal = self._find_lookahead(rx, ry)

        # Reached the goal: stop and clear the path.
        if dist_to_goal <= self._goal_tolerance:
            self.get_logger().info('Goal reached; stopping.')
            self._publish_stop()
            self._path = None
            return

        # Heading error toward the lookahead point, in the robot frame.
        desired_yaw = math.atan2(py - ry, px - rx)
        error = normalize_angle(desired_yaw - ryaw)

        # --- PID on heading error ---
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

        # Linear velocity: full speed when aligned, reduced for large heading
        # error, and ramped down as we approach the goal.
        linear = self._max_linear * max(0.0, math.cos(error))
        linear = min(linear, self._max_linear * (dist_to_goal / self._lookahead))
        linear = max(0.0, min(self._max_linear, linear))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self._cmd_pub.publish(cmd)

    def _publish_stop(self):
        self._cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
