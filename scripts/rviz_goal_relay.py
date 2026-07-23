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

"""Relay RViz "2D Goal Pose" clicks to the smac_planner_node action server.

The Nav2 RViz "2D Goal Pose" tool publishes a geometry_msgs/PoseStamped on
the ``goal_pose`` topic. This node listens for those clicks and forwards each
one as a ComputePathToPose goal to smac_planner_node, then publishes the
returned path on ``plan`` so it can be visualised back in RViz.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path


class RvizGoalRelay(Node):
    def __init__(self):
        super().__init__('rviz_goal_relay')

        # Match the C++ node's default action name (override if you remapped it).
        self._action_name = self.declare_parameter('action_name', 'compute_path_to_pose').value
        self._goal_topic = self.declare_parameter('goal_topic', 'goal_pose').value
        # If True, plan from the robot's current pose; if False, plan from origin.
        self._use_start = self.declare_parameter('use_start', False).value

        self._action_client = ActionClient(self, ComputePathToPose, self._action_name)

        self._goal_sub = self.create_subscription(
            PoseStamped, self._goal_topic, self._on_goal_clicked, 10)
        self._plan_pub = self.create_publisher(Path, 'plan', 1)

        # Track an in-flight goal so a fresh click cancels the previous request.
        self._goal_handle = None

        self.get_logger().info(
            f"Listening for RViz goals on '{self._goal_topic}', "
            f"calling action '{self._action_name}'.")

    def _on_goal_clicked(self, pose: PoseStamped):
        self.get_logger().info(
            f"Goal clicked in frame '{pose.header.frame_id}': "
            f"x={pose.pose.position.x:.2f} y={pose.pose.position.y:.2f}")

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"Action server '{self._action_name}' not available; is smac_planner_node running?")
            return

        # Cancel any goal still being computed so we plan to the latest click.
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = pose
        goal_msg.use_start = self._use_start
        # goal_msg.start stays default; only used when use_start is True.

        send_future = self._action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Goal rejected by smac_planner_node.')
            return
        self._goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result_wrapper = future.result()
        result = result_wrapper.result
        self._goal_handle = None

        if result.error_code != ComputePathToPose.Result.NONE:
            self.get_logger().warning(
                f"Planning failed (error_code={result.error_code}): {result.error_msg}")
            return

        n = len(result.path.poses)
        secs = result.planning_time.sec + result.planning_time.nanosec * 1e-9
        self.get_logger().info(f"Got path with {n} poses in {secs:.3f}s.")
        self._plan_pub.publish(result.path)


def main(args=None):
    rclpy.init(args=args)
    node = RvizGoalRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
