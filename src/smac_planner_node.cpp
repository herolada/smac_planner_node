// Copyright (c) 2026 Adam Herold
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone Hybrid-A* planner node.
//
// This node drives nav2_smac_planner's templated A* (AStarAlgorithm<NodeHybrid>)
// directly, without the rest of the Nav2 stack (no Costmap2DROS, no lifecycle
// manager, no planner server). It builds its planning costmap from a
// traversability PointCloud2 and exposes planning through a ComputePathToPose
// action server (geometry_msgs/PoseStamped goal in, nav_msgs/Path out).
//
// The planner-specific logic mirrors nav2_smac_planner/src/smac_planner_hybrid.cpp
// as closely as makes sense for a costmap-stack-free node.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav2_msgs/action/compute_path_to_pose.hpp"

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

#include "tf2/utils.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/smoother.hpp"
#include "nav2_smac_planner/constants.hpp"
#include "nav2_smac_planner/types.hpp"
#include "nav2_smac_planner/utils.hpp"

namespace smac_planner_node
{

using namespace std::chrono_literals;  // NOLINT
using ComputePathToPose = nav2_msgs::action::ComputePathToPose;
using GoalHandle = rclcpp_action::ServerGoalHandle<ComputePathToPose>;

/**
 * @class SmacPlannerNode
 * @brief Standalone Hybrid-A* planner: traversability PointCloud2 -> costmap ->
 *        AStarAlgorithm<NodeHybrid> -> Path, served over a ComputePathToPose action.
 */
class SmacPlannerNode : public rclcpp::Node
{
public:
  explicit SmacPlannerNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("smac_planner_node", options),
    _collision_checker(nullptr, 1, nullptr)
  {
    declareAndGetParameters();
    initializePlanner();

    _tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    _tf_listener = std::make_shared<tf2_ros::TransformListener>(*_tf_buffer);

    _traversability_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      _traversability_topic, rclcpp::SensorDataQoS(),
      std::bind(&SmacPlannerNode::traversabilityCallback, this, std::placeholders::_1));

    _plan_publisher = this->create_publisher<nav_msgs::msg::Path>("plan", 1);

    _action_server = rclcpp_action::create_server<ComputePathToPose>(
      this, _action_name,
      std::bind(&SmacPlannerNode::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&SmacPlannerNode::handleCancel, this, std::placeholders::_1),
      std::bind(&SmacPlannerNode::handleAccepted, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "smac_planner_node ready. Listening for traversability on '%s', "
      "serving ComputePathToPose on '%s'.",
      _traversability_topic.c_str(), _action_name.c_str());
  }

  ~SmacPlannerNode() override
  {
    nav2_smac_planner::NodeHybrid::destroyStaticAssets();
  }

private:
  // ----------------------------------------------------------------------- //
  // Setup
  // ----------------------------------------------------------------------- //
  void declareAndGetParameters()
  {
    // Topics / interfaces / frames
    _traversability_topic =
      declare_parameter<std::string>("traversability_topic", "traversability");
    _action_name = declare_parameter<std::string>("action_name", "compute_path_to_pose");
    _robot_base_frame = declare_parameter<std::string>("robot_base_frame", "base_link");
    _transform_tolerance = declare_parameter<double>("transform_tolerance", 0.1);

    // Costmap construction from the traversability point cloud
    _traversability_field =
      declare_parameter<std::string>("traversability_field", "traversability");
    _costmap_resolution = declare_parameter<double>("resolution", 0.1);
    _lethal_threshold = declare_parameter<double>("lethal_threshold", 0.65);

    // Collision checking (circular robot)
    _robot_radius = declare_parameter<double>("robot_radius", 0.3);

    // General planner params (mirror SmacPlannerHybrid defaults)
    int angle_quantizations = declare_parameter<int>("angle_quantization_bins", 72);
    _angle_bin_size = 2.0 * M_PI / angle_quantizations;
    _angle_quantizations = static_cast<unsigned int>(angle_quantizations);

    _tolerance = static_cast<float>(declare_parameter<double>("tolerance", 0.25));
    _allow_unknown = declare_parameter<bool>("allow_unknown", true);
    _max_iterations = declare_parameter<int>("max_iterations", 1000000);
    _max_on_approach_iterations = declare_parameter<int>("max_on_approach_iterations", 1000);
    _terminal_checking_interval = declare_parameter<int>("terminal_checking_interval", 5000);
    _smooth_path = declare_parameter<bool>("smooth_path", true);

    _minimum_turning_radius_global_coords =
      declare_parameter<double>("minimum_turning_radius", 0.4);
    _search_info.allow_primitive_interpolation =
      declare_parameter<bool>("allow_primitive_interpolation", false);
    _search_info.cache_obstacle_heuristic =
      declare_parameter<bool>("cache_obstacle_heuristic", false);
    _search_info.reverse_penalty =
      static_cast<float>(declare_parameter<double>("reverse_penalty", 2.0));
    _search_info.change_penalty =
      static_cast<float>(declare_parameter<double>("change_penalty", 0.0));
    _search_info.non_straight_penalty =
      static_cast<float>(declare_parameter<double>("non_straight_penalty", 1.2));
    _search_info.cost_penalty =
      static_cast<float>(declare_parameter<double>("cost_penalty", 2.0));
    _search_info.retrospective_penalty =
      static_cast<float>(declare_parameter<double>("retrospective_penalty", 0.015));
    _search_info.analytic_expansion_ratio =
      static_cast<float>(declare_parameter<double>("analytic_expansion_ratio", 3.5));
    _search_info.analytic_expansion_max_cost =
      static_cast<float>(declare_parameter<double>("analytic_expansion_max_cost", 200.0));
    _search_info.analytic_expansion_max_cost_override =
      declare_parameter<bool>("analytic_expansion_max_cost_override", false);
    _search_info.use_quadratic_cost_penalty =
      declare_parameter<bool>("use_quadratic_cost_penalty", false);
    _search_info.downsample_obstacle_heuristic =
      declare_parameter<bool>("downsample_obstacle_heuristic", true);

    double analytic_expansion_max_length_m =
      declare_parameter<double>("analytic_expansion_max_length", 3.0);
    _search_info.analytic_expansion_max_length =
      analytic_expansion_max_length_m / _costmap_resolution;

    _max_planning_time = declare_parameter<double>("max_planning_time", 5.0);
    _lookup_table_size = declare_parameter<double>("lookup_table_size", 20.0);

    _motion_model_for_search =
      declare_parameter<std::string>("motion_model_for_search", "DUBIN");
    _motion_model = nav2_smac_planner::fromString(_motion_model_for_search);
    if (_motion_model == nav2_smac_planner::MotionModel::UNKNOWN) {
      RCLCPP_WARN(
        get_logger(),
        "Unable to get MotionModel search type. Given '%s', "
        "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP, STATE_LATTICE. "
        "Defaulting to DUBIN.",
        _motion_model_for_search.c_str());
      _motion_model = nav2_smac_planner::MotionModel::DUBIN;
    }

    std::string goal_heading_type =
      declare_parameter<std::string>("goal_heading_mode", "DEFAULT");
    _goal_heading_mode = nav2_smac_planner::fromStringToGH(goal_heading_type);
    if (_goal_heading_mode == nav2_smac_planner::GoalHeadingMode::UNKNOWN) {
      RCLCPP_WARN(
        get_logger(),
        "Unable to get GoalHeadingMode. Given '%s', valid options are DEFAULT, "
        "BIDIRECTIONAL, ALL_DIRECTION. Defaulting to DEFAULT.",
        goal_heading_type.c_str());
      _goal_heading_mode = nav2_smac_planner::GoalHeadingMode::DEFAULT;
    }

    _coarse_search_resolution = declare_parameter<int>("coarse_search_resolution", 1);

    // Sanitize, matching SmacPlannerHybrid::configure
    if (_max_on_approach_iterations <= 0) {
      _max_on_approach_iterations = std::numeric_limits<int>::max();
    }
    if (_max_iterations <= 0) {
      _max_iterations = std::numeric_limits<int>::max();
    }
    if (_coarse_search_resolution <= 0) {
      _coarse_search_resolution = 1;
    }
    if (_angle_quantizations % _coarse_search_resolution != 0) {
      RCLCPP_WARN(
        get_logger(),
        "coarse_search_resolution should be an increment of angle_quantization_bins. "
        "Disabling coarse search.");
      _coarse_search_resolution = 1;
    }
    if (_minimum_turning_radius_global_coords < _costmap_resolution) {
      RCLCPP_WARN(
        get_logger(), "Min turning radius cannot be less than the costmap resolution!");
      _minimum_turning_radius_global_coords = _costmap_resolution;
    }

    _lethal_threshold = std::clamp(_lethal_threshold, 1e-3, 1.0);
  }

  void initializePlanner()
  {
    _search_info.minimum_turning_radius =
      static_cast<float>(_minimum_turning_radius_global_coords / _costmap_resolution);

    _lookup_table_dim =
      static_cast<float>(_lookup_table_size) / static_cast<float>(_costmap_resolution);
    _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));
    if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
      _lookup_table_dim += 1.0f;
    }

    // Precompute the angle bins for the collision checker (irregular-bin constructor,
    // re-enabled in nav2_smac_planner just for this standalone use case).
    _angles.reserve(_angle_quantizations);
    for (unsigned int i = 0; i != _angle_quantizations; ++i) {
      _angles.push_back(static_cast<float>(_angle_bin_size) * i);
    }
    _collision_checker = nav2_smac_planner::GridCollisionChecker(nullptr, _angles);

    _a_star =
      std::make_unique<nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid>>(
      _motion_model, _search_info);
    _a_star->initialize(
      _allow_unknown,
      _max_iterations,
      _max_on_approach_iterations,
      _terminal_checking_interval,
      _max_planning_time,
      _lookup_table_dim,
      _angle_quantizations);

    if (_smooth_path) {
      // SmootherParams::get() needs a LifecycleNode, which we are not, so declare the
      // smoother params on this node directly and fill the struct ourselves.
      nav2_smac_planner::SmootherParams params;
      params.tolerance_ = declare_parameter<double>("smoother.tolerance", 1e-10);
      params.max_its_ = declare_parameter<int>("smoother.max_iterations", 1000);
      params.w_data_ = declare_parameter<double>("smoother.w_data", 0.2);
      params.w_smooth_ = declare_parameter<double>("smoother.w_smooth", 0.3);
      params.do_refinement_ = declare_parameter<bool>("smoother.do_refinement", true);
      params.refinement_num_ = declare_parameter<int>("smoother.refinement_num", 2);
      params.holonomic_ = false;
      _smoother = std::make_unique<nav2_smac_planner::Smoother>(params);
      _smoother->initialize(_minimum_turning_radius_global_coords);
    }
  }

  // ----------------------------------------------------------------------- //
  // Traversability -> costmap
  // ----------------------------------------------------------------------- //
  void traversabilityCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto costmap = buildCostmap(*msg);
    if (!costmap) {
      return;
    }
    std::lock_guard<std::mutex> lock(_costmap_mutex);
    _costmap = std::move(costmap);
    _global_frame = msg->header.frame_id;
  }

  unsigned char costFromTraversability(float t) const
  {
    if (!std::isfinite(t)) {
      return nav2_costmap_2d::NO_INFORMATION;
    }
    t = std::clamp(t, 0.0f, 1.0f);
    if (t >= static_cast<float>(_lethal_threshold)) {
      return nav2_costmap_2d::LETHAL_OBSTACLE;
    }
    const float scaled =
      (t / static_cast<float>(_lethal_threshold)) * nav2_smac_planner::MAX_NON_OBSTACLE_COST;
    return static_cast<unsigned char>(std::lround(scaled));
  }

  // Build a Costmap2D sized to the cloud's XY extent. The cloud is assumed to be a
  // regular square grid with cell spacing == resolution. Returns nullptr on failure.
  std::shared_ptr<nav2_costmap_2d::Costmap2D> buildCostmap(
    const sensor_msgs::msg::PointCloud2 & cloud)
  {
    bool has_x = false, has_y = false, has_t = false;
    for (const auto & f : cloud.fields) {
      has_x |= (f.name == "x");
      has_y |= (f.name == "y");
      has_t |= (f.name == _traversability_field);
    }
    if (!has_x || !has_y || !has_t) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Traversability cloud missing required fields (need x, y, %s).",
        _traversability_field.c_str());
      return nullptr;
    }

    const size_t n = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
    if (n == 0) {
      return nullptr;
    }

    // First pass: XY bounds of valid points.
    sensor_msgs::PointCloud2ConstIterator<float> it_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(cloud, "y");
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    bool any = false;
    for (; it_x != it_x.end(); ++it_x, ++it_y) {
      const float x = *it_x;
      const float y = *it_y;
      if (!std::isfinite(x) || !std::isfinite(y)) {
        continue;
      }
      min_x = std::min(min_x, static_cast<double>(x));
      min_y = std::min(min_y, static_cast<double>(y));
      max_x = std::max(max_x, static_cast<double>(x));
      max_y = std::max(max_y, static_cast<double>(y));
      any = true;
    }
    if (!any) {
      return nullptr;
    }

    const double res = _costmap_resolution;
    const unsigned int size_x =
      static_cast<unsigned int>(std::lround((max_x - min_x) / res)) + 1u;
    const unsigned int size_y =
      static_cast<unsigned int>(std::lround((max_y - min_y) / res)) + 1u;
    // Origin is the lower-left cell corner; point at (min_x, min_y) lands in cell (0, 0)'s
    // center, i.e. cell center == origin + (i + 0.5) * res == min + i * res.
    const double origin_x = min_x - 0.5 * res;
    const double origin_y = min_y - 0.5 * res;

    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
      size_x, size_y, res, origin_x, origin_y, nav2_costmap_2d::NO_INFORMATION);

    // Second pass: stamp costs.
    sensor_msgs::PointCloud2ConstIterator<float> px(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> py(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> pt(cloud, _traversability_field);
    for (; px != px.end(); ++px, ++py, ++pt) {
      const float x = *px;
      const float y = *py;
      if (!std::isfinite(x) || !std::isfinite(y)) {
        continue;
      }
      const int mx = static_cast<int>(std::lround((x - min_x) / res));
      const int my = static_cast<int>(std::lround((y - min_y) / res));
      if (mx < 0 || my < 0 ||
        mx >= static_cast<int>(size_x) || my >= static_cast<int>(size_y))
      {
        continue;
      }
      costmap->setCost(
        static_cast<unsigned int>(mx), static_cast<unsigned int>(my),
        costFromTraversability(*pt));
    }

    return costmap;
  }

  // ----------------------------------------------------------------------- //
  // Action server
  // ----------------------------------------------------------------------- //
  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ComputePathToPose::Goal>)
  {
    std::lock_guard<std::mutex> lock(_costmap_mutex);
    if (!_costmap) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: no traversability map received yet.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandle>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    // Run planning off the executor thread; serialized via _planning_mutex.
    std::thread{std::bind(&SmacPlannerNode::execute, this, goal_handle)}.detach();
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    std::lock_guard<std::mutex> plan_lock(_planning_mutex);
    auto result = std::make_shared<ComputePathToPose::Result>();
    const auto goal = goal_handle->get_goal();
    const auto t0 = std::chrono::steady_clock::now();

    // Snapshot the latest costmap and its frame.
    std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap;
    std::string global_frame;
    {
      std::lock_guard<std::mutex> lock(_costmap_mutex);
      costmap = _costmap;
      global_frame = _global_frame;
    }
    if (!costmap) {
      result->error_code = ComputePathToPose::Result::UNKNOWN;
      result->error_msg = "No traversability map available";
      goal_handle->abort(result);
      return;
    }

    // Resolve goal and start poses in the costmap (global) frame.
    geometry_msgs::msg::PoseStamped goal_pose;
    geometry_msgs::msg::PoseStamped start_pose;
    if (!transformToGlobal(goal->goal, global_frame, goal_pose)) {
      result->error_code = ComputePathToPose::Result::TF_ERROR;
      result->error_msg = "Could not transform goal into '" + global_frame + "'";
      goal_handle->abort(result);
      return;
    }
    if (goal->use_start) {
      if (!transformToGlobal(goal->start, global_frame, start_pose)) {
        result->error_code = ComputePathToPose::Result::TF_ERROR;
        result->error_msg = "Could not transform start into '" + global_frame + "'";
        goal_handle->abort(result);
        return;
      }
    } else if (!getRobotPose(global_frame, start_pose)) {
      result->error_code = ComputePathToPose::Result::TF_ERROR;
      result->error_msg = "Could not look up robot pose in '" + global_frame + "'";
      goal_handle->abort(result);
      return;
    }

    auto cancel_checker = [goal_handle]() {return goal_handle->is_canceling();};

    nav_msgs::msg::Path path;
    uint16_t error_code = ComputePathToPose::Result::NONE;
    std::string error_msg;
    const bool ok = createPlan(
      start_pose, goal_pose, costmap, global_frame, cancel_checker, path, error_code, error_msg);

    if (goal_handle->is_canceling()) {
      result->error_code = ComputePathToPose::Result::UNKNOWN;
      result->error_msg = "Planning canceled";
      goal_handle->canceled(result);
      return;
    }

    if (!ok) {
      result->error_code = error_code;
      result->error_msg = error_msg;
      goal_handle->abort(result);
      RCLCPP_WARN(get_logger(), "Planning failed: %s", error_msg.c_str());
      return;
    }

    result->path = path;
    const auto t1 = std::chrono::steady_clock::now();
    result->planning_time =
      rclcpp::Duration::from_seconds(std::chrono::duration<double>(t1 - t0).count());
    result->error_code = ComputePathToPose::Result::NONE;

    if (_plan_publisher->get_subscription_count() > 0) {
      _plan_publisher->publish(path);
    }
    goal_handle->succeed(result);
  }

  // ----------------------------------------------------------------------- //
  // Planning core (mirrors SmacPlannerHybrid::createPlan)
  // ----------------------------------------------------------------------- //
  bool createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const std::shared_ptr<nav2_costmap_2d::Costmap2D> & costmap,
    const std::string & global_frame,
    std::function<bool()> cancel_checker,
    nav_msgs::msg::Path & plan,
    uint16_t & error_code,
    std::string & error_msg)
  {
    _plan_start = std::chrono::steady_clock::now();
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));

    // Point the collision checker and A* at the current costmap.
    _collision_checker.setCostmap(costmap.get());
    nav2_costmap_2d::Footprint footprint;  // unused for circular checking
    _collision_checker.setFootprint(
      footprint, /*radius=*/ true, static_cast<double>(nav2_smac_planner::INSCRIBED_COST));
    _a_star->setCollisionChecker(&_collision_checker);

    // Start, in A* bin search coordinates.
    float mx_start, my_start, mx_goal, my_goal;
    if (!costmap->worldToMapContinuous(
        start.pose.position.x, start.pose.position.y, mx_start, my_start))
    {
      error_code = ComputePathToPose::Result::START_OUTSIDE_MAP;
      error_msg = "Start outside costmap bounds";
      return false;
    }
    unsigned int start_bin = orientationToBin(tf2::getYaw(start.pose.orientation));
    _a_star->setStart(mx_start, my_start, start_bin);

    // Goal, in A* bin search coordinates.
    if (!costmap->worldToMapContinuous(
        goal.pose.position.x, goal.pose.position.y, mx_goal, my_goal))
    {
      error_code = ComputePathToPose::Result::GOAL_OUTSIDE_MAP;
      error_msg = "Goal outside costmap bounds";
      return false;
    }
    unsigned int goal_bin = orientationToBin(tf2::getYaw(goal.pose.orientation));
    _a_star->setGoal(mx_goal, my_goal, goal_bin, _goal_heading_mode, _coarse_search_resolution);

    // Setup output message.
    plan.header.stamp = now();
    plan.header.frame_id = global_frame;
    geometry_msgs::msg::PoseStamped pose;
    pose.header = plan.header;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;

    // Corner case: start and goal in the same cell + heading.
    if (std::floor(mx_start) == std::floor(mx_goal) &&
      std::floor(my_start) == std::floor(my_goal) && start_bin == goal_bin)
    {
      pose.pose = start.pose;
      pose.pose.orientation = goal.pose.orientation;
      plan.poses.push_back(pose);
      return true;
    }

    // Compute plan.
    nav2_smac_planner::NodeHybrid::CoordinateVector path;
    int num_iterations = 0;
    const float tolerance = _tolerance / static_cast<float>(costmap->getResolution());
    if (!_a_star->createPath(path, num_iterations, tolerance, cancel_checker, nullptr)) {
      if (num_iterations == 1) {
        error_code = ComputePathToPose::Result::START_OCCUPIED;
        error_msg = "Start occupied";
      } else if (num_iterations < _a_star->getMaxIterations()) {
        error_code = ComputePathToPose::Result::NO_VALID_PATH;
        error_msg = "No valid path could be found";
      } else {
        error_code = ComputePathToPose::Result::TIMEOUT;
        error_msg = "Exceeded maximum iterations";
      }
      return false;
    }

    // Convert to world coordinates (backtrace yields goal-to-start order).
    plan.poses.reserve(path.size());
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
      pose.pose = nav2_smac_planner::getWorldCoords(path[i].x, path[i].y, costmap.get());
      pose.pose.orientation = nav2_smac_planner::getWorldOrientation(path[i].theta);
      plan.poses.push_back(pose);
    }

    // Smooth, with whatever time is left.
    if (_smoother && num_iterations > 1) {
      const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - _plan_start).count();
      double time_remaining = _max_planning_time - elapsed;
      _smoother->smooth(plan, costmap.get(), time_remaining);
    }

    return true;
  }

  unsigned int orientationToBin(double yaw) const
  {
    double bin = std::round(yaw / _angle_bin_size);
    while (bin < 0.0) {
      bin += static_cast<double>(_angle_quantizations);
    }
    if (bin >= static_cast<double>(_angle_quantizations)) {
      bin -= static_cast<double>(_angle_quantizations);
    }
    return static_cast<unsigned int>(bin);
  }

  // ----------------------------------------------------------------------- //
  // TF helpers
  // ----------------------------------------------------------------------- //
  bool transformToGlobal(
    const geometry_msgs::msg::PoseStamped & in,
    const std::string & global_frame,
    geometry_msgs::msg::PoseStamped & out)
  {
    if (in.header.frame_id.empty() || in.header.frame_id == global_frame) {
      out = in;
      out.header.frame_id = global_frame;
      return true;
    }
    try {
      out = _tf_buffer->transform(
        in, global_frame, tf2::durationFromSec(_transform_tolerance));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "TF transform to '%s' failed: %s", global_frame.c_str(), ex.what());
      return false;
    }
    return true;
  }

  bool getRobotPose(const std::string & global_frame, geometry_msgs::msg::PoseStamped & pose)
  {
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = _tf_buffer->lookupTransform(
        global_frame, _robot_base_frame, tf2::TimePointZero,
        tf2::durationFromSec(_transform_tolerance));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Robot pose lookup failed: %s", ex.what());
      return false;
    }
    pose.header.frame_id = global_frame;
    pose.header.stamp = tf.header.stamp;
    pose.pose.position.x = tf.transform.translation.x;
    pose.pose.position.y = tf.transform.translation.y;
    pose.pose.position.z = tf.transform.translation.z;
    pose.pose.orientation = tf.transform.rotation;
    return true;
  }

  // ----------------------------------------------------------------------- //
  // Members
  // ----------------------------------------------------------------------- //
  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _traversability_sub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _plan_publisher;
  rclcpp_action::Server<ComputePathToPose>::SharedPtr _action_server;
  std::shared_ptr<tf2_ros::Buffer> _tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> _tf_listener;

  // Planner
  std::unique_ptr<nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid>> _a_star;
  nav2_smac_planner::GridCollisionChecker _collision_checker;
  std::unique_ptr<nav2_smac_planner::Smoother> _smoother;
  std::vector<float> _angles;

  // Costmap (rebuilt on every traversability message; only the latest is kept)
  std::shared_ptr<nav2_costmap_2d::Costmap2D> _costmap;
  std::string _global_frame;
  std::mutex _costmap_mutex;
  std::mutex _planning_mutex;
  std::chrono::steady_clock::time_point _plan_start;

  // Parameters
  std::string _traversability_topic, _action_name, _robot_base_frame, _traversability_field;
  double _transform_tolerance;
  double _costmap_resolution;
  double _lethal_threshold;
  double _robot_radius;
  float _tolerance;
  bool _allow_unknown;
  int _max_iterations;
  int _max_on_approach_iterations;
  int _terminal_checking_interval;
  bool _smooth_path;
  double _angle_bin_size;
  unsigned int _angle_quantizations;
  nav2_smac_planner::SearchInfo _search_info;
  double _max_planning_time;
  double _lookup_table_size;
  double _minimum_turning_radius_global_coords;
  float _lookup_table_dim;
  std::string _motion_model_for_search;
  nav2_smac_planner::MotionModel _motion_model;
  nav2_smac_planner::GoalHeadingMode _goal_heading_mode;
  int _coarse_search_resolution;
};

}  // namespace smac_planner_node

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<smac_planner_node::SmacPlannerNode>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
