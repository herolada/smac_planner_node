# smac_planner_node

A standalone **Hybrid-A\*** planner node that reuses
[`nav2_smac_planner`](../navigation2/nav2_smac_planner)'s templated A\* core
(`AStarAlgorithm<NodeHybrid>`) **without** the rest of the Nav2 stack — no
`Costmap2DROS`, no costmap layers, no lifecycle/planner-server machinery.

It builds its planning costmap directly from a **traversability point cloud** and
exposes planning through a `ComputePathToPose` action server.

## Interfaces

| Direction | Name | Type | Notes |
|---|---|---|---|
| Subscribe | `traversability` | `sensor_msgs/PointCloud2` | Square grid; float field `traversability` in `[0, 1]` (0 = most traversable). Only the **latest** cloud is kept. |
| Action server | `compute_path_to_pose` | `nav2_msgs/action/ComputePathToPose` | Goal `PoseStamped` in → result `nav_msgs/Path` out. |
| Publish | `plan` | `nav_msgs/Path` | Last planned path (for debug/visualization). |
| TF | — | — | Looks up `global_frame → robot_base_frame` for the start pose when `use_start=false`. |

The **global frame** is taken from the `frame_id` of the latest traversability
cloud. Goal (and `start`, when `use_start=true`) poses are transformed into that
frame via TF.

## Costmap construction

The cloud is treated as a regular grid with cell spacing equal to the `resolution`
parameter. The costmap is sized to the cloud's XY extent. Each point's
traversability `t` maps to a Nav2 cost:

- `t >= lethal_threshold` → `LETHAL_OBSTACLE` (254)
- otherwise → linearly scaled into `[0, 252]`
- cells with no point / non-finite values → `NO_INFORMATION` (255)

## Build & run

```bash
# from the workspace root
bash ./build.sh --packages-select nav2_smac_planner   # Part 1 core changes
bash ./build.sh --packages-select smac_planner_node

. /opt/ros/kilted/setup.sh && . install/setup.bash
ros2 launch smac_planner_node smac_planner_node.launch.py
# or
ros2 run smac_planner_node smac_planner_node --ros-args \
  --params-file install/smac_planner_node/share/smac_planner_node/config/smac_planner_node.yaml
```

Parameters mirror `SmacPlannerHybrid` (see `config/smac_planner_node.yaml`).

## Relationship to `nav2_smac_planner`

This package depends on a small change in `nav2_smac_planner` (Part 1): the
obstacle-heuristic plumbing and the `GridCollisionChecker` accept a raw
`nav2_costmap_2d::Costmap2D*` in addition to a `Costmap2DROS`, so the planner core
can be driven without the full ROS costmap node. The `SmacPlannerHybrid` plugin is
unchanged in behavior.

## Known limitation

Collision checking is **center-cell, radius-based** against the traversability
costmap; there is no inflation layer. The traversability values are expected to
already encode proximity cost. If you need the robot footprint enforced with a
hard inflation margin, inflate the costmap (or pre-inflate the traversability map)
before/within costmap construction.
