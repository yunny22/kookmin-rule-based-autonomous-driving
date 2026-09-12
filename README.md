# Kookmin Final – Rule-based Autonomous Driving

This repository is provided for portfolio and research demonstration purposes.
It is a curated, sanitized source subset of TeamKAI's final Kookmin autonomous
driving competition stack. It is not a vehicle-ready deployment package and
does not grant a separate open-source reuse licence.

## Overview

The final competition lane-driving path used a direct **Xbin yellow-centerline**
model. Its centerline points were converted directly to a vehicle-frame metric
path on /perception/xbin_direct_centerline; the controller then produced a
RULE candidate through fused Pure Pursuit and Stanley control.

The complete team stack combined that normal-driving candidate with traffic,
shortcut, cone, and YOLO/LiDAR vehicle-avoidance candidates. The final command
was selected by a mission/lap state machine, sent to a shadow command topic,
then gated before /xycar_motor reached the external VESC interface.

The source and execution wiring were audited against
[TeamKAI main at 0ae216c](https://github.com/steveandy-sudo/kookmin_autonomous_competition_teamKAI/tree/0ae216c6255e25404560948f3e6d5479a7a7ba8f).
See [PUBLIC_RELEASE_NOTES.md](PUBLIC_RELEASE_NOTES.md) for provenance and
exclusions.

## Motivation

The preceding learning and Sim-to-Real work exposed a high-speed transfer gap
that could not be reduced sufficiently within the competition schedule. The
final approach therefore used an explicitly tunable, real-vehicle Rule-based
architecture. This release documents the final architecture; it does not
present Rule-based driving as a replacement for the research value of the
preceding learning experiments.

## Final System Architecture

~~~text
Wide camera
  ├─ Xbin yellow-centerline perception
  │    └─ vehicle-frame metric centerline/path
  │         └─ Pure Pursuit + Stanley → RULE candidate
  ├─ YOLO object detections
  │    └─ traffic state / vehicle mission input
  └─ LR-ASPP shortcut/W1 perception (team subsystem)
       └─ SHORTCUT candidate

LiDAR
  ├─ cone-boundary input (team subsystem) → CONE candidate
  └─ vehicle distance / free-space input → AVOIDANCE candidate

TRAFFIC > SHORTCUT > CONE > YOLO + LiDAR AVOIDANCE > RULE
  └─ sequential_hybrid_driver
       └─ /hybrid_gate/xycar_motor_shadow
            └─ space_drive_gate
                 └─ /xycar_motor → external VESC driver
~~~

In the audited competition script, sequential_hybrid_driver was started with
drive_enabled=false, so it emitted the shadow candidate only. The separately
started space_drive_gate was the actual /xycar_motor publisher. The VESC
driver is an external hardware package and is not copied here.

## Xbin Lane Perception

lane_seg_control contains the audited direct-centerline path:

1. The Xbin centerline model receives the wide-camera image.
2. Each confident yellow-centerline row yields an x-bin center estimate.
3. xbin_direct_centerline.py projects those image points with a local
   projective mapping into [forward_m, lateral_m] coordinates.
4. The perception node publishes the metric points as
   /perception/xbin_direct_centerline.
5. canonical_stanley_pursuit_driver consumes that topic as its external
   path input.

The model weight, camera calibration, and measured projective geometry are
not published. The public source has safe example defaults only.
The generic file name lraspp_inference_node.py was retained from the shared
TorchScript interface; the audited final launch supplied the Xbin model and
enabled its direct-centerline path.

### LR-ASPP's final role

LR-ASPP was **not** the final normal lane-driving perception path. In the
audited final launcher it was supplied to the shortcut/W1 subsystem. That
subsystem was developed by another team member and is represented here only by
its candidate-command interface, not by copied perception source.

## RULE Controller

xycar_rule_drive/canonical_stanley_pursuit_driver.py builds a connected metric
path and computes:

- a Pure Pursuit term from a look-ahead point;
- a Stanley term from cross-track and heading error; and
- a fused steering command with straight/curve handling, smoothing and
  steering-rate limiting.

The project used standard Pure Pursuit and Stanley ideas; the contribution was
their vehicle-side implementation, parameter tuning, and fused integration,
not a claim of inventing either algorithm. The published steering-map default
is a neutral placeholder. It must not be used as a vehicle calibration.

## Traffic and Vehicle Avoidance

xycar_map_nav contains the final selector-side logic:

- traffic_light_control.py latches stop/go/left-arrow state from YOLO
  detections and supplies a traffic mission decision.
- yolo_lidar_avoidance.py tracks a YOLO vehicle observation, uses LiDAR
  distance and side/free-space information, requests a lateral avoidance
  offset, and returns to the RULE path after the obstacle clears.
- sequential_hybrid_driver.py combines those mission states with the RULE
  candidate, shortcut event and cone candidate.

The public subset intentionally does **not** include YOLO model weights,
camera-to-LiDAR calibration, object-detector runtime code, or a ready-to-run
vehicle configuration.

## Mission Arbitration

The audited sequential_hybrid_driver.py applies the following precedence:

~~~text
TRAFFIC
> SHORTCUT
> CONE
> YOLO + LiDAR AVOIDANCE
> RULE
~~~

Traffic can stop the vehicle; shortcut and cone candidates can replace the
RULE command; vehicle avoidance retains the RULE steering path while applying
an avoidance offset and speed constraint. The selector publishes the chosen
result to /hybrid_gate/xycar_motor_shadow. space_drive_gate.py provides the
final operator-gated output to /xycar_motor.

## Team Scope and My Contribution

### Taeyun Kim

- Implemented and integrated Xbin lane perception, yellow-centerline
  extraction, and direct metric-path generation.
- Implemented and tuned the Pure Pursuit–Stanley fused RULE controller for
  vehicle path tracking.
- Implemented traffic-light perception/mission logic and traffic control.
- Implemented YOLO + LiDAR vehicle-avoidance logic, recovery handling, and
  its integration with the RULE path.
- Implemented mission/lap state and priority arbitration, then integrated the
  final command flow, real vehicle, and tuning work.

### Other Team Members

- The shortcut/W1 perception and its driving subsystem were developed by a
  team member.
- The cone perception and cone-driving subsystem were developed by a team
  member.

Those two subsystems were integrated into the final mission stack, but their
underlying algorithms are not claimed here as Taeyun Kim's individual work.

## Team Result

TeamKAI's final main README records a **three-lap integrated completion on
2026-08-23**. This is presented as a team result from the source provenance;
this public subset contains no independent bag, video, or benchmark artifact
for replay.

## Public Subset

~~~text
xycar_ws/src/
├── lane_seg_control/    # Xbin path extraction and ROS interface
├── xycar_rule_drive/    # fused Pure Pursuit–Stanley controller
└── xycar_map_nav/       # traffic, avoidance, arbitration, output gate
~~~

The public xbin_rule_shadow.launch.py demonstrates only the Xbin-to-RULE
boundary and keeps physical motor output disabled. It does not start a final
competition run because the model, calibration, hardware interface, shortcut
module and cone module are intentionally excluded.

## Environment and Validation

- ROS 2 Humble source layout
- Python 3, NumPy, OpenCV, PyTorch and the ROS dependencies declared in the
  package manifests
- External TeamKAI message and hardware packages, supplied separately

~~~bash
cd xycar_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lane_seg_control xycar_rule_drive xycar_map_nav
source install/setup.bash

ros2 launch xycar_map_nav xbin_rule_shadow.launch.py \
  model_path:=/absolute/path/to/authorised/xbin_centerline_model.pt
~~~

This command is a source-interface review aid, not a ready-to-drive vehicle
instruction. Provide locally authorised model, calibration, message interfaces
and safety validation before any hardware integration.

## Credits

This release derives from the
[TeamKAI competition repository](https://github.com/steveandy-sudo/kookmin_autonomous_competition_teamKAI).
ROS 2, NumPy, OpenCV, PyTorch, Ultralytics, and ROS message/interface packages
are external dependencies; their source is not vendored or re-licensed here.
