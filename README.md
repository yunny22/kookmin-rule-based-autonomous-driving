# Kookmin Final – Rule-based Autonomous Driving

## Overview

This is a public-source staging repository for the real-vehicle Rule-based Autonomous Driving System prepared for the Kookmin autonomous-driving competition final. It contains the audited centerline-perception and vehicle-control path only. Competition assets, model weights, calibration files, bags, media, and source that was not verified as part of this path are intentionally absent.

## Motivation

After the preceding RL / Sim-to-Real effort could not reduce the high-speed simulation-to-real gap sufficiently within the competition schedule, the final architecture was reorganized as a Rule-based system that could be explicitly tuned and verified on the vehicle. Lane information is converted into a vehicle-frame target path, then a fused Stanley and Pure Pursuit controller publishes vehicle commands. The repository is a reviewable source release, not a vehicle-ready deployment package.

## System Architecture

```text
camera image
  -> LR-ASPP semantic segmentation
  -> white-boundary / yellow-centerline masks
  -> BEV and canonical-road rendering
  -> centerline target path
  -> adaptive Stanley + Pure Pursuit fusion
  -> shadow vehicle command (motor output disabled by default)
```

The public launcher composes the two audited ROS 2 packages:

- `lane_seg_control`: perception and canonical-road generation.
- `xycar_rule_drive`: target-path extraction and fused steering control.

It is a sanitized composition of the `driver_mode=rule` branch in the original mixed final launcher. It does **not** include the original launcher because that file also selects excluded RL and Hybrid paths.

## Centerline Perception

The audited source implements **LR-ASPP** TorchScript semantic segmentation, not an Xbin model. The source code contains no Xbin class or module, so this repository does not describe it as an Xbin reimplementation.

LR-ASPP produces white-boundary and yellow-centerline masks. The canonical adapter warps those masks to a bird's-eye representation and renders a canonical road image for the controller. A compatible TorchScript model must be supplied locally through `model_path`; the original launch convention used `kookmin_lane_lraspp_mbv3s_256x144.pt`, but no weight is published here.

## Vehicle Control

`canonical_stanley_pursuit_driver.py` derives a Pure Pursuit steering term from a look-ahead target and a Stanley term from cross-track and heading error. It adaptively blends the terms: straight-path and departure guards modify the Pure Pursuit weight, and opposing terms are handled before smoothing and rate limiting. This is a fused controller, not a fixed mode switch.

The launcher sets `drive_enabled:=false`. It publishes only to the shadow-command path until a locally calibrated vehicle configuration and an explicit operator decision enable motor output.

## Mission Integration

This staging release preserves the verified final perception-and-control boundary. The mixed original final launcher proves that the rule branch selected these two packages, but it does not identify a self-contained competition-day traffic, obstacle, stop-line, or cone mission stack.

`track_drive_sve` identifies itself as a preliminary Gazebo package, while `study/my_rule` is SLAM-coupled and has local working-tree changes. They are intentionally not represented as final mission modules here. The competition-day mission integration remains **UNKNOWN** pending a verified final launch command or deployment record.

## My Contribution

Source-history evidence attributes portions of the LR-ASPP/canonical pipeline and Stanley–Pure Pursuit integration to the local author identity `as`. That is **PERSONAL evidence for source contributions**, not proof that every final-system module was implemented individually.

- **PERSONAL evidence:** core perception/canonical and fused-control source contributions in the audited commit history.
- **CO-DEVELOPED:** integration of perception and control into the competition Rule path.
- **TEAM:** final vehicle system, competition operation, and any result that depends on components omitted from this public release.

## Competition / Validation

The source was selected from commit `678522e99f6527fa151796686b3c3574df55e18f` in the local TeamKAI repository because its mixed final launcher explicitly exposes a Rule branch that includes the retained perception and controller packages. This repository makes no claim about competition ranking, completion rate, real-vehicle performance, or exact inference frequency.

## Limitations

- A local TorchScript LR-ASPP model is required and is not supplied.
- Camera geometry, steering calibration, speed settings, and vehicle-interface parameters are hardware-specific and must be measured locally.
- Motor output is disabled by default.
- The public source does not include an audited final traffic/obstacle/cone mission stack.
- Post-competition improvement should balance perception accuracy with execution frequency, expand representative data, and validate controller parameters on the actual vehicle.

## Repository Structure

```text
xycar_ws/src/
├── lane_seg_control/      # LR-ASPP masks and canonical-road generation
└── xycar_rule_drive/      # Stanley + Pure Pursuit fused controller
```

`PUBLIC_RELEASE_NOTES.md` records the source-selection decision, exclusions, and provenance boundaries.

## Environment

- Ubuntu with ROS 2 Humble
- Python 3, `numpy`, `opencv-python`, `PyYAML`, and PyTorch compatible with the supplied TorchScript model
- ROS dependencies declared in both package manifests
- The external `kaiev26_msgs` message package, supplied separately by the team environment

The external message package is not copied here because it belongs to a separate vehicle/SITL interface source set.

## How to Run

```bash
cd xycar_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lane_seg_control xycar_rule_drive
source install/setup.bash

ros2 launch xycar_rule_drive final_rule_only.launch.py \
  model_path:=/absolute/path/to/local-lraspp-model.pt \
  image_topic:=/camera/topic
```

The launch is intentionally safe by default: `drive_enabled:=false`. To use a real vehicle, provide a locally measured, non-public parameter file based on `config/vehicle.example.yaml`, review the vehicle interface, and make an explicit operator decision before enabling output. This repository does not provide a ready-to-drive configuration.

## Team / Credits

The source was audited from the [TeamKAI competition repository](https://github.com/steveandy-sudo/kookmin_autonomous_competition_teamKAI). This staging repository preserves only an attributable, reviewable slice of a broader team system. The original repository history and any team-level attribution should be consulted before publication or reuse.
