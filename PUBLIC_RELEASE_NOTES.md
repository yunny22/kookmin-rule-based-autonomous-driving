# Public Release Notes

## Provenance

- **Original repository:** https://github.com/steveandy-sudo/kookmin_autonomous_competition_teamKAI
- **Audited branch:** main
- **Audited source commit:** 0ae216c6255e25404560948f3e6d5479a7a7ba8f
- **Audit date:** 2026-09-12
- **Selection basis:** traced the final execution path from
  run_complete_space_hybrid.sh through run_space_hybrid_test.sh,
  real_sequential_hybrid_drive.launch.py, the Xbin perception node, fused RULE
  controller, sequential_hybrid_driver, space_drive_gate, and the external
  VESC subscriber.

## Architecture confirmed from source

- Normal lane driving used direct Xbin yellow-centerline perception.
- The direct centerline was projected to vehicle-frame metric points and
  published on /perception/xbin_direct_centerline.
- The final RULE candidate used fused Pure Pursuit and Stanley control.
- LR-ASPP was wired to the shortcut/W1 subsystem, not the normal
  lane-driving path.
- The selector priority was TRAFFIC > SHORTCUT > CONE > YOLO + LiDAR
  AVOIDANCE > RULE.
- The competition script started the selector in shadow mode; space_drive_gate
  was the final /xycar_motor publisher, followed by the external VESC driver.

## Included source

| Area | Public files |
| --- | --- |
| Xbin lane path | lane_seg_control Xbin model/interface, direct-centerline projection, camera and canonical helpers, one unit test |
| RULE controller | xycar_rule_drive fused Pure Pursuit–Stanley controller and supporting path helpers |
| Traffic / avoidance | xycar_map_nav traffic state, YOLO/LiDAR avoidance state, LiDAR processing and mission/lap helpers |
| Arbitration / output | sequential_hybrid_driver.py, space_drive_gate.py, and their pure cores |
| Safe launch | xbin_rule_shadow.launch.py, which leaves motor output disabled |

## Deliberately excluded

- Xbin, YOLO and LR-ASPP weights; datasets; logs; rosbag files; videos; images;
  and generated build products.
- Camera calibration, camera–LiDAR extrinsics, measured steering maps, speed
  tables, hardware ports, network settings, and real competition coordinates.
- The VESC hardware driver and vehicle interface.
- The shortcut/W1 and cone perception/driving implementations, which are
  teammate subsystems. Their candidate interfaces are documented only.
- Third-party and external TeamKAI packages required by the original runtime.

## Contribution boundary

**Taeyun Kim:** Xbin lane path, direct metric path, Pure Pursuit–Stanley
vehicle controller implementation/tuning, traffic logic, YOLO/LiDAR vehicle
avoidance, mission/lap priority arbitration, and final vehicle-command
integration and tuning.

**Other team members:** shortcut/W1 perception/driving and cone
perception/driving. Their commands/events were integrated into the final stack,
but their underlying algorithms are not claimed as Taeyun Kim's work.

## Team result

The TeamKAI README at the audited commit records a three-lap integrated
completion on 2026-08-23. This is retained only as a **team result**. No
independent result artifact is distributed in this release.

## Sanitization and release policy

- No model, dataset, vehicle log, calibration, route, credential, private IP,
  or source-machine path is included.
- The root repository intentionally has no open-source LICENSE file. This
  portfolio/research demonstration release does not re-license dependencies.
- Package metadata is marked UNLICENSED for this curated release.
- ROS 2, OpenCV, NumPy, PyTorch, Ultralytics, and external message/hardware
  interfaces remain third-party or external dependencies.

## Validation

- Python syntax compilation of all published Python sources: **passed**.
- Selected pure unit tests for direct Xbin projection, traffic logic,
  output-gate core, and lap policy: **43 passed**.
- `test_yolo_lidar_avoidance.py` was not run in this environment because its
  ROS node import requires the external `rclpy` runtime and message packages.
- XML parsing for all package manifests: **passed**.
- YAML validation is **not applicable**: real vehicle configuration is
  deliberately excluded.
- `colcon build --packages-select lane_seg_control xycar_rule_drive
  xycar_map_nav`: **passed** (3 packages).
- `ros2 launch xycar_map_nav xbin_rule_shadow.launch.py --show-args`:
  **passed**. A full vehicle launch was not run because local authorised model,
  calibration, message interfaces, and hardware are intentionally excluded.

## Scope limitation

This repository is an auditable portfolio subset. It documents the final
competition architecture without exposing vehicle-ready calibration or
teammate implementation source. It must not be used to command a vehicle
without locally authorised models, calibration, interfaces, and safety review.
