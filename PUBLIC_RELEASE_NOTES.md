# Public Release Notes

## Source selection

- **Audited source branch:** `codex/slam-gazebo-controller`
- **Audited commit:** `678522e99f6527fa151796686b3c3574df55e18f`
- **Original remote:** `git@github.com:steveandy-sudo/kookmin_autonomous_competition_teamKAI.git`
- **Selection basis:** `xycar_final_drive/launch/final_real_stack.launch.py` exposes a `driver_mode=rule` branch that launches LR-ASPP canonical perception and the canonical Stanley–Pure Pursuit driver.

The original final launcher is **SUPPORTING** evidence, not copied source: it also selects RL and Hybrid launch paths that are outside this Rule-only release.

## Classification

| Classification | Source | Public-release decision |
| --- | --- | --- |
| FINAL | `lane_seg_control` LR-ASPP, mask-to-canonical path | Included after model/calibration sanitization |
| FINAL | `xycar_rule_drive` canonical Stanley–Pure Pursuit driver | Included after launch/config sanitization |
| SUPPORTING | `xycar_final_drive/final_real_stack.launch.py` Rule branch | Documented, not copied because mixed with RL/Hybrid |
| EXPERIMENT / PRELIMINARY | `track_drive_sve` | Excluded; package documentation identifies a Gazebo preliminary path |
| EXPERIMENT / SLAM-COUPLED | `study/my_rule`, `xycar_map_nav` | Excluded; SLAM dependencies and working-tree changes make them unsuitable for this scoped release |
| OBSOLETE / NONESSENTIAL | legacy camera, keyboard, bag, RViz launch variants | Excluded |
| UNKNOWN | competition-day traffic, obstacle, and cone mission stack | Omitted pending a verified final deployment command |

## Included

- LR-ASPP input preprocessing, semantic-mask generation, BEV/canonical rendering, and white-lane fitting utilities.
- Canonical centerline path processing and adaptive Stanley–Pure Pursuit fusion.
- A new `final_rule_only.launch.py` that composes only the audited Rule path.
- Unit-test sources and package manifests required to build the retained packages.

## Provenance

| Scope | Classification | Basis |
| --- | --- | --- |
| LR-ASPP/canonical and fused-control source portions | PERSONAL | Selected-commit history attributes relevant code portions to one local author identity |
| Rule-path assembly | CO-DEVELOPED | The final launcher belongs to a broader team workspace and selects shared packages |
| Vehicle deployment and competition operation | TEAM | The retained source does not establish individual ownership of the complete vehicle system |
| ROS 2, OpenCV, NumPy, PyTorch, and external message interfaces | THIRD-PARTY / EXTERNAL | Runtime dependencies are referenced but their source is not copied |
| Competition-day traffic, obstacle, stop-line, and cone modules | UNKNOWN | A self-contained final deployment reference was not found |

## Intentionally excluded

- All model weights (`.pt`, `.pth`, `.onnx`, `.engine`), including LR-ASPP and object models.
- Sim-to-Real, RL, Hybrid, SLAM, parking, and KAI source sets.
- ROS bags, datasets, camera recordings, images, video, experiment outputs, and large archives.
- Hardware-specific camera calibration, measured steering maps, speed tables, and competition vehicle configuration.
- YOLO/vendor source and nonessential RViz or debugging artifacts.

## Sanitization

- The public launch requires `model_path` rather than embedding a model file.
- Camera calibration defaults are empty and rectification is disabled unless a local file is supplied.
- The public launch defaults to shadow operation with `drive_enabled=false`.
- The measured steering-map defaults are neutral placeholders; local calibration belongs in an ignored `vehicle.local.yaml`.
- No source-machine paths, known model binaries, or source-machine calibration files are included.

## Validation

- `python3 -m compileall -q xycar_ws/src/lane_seg_control xycar_ws/src/xycar_rule_drive` completed successfully.
- Both package manifests parsed as XML.
- `colcon build --packages-select lane_seg_control xycar_rule_drive` completed successfully in a ROS 2 Humble environment.
- The retained perception tests passed: 2 tests.
- The retained controller tests passed: 42 tests after making pure controller functions importable without the external `kaiev26_msgs` runtime package.
- `ros2 launch ... --show-args` resolved both the perception and public Rule-only launch descriptions.
- An artifact/path scan found no model weight, recording, dataset, or source-machine path in the staged source tree.

## Remaining Issues

- Package metadata is `UNLICENSED`; this portfolio repository grants no separate
  open-source reuse license. Third-party dependency terms remain in force.
- Confirm whether the LR-ASPP model weight may be redistributed; it is currently excluded.
- Obtain the exact competition-day launch/deployment record before adding mission, traffic, obstacle, stop-line, or cone modules.
- Repeat full vehicle validation only with a locally measured camera, steering, speed, and vehicle-interface configuration.

## Attribution boundary

Git blame at the selected commit associates portions of the retained
perception/canonical and controller fusion code with one local author identity.
That evidence supports source-level **PERSONAL** contribution only. The final
competition system and its result are **TEAM** work; perception-control
integration is presented as **CO-DEVELOPED**. No unaudited mission module is
claimed as an individual contribution.

## Release status

This directory is local staging only. It has no remote, has not been pushed, and has not been published. A maintainer must complete the remaining checks before making any repository public.
