# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the Wuji Glove model under `glove/body/`: left and right URDF skeletons at `glove/body/urdf/{left,right}.urdf`, each with 21 revolute joints across the five fingers, an electromagnetic transmitter base on the wrist, and a receiver coil on every fingertip for hand motion tracking.
- Added the glove transmitter and receiver coil meshes at `glove/body/mesh/base_link_TX.STL` and `glove/body/mesh/base_link_RX.STL`.
- Added the transmitter top-cover STEP file and assembled PDF drawing at `glove/body/step/EMFTXC_topcover.step` and `glove/body/step/EMFTXC_topcover.pdf`.

### Removed

- Removed the standalone glove mounting-interface STEP `glove/attachment/glove-attachment.step`. Glove assets now live under `glove/body/`.

## [2026.05.19]

### Fixed

- Corrected the left palm inertia of the Wuji Hand so that the center of mass and inertia tensor are a proper XZ-plane mirror of the right palm. Updated `hand/body/urdf/left.urdf`, `hand/body/urdf/left-ros.urdf`, `hand/body/mjcf/left.xml`, and `hand/body/usd/left/wujihand.usd`.

## [2026.05.18]

### Added

- Added the `wuji_description` ROS2 package under `hand/body/`, with `launch/display.launch.py`, RViz presets, `CMakeLists.txt`, and `package.xml` for left and right Wuji Hand visualization.
- Added URDF models for the left and right Wuji Hand at `hand/body/urdf/{left,right}.urdf` (relative mesh paths) and `hand/body/urdf/{left,right}-ros.urdf` (`package://` paths for ROS2).
- Added MuJoCo MJCF models at `hand/body/mjcf/{left,right}.xml` and STL visual/collision meshes at `hand/body/meshes/{left,right}/`.
- Added Isaac Sim USD assets at `hand/body/usd/{left,right}/`, including fused meshes, PBR materials, physics properties, and collision filter pairs.
- Added simplified structural STEP files of the hand frame at `hand/body/step/`.
- Added the impact-resistant docking attachment at `hand/attachment/impact-resistant-attachment/` with STL, URDF, MJCF, and USD assets, including the ROS URDF that references `package://wuji_description/attachment/impact-resistant-attachment/meshes/hand_docking_link.STL`.
- Added the Unitree G1 mounting adapter at `hand/attachment/unitree-g1-attachment/unitree-g1-docking-adapter.stl`.
- Added adapter STEP files, assembled PDF drawings, and installation notes at `hand/attachment/step/`.
- Added the Glove mounting interface STEP asset at `glove/attachment/glove-attachment.step`.
- Added the top-level `README.md`, `LICENSE` (MIT), and this `CHANGELOG.md`.

[Unreleased]: https://github.com/wuji-technology/wuji-description/compare/v2026.05.19...HEAD
[2026.05.19]: https://github.com/wuji-technology/wuji-description/releases/tag/v2026.05.19
[2026.05.18]: https://github.com/wuji-technology/wuji-description/releases/tag/v2026.05.18
