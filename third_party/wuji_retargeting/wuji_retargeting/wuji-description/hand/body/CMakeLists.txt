cmake_minimum_required(VERSION 3.10)
project(wuji_description)

find_package(ament_cmake REQUIRED)

install(
    DIRECTORY launch meshes mjcf rviz step urdf usd
    DESTINATION share/${PROJECT_NAME}
)

# Install sibling hand/attachment/ so that package:// URLs in docking-ros.urdf
# resolve via the package share. Requires the full repository layout to be
# present at build time (do not copy hand/body/ in isolation).
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../attachment/
    DESTINATION share/${PROJECT_NAME}/attachment
)

ament_package()
