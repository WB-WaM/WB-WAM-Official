cmake_minimum_required(VERSION 3.10)
project(g1_comp_servo_service)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    link_directories(lib/arm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    link_directories(lib/x86)
else()
    message(STATUS "Architecture not supported")
endif()


find_package(Boost COMPONENTS program_options REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  thirdparty/DynamixelSDK/c++/include
)

link_libraries(unitree_sdk2 ddsc ddscxx rt pthread) 
link_libraries(libboost_program_options.a yaml-cpp) 
link_libraries(dxl_x64_cpp fmt)

# test
add_executable(main main.cpp)
add_executable(test_read_yaml test/test_read_yaml.cpp)
add_executable(test_calibration test/test_calibration.cpp)
add_executable(test_joint0_control test/test_joint0_control.cpp)
add_executable(test_joint1_control test/test_joint1_control.cpp)
add_executable(test_servo_control test/test_servo_control.cpp)
add_executable(test_read_angle test/test_read_angle.cpp)
add_executable(test_servo_homePostion test/test_servo_homePostion.cpp)