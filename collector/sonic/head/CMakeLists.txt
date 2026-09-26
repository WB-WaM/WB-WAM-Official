cmake_minimum_required(VERSION 3.16)
project(wbwam_head_servo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/vendor/DynamixelSDK/c++)
add_library(dynamixel STATIC
  ${SDK}/src/dynamixel_sdk/port_handler.cpp
  ${SDK}/src/dynamixel_sdk/port_handler_linux.cpp
  ${SDK}/src/dynamixel_sdk/packet_handler.cpp
  ${SDK}/src/dynamixel_sdk/protocol1_packet_handler.cpp
  ${SDK}/src/dynamixel_sdk/protocol2_packet_handler.cpp)
target_include_directories(dynamixel PUBLIC ${SDK}/include/dynamixel_sdk)
target_compile_definitions(dynamixel PRIVATE LINUX)
add_executable(head_servo head_servo.cpp)
target_link_libraries(head_servo PRIVATE dynamixel pthread)
target_compile_options(head_servo PRIVATE -Wall -Wextra -Werror)
enable_testing()
add_test(NAME cli_help COMMAND head_servo --help)
add_test(NAME invalid_duration COMMAND head_servo --hold-current --motion-approved --duration -1)
set_tests_properties(invalid_duration PROPERTIES WILL_FAIL TRUE)
add_test(NAME missing_motion_approval COMMAND head_servo --hold-current)
set_tests_properties(missing_motion_approval PROPERTIES WILL_FAIL TRUE)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME serial_protocol_and_cleanup
  COMMAND ${CMAKE_COMMAND} -E env WBWAM_HEAD_SERVO_TEST_BINARY=$<TARGET_FILE:head_servo>
  ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_head_servo.py)
