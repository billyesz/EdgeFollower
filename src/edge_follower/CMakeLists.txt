cmake_minimum_required(VERSION 3.0.2)
project(edge_follower)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
  visualization_msgs
)

# find_package(OpenCV REQUIRED)
find_package(OpenCV 3 REQUIRED COMPONENTS core imgproc)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs tf2_ros cv_bridge visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(edge_follower_node
  src/edge_follower_node.cpp
  src/edge_follower.cpp
)

target_link_libraries(edge_follower_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)