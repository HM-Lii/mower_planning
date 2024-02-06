cmake_minimum_required(VERSION 3.0.2)
project(mower_planning)

## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  rospy
  std_msgs
  geometry_msgs
)
find_package(OsqpEigen REQUIRED)
catkin_package(
)


include_directories(
    include
  ${catkin_INCLUDE_DIRS}
)



add_executable(path_planning src/piecewise_jerk.cpp)
target_link_libraries(path_planning ${catkin_LIBRARIES} OsqpEigen::OsqpEigen)
