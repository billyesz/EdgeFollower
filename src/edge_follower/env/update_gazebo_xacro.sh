#!/bin/sh

### !!! 修改Turtlebot模型中激光雷达的参数 !!! ###

URDF_PATH=/opt/ros/melodic/share/turtlebot3_description/urdf

sudo mv $URDF_PATH/turtlebot3_burger.gazebo.xacro $URDF_PATH/turtlebot3_burger.gazebo.xacro.bk

sudo cp turtlebot3_burger.gazebo.xacro $URDF_PATH/

echo "Update done."

