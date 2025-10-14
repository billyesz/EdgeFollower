#pragma once

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf/tf.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <limits>
#include <mutex>

class EdgeFollower
{
public:
  EdgeFollower(ros::NodeHandle &nh);
  void laserCallback(const sensor_msgs::LaserScanConstPtr &scan);
  bool getRobotPose(geometry_msgs::PoseStamped &pose); // 需你实现或注入

protected:
  struct EdgeSegment
  { // 如果需要的话，可以调整访问级别
    cv::Point2f centroid{0, 0};
    cv::Point2f direction{0, 0};
    double length{0.0};
  };

private:
  // 参数
  double map_size_m_ = 3.0;
  double resolution_ = 0.05;
  int map_size_px_;
  double safe_distance_ = 0.3;
  bool follow_left_ = true; // true: 沿左侧走（障碍在右）

  // ROS
  ros::Subscriber laser_sub_;
  ros::Publisher cmd_vel_pub_;
  ros::Publisher marker_pub1_;
  ros::Publisher marker_pub2_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // 内部地图
  cv::Mat local_map_;
  mutable std::mutex map_mutex_;

  // 核心逻辑
  std::vector<cv::Point2f> getAllObstaclePoints();
  std::vector<std::vector<cv::Point2f>> clusterPoints(
      const std::vector<cv::Point2f> &points, double eps = 0.2);

  EdgeSegment fitEdgeWithPCA(const std::vector<cv::Point2f> &pts);
  bool findTarget(const std::vector<EdgeSegment> &segments, cv::Point2f &target);
  void publishVelocity(const cv::Point2f &target);
  void publishVisualization(
      const std::vector<EdgeSegment> &segments,
      const cv::Point2f &target_local,
      const geometry_msgs::PoseStamped &robot_pose,
      const ros::Time &stamp); // 添加时间戳参数

  void publishVisualization2(
      const std::vector<EdgeSegment> &segments,
      const cv::Point2f &target_local,
      const geometry_msgs::PoseStamped &robot_pose,
      const ros::Time &stamp);
};