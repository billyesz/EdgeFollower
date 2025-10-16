#pragma once

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
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
  std::vector<cv::Point2f> extractContourPoints(const sensor_msgs::LaserScan &scan);
  nav_msgs::Path generateLocalTrajectory(const sensor_msgs::LaserScan &scan,
                                         const std::string &global_frame = "odom");

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
  // follow_left_ 定义：
  // true: 障碍物（墙）在机器人左侧，机器人沿其右侧边缘行驶,
  // false:	障碍物（墙）在机器人右侧，机器人沿其左侧边缘行驶
  bool follow_left_ = true;

  // ROS
  ros::Subscriber laser_sub_;
  ros::Publisher traj_pub_; // 用于可视化轨迹
  ros::Publisher fitted_contour_pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // 内部地图
  cv::Mat local_map_;
  mutable std::mutex map_mutex_;

  std::vector<cv::Point2f> offsetPoints(
      const std::vector<cv::Point2f> &points,
      float robot_radius);
  std::vector<cv::Point2f> fitSplineContour(const std::vector<cv::Point2f> &raw_points, int num_samples = 50);

  std::vector<cv::Point2f> offsetSegment(
      const std::vector<cv::Point2f> &segment,
      float robot_radius);

  void publishPointCloud(const std::vector<cv::Point2f> &points,
                         const std::string &frame_id,
                         const ros::Time &stamp,
                         ros::Publisher &pub);
};