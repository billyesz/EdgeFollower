#include "edge_follower/edge_follower.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc.hpp>
// #include <opencv2/core/eigen.hpp>

EdgeFollower::EdgeFollower(ros::NodeHandle &nh)
    : tf_listener_(tf_buffer_),
      map_size_px_(static_cast<int>(map_size_m_ / resolution_))
{

  local_map_ = cv::Mat::zeros(map_size_px_, map_size_px_, CV_8UC1);

  laser_sub_ = nh.subscribe("scan", 1, &EdgeFollower::laserCallback, this);
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
  // marker_pub_ = nh.advertise<visualization_msgs::Marker>("edge_following_debug", 10);
  marker_pub_ = nh.advertise<visualization_msgs::Marker>("/edge_following_debug", 1);

  // 可选：发布局部地图图像用于调试
  // auto map_pub = nh.advertise<sensor_msgs::Image>("local_map", 1);
}

// ===== 激光回调 =====
void EdgeFollower::laserCallback(const sensor_msgs::LaserScanConstPtr &scan)
{
  // === 1. 获取机器人在 odom 坐标系中的位姿 ===
  geometry_msgs::TransformStamped robot_tf;
  try
  {
    robot_tf = tf_buffer_.lookupTransform(
        "odom", "base_footprint", scan->header.stamp, ros::Duration(0.1));
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN_THROTTLE(1.0, "Failed to get robot pose in odom: %s", ex.what());
    return;
  }

  double robot_x = robot_tf.transform.translation.x;
  double robot_y = robot_tf.transform.translation.y;
  double robot_yaw = tf2::getYaw(robot_tf.transform.rotation);
  double cos_yaw = std::cos(robot_yaw);
  double sin_yaw = std::sin(robot_yaw);

  // === 2. 更新局部地图（保持原有逻辑，但点来自 odom）===
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    local_map_ = cv::Mat::zeros(map_size_px_, map_size_px_, CV_8UC1);
  }

  int center = map_size_px_ / 2;

  // === 3. 将每个激光点转换到 odom 坐标系，并投影到局部地图 ===
  for (size_t i = 0; i < scan->ranges.size(); ++i)
  {
    float range = scan->ranges[i];
    if (!std::isfinite(range) || range < scan->range_min || range > scan->range_max)
      continue;

    double angle = scan->angle_min + i * scan->angle_increment;
    double lx = range * std::cos(angle);
    double ly = range * std::sin(angle);

    // 手动将点从 base_scan 转到 base_footprint（通常 z=0, 且 base_scan 与 base_link 同轴）
    // 注意：TurtleBot3 中 base_scan 是 base_link 的子坐标系，而 base_link 与 base_footprint 很近
    // 为简化，我们假设 base_scan ≈ base_footprint（或通过 URDF 精确转换）
    // 但更准确的做法是：先转到 base_link，再用 TF 转到 odom —— 但我们已用 scan->header.stamp 获取了 odom->base_footprint

    // 所以这里我们直接把 (lx, ly) 当作 base_footprint 下的点（近似）
    // 如果精度要求高，应使用完整 TF 链：base_scan -> base_link -> base_footprint -> odom
    // 但 TurtleBot3 的 base_scan 到 base_footprint 平移很小（z=0.18），xy≈0

    // 因此，我们仍用原有方式计算 wx, wy（因为局部地图是以机器人中心为原点）
    double wx = robot_x + cos_yaw * lx - sin_yaw * ly;
    double wy = robot_y + sin_yaw * lx + cos_yaw * ly;

    double dx = wx - robot_x;
    double dy = wy - robot_y;

    int px = static_cast<int>(dx / resolution_ + center);
    int py = static_cast<int>(dy / resolution_ + center);

    if (px >= 0 && px < map_size_px_ && py >= 0 && py < map_size_px_)
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      local_map_.at<uchar>(py, px) = 255;
    }
  }

  // === 4. 边缘跟随逻辑（保持不变）===
  auto points = getAllObstaclePoints();
  if (points.empty())
  {
    publishVelocity(cv::Point2f(0, 0)); // 停止
    return;
  }

  auto clusters = clusterPoints(points, 0.2);
  std::vector<EdgeSegment> segments;
  for (const auto &cl : clusters)
  {
    if (cl.size() >= 3)
    {
      segments.push_back(fitEdgeWithPCA(cl));
    }
  }

  cv::Point2f target;
  if (findTarget(segments, target))
  {
    publishVelocity(target);
    // 关键：传入 robot_pose 用于可视化，但我们改用 odom 中的 pose
    geometry_msgs::PoseStamped robot_pose_odom;
    robot_pose_odom.header.frame_id = "odom";
    robot_pose_odom.header.stamp = scan->header.stamp;
    robot_pose_odom.pose.position.x = robot_x;
    robot_pose_odom.pose.position.y = robot_y;
    robot_pose_odom.pose.position.z = 0.0;
    robot_pose_odom.pose.orientation = robot_tf.transform.rotation;
    publishVisualization(segments, target, robot_pose_odom);
  }
  else
  {
    publishVelocity(cv::Point2f(0, 0));
  }
}

// ===== 获取障碍点 =====
std::vector<cv::Point2f> EdgeFollower::getAllObstaclePoints()
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  std::vector<cv::Point2f> points;
  int center = map_size_px_ / 2;
  for (int py = 0; py < local_map_.rows; ++py)
  {
    for (int px = 0; px < local_map_.cols; ++px)
    {
      if (local_map_.at<uchar>(py, px) == 255)
      {
        double dx = (px - center) * resolution_;
        double dy = (py - center) * resolution_;
        points.emplace_back(dx, dy);
      }
    }
  }
  return points;
}

// ===== 聚类 =====
std::vector<std::vector<cv::Point2f>> EdgeFollower::clusterPoints(
    const std::vector<cv::Point2f> &points, double eps)
{
  std::vector<std::vector<cv::Point2f>> clusters;
  std::vector<bool> visited(points.size(), false);

  for (size_t i = 0; i < points.size(); ++i)
  {
    if (visited[i])
      continue;
    std::vector<cv::Point2f> cluster;
    std::queue<size_t> q;
    q.push(i);
    visited[i] = true;

    while (!q.empty())
    {
      size_t idx = q.front();
      q.pop();
      cluster.push_back(points[idx]);
      for (size_t j = 0; j < points.size(); ++j)
      {
        if (!visited[j] && cv::norm(points[idx] - points[j]) <= eps)
        {
          visited[j] = true;
          q.push(j);
        }
      }
    }
    clusters.push_back(cluster);
  }
  return clusters;
}

// ===== PCA 拟合 =====
EdgeFollower::EdgeSegment EdgeFollower::fitEdgeWithPCA(const std::vector<cv::Point2f> &pts)
{
  cv::Mat data(pts.size(), 2, CV_32F);
  for (size_t i = 0; i < pts.size(); ++i)
  {
    data.at<float>(i, 0) = pts[i].x;
    data.at<float>(i, 1) = pts[i].y;
  }
  cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW);
  cv::Point2f mean(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
  cv::Point2f eigvec(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
  float norm = cv::norm(eigvec);
  if (norm > 1e-6)
    eigvec /= norm;
  double length = std::sqrt(pca.eigenvalues.at<float>(0));
  return {mean, eigvec, length};
}

// ===== 选择目标 =====
bool EdgeFollower::findTarget(const std::vector<EdgeSegment> &segments, cv::Point2f &target)
{
  constexpr double MIN_DIST = 0.3;
  constexpr double MAX_DIST = 1.8;
  constexpr double ANGLE_HALF = 75.0 * M_PI / 180.0;

  const EdgeSegment *best = nullptr;
  double best_dist = std::numeric_limits<double>::max();

  for (const auto &seg : segments)
  {
    double dist = cv::norm(seg.centroid);
    double angle = std::atan2(seg.centroid.y, seg.centroid.x);
    if (dist < MIN_DIST || dist > MAX_DIST)
      continue;
    if (std::abs(angle) > ANGLE_HALF)
      continue;
    if (dist < best_dist)
    {
      best_dist = dist;
      best = &seg;
    }
  }

  if (!best)
    return false;

  cv::Point2f to_robot = -best->centroid;
  double n = cv::norm(to_robot);
  if (n < 1e-3)
    return false;
  cv::Point2f normal = to_robot / n;

  cv::Point2f tangent = best->direction;
  if (follow_left_)
  {
    tangent = cv::Point2f(-tangent.y, tangent.x); // 左侧跟随
  }
  else
  {
    tangent = cv::Point2f(tangent.y, -tangent.x);
  }

  target = best->centroid + safe_distance_ * normal + 0.2f * tangent;
  return true;
}

// ===== 发布速度 =====
void EdgeFollower::publishVelocity(const cv::Point2f &target)
{
  double dist = cv::norm(target);
  double angle = std::atan2(target.y, target.x);

  geometry_msgs::Twist cmd;
  constexpr double Kp_ang = 1.2, Kp_lin = 0.5;
  constexpr double MAX_ANG = 0.8, MAX_LIN = 0.3;

  // cmd.angular.z = std::clamp(Kp_ang * angle, -MAX_ANG, MAX_ANG);
  double ang = Kp_ang * angle;
  if (ang > MAX_ANG)
    ang = MAX_ANG;
  if (ang < -MAX_ANG)
    ang = -MAX_ANG;
  cmd.angular.z = ang;

  if (std::abs(angle) < 0.2)
  {
    cmd.linear.x = std::min(Kp_lin * dist, MAX_LIN);
  }

  cmd_vel_pub_.publish(cmd);
}

// ===== 可视化调试 =====
void EdgeFollower::publishVisualization(
    const std::vector<EdgeSegment> &segments,
    const cv::Point2f &target,
    const geometry_msgs::PoseStamped &robot_pose)
{

  visualization_msgs::Marker marker;
  marker.header.frame_id = "odom";
  marker.header.stamp = ros::Time::now();
  marker.ns = "edge_following";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.02;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  double rx = robot_pose.pose.position.x;
  double ry = robot_pose.pose.position.y;

  for (const auto &seg : segments)
  {
    // 绘制主方向线段（±0.5m）
    cv::Point2f p1 = seg.centroid + 0.5f * seg.direction;
    cv::Point2f p2 = seg.centroid - 0.5f * seg.direction;

    geometry_msgs::Point gp1, gp2;
    gp1.x = rx + p1.x;
    gp1.y = ry + p1.y;
    gp1.z = 0.0;
    gp2.x = rx + p2.x;
    gp2.y = ry + p2.y;
    gp2.z = 0.0;
    marker.points.push_back(gp1);
    marker.points.push_back(gp2);
  }

  ROS_INFO("Marker stamp: %.3f", marker.header.stamp.toSec());
  marker_pub_.publish(marker);

  // 发布目标点
  visualization_msgs::Marker target_marker;
  target_marker.header = marker.header;
  target_marker.ns = "target";
  target_marker.id = 0;
  target_marker.type = visualization_msgs::Marker::SPHERE;
  target_marker.action = visualization_msgs::Marker::ADD;
  target_marker.pose.position.x = rx + target.x;
  target_marker.pose.position.y = ry + target.y;
  target_marker.pose.position.z = 0.1;
  target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.1;
  target_marker.color.g = 1.0;
  target_marker.color.a = 1.0;
  marker_pub_.publish(target_marker);
}

// ===== 你需要实现的位姿获取函数 =====
bool EdgeFollower::getRobotPose(geometry_msgs::PoseStamped &pose)
{
  try
  {
    pose.header.frame_id = "odom";
    pose.header.stamp = ros::Time(0);
    geometry_msgs::TransformStamped transform =
        tf_buffer_.lookupTransform("odom", "base_footprint", ros::Time(0), ros::Duration(0.1));
    pose.pose.position.x = transform.transform.translation.x;
    pose.pose.position.y = transform.transform.translation.y;
    pose.pose.orientation = transform.transform.rotation;
    return true;
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN("%s", ex.what());
    return false;
  }
}