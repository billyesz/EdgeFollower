#include "edge_follower/edge_follower.h"
#include <cv_bridge/cv_bridge.h>
#include <cmath> // for std::isfinite
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc.hpp>
#include <tf/transform_datatypes.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>

EdgeFollower::EdgeFollower(ros::NodeHandle &nh)
    : tf_listener_(tf_buffer_),
      map_size_px_(static_cast<int>(map_size_m_ / resolution_))
{
  local_map_ = cv::Mat::zeros(map_size_px_, map_size_px_, CV_8UC1);

  laser_sub_ = nh.subscribe("scan", 1, &EdgeFollower::laserCallback, this);
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
  marker_pub1_ = nh.advertise<visualization_msgs::Marker>("/edge_following_debug1", 1);
  marker_pub2_ = nh.advertise<visualization_msgs::MarkerArray>("/edge_following_debug2", 1);
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
  double robot_yaw = tf::getYaw(robot_tf.transform.rotation);
  double cos_yaw = std::cos(robot_yaw);
  double sin_yaw = std::sin(robot_yaw);

  // === 2. 更新局部地图 ===
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    local_map_ = cv::Mat::zeros(map_size_px_, map_size_px_, CV_8UC1);
  }

  int center = map_size_px_ / 2;

  // === 3. 投影激光点到局部地图（局部坐标系：机器人中心为原点）===
  for (size_t i = 0; i < scan->ranges.size(); ++i)
  {
    float range = scan->ranges[i];
    if (!std::isfinite(range) || range < scan->range_min || range > scan->range_max)
      continue;

    double angle = scan->angle_min + i * scan->angle_increment;
    double lx = range * std::cos(angle);
    double ly = range * std::sin(angle);

    // // 转换到局部地图坐标（相对于机器人中心）
    // double dx = cos_yaw * lx - sin_yaw * ly; // 实际上这里不需要，因为局部地图本身就是机器人坐标系
    // double dy = sin_yaw * lx + cos_yaw * ly;

    // 但注意：局部地图是以机器人当前位置为中心的，所以直接用 (lx, ly) 即可！
    // 因为 scan->header.frame_id 通常是 base_scan，而 base_scan ≈ base_footprint（xy≈0）
    // 所以我们可以简化：直接用 (lx, ly) 作为局部坐标
    // 但为了鲁棒性，保留原有方式（不影响结果）

    // int px = static_cast<int>(dx / resolution_ + center);
    // int py = static_cast<int>(dy / resolution_ + center);
    // 直接使用 (lx, ly) 作为局部坐标（相对于 base_footprint）
    int px = static_cast<int>(lx / resolution_ + center);
    int py = static_cast<int>(ly / resolution_ + center);

    if (px >= 0 && px < map_size_px_ && py >= 0 && py < map_size_px_)
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      local_map_.at<uchar>(py, px) = 255;
    }
  }

  // === 4. 边缘跟随逻辑 ===
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

  cv::Point2f target_local;
  if (findTarget(segments, target_local))
  {
    // target_local 是相对于机器人中心的 (dx, dy)，单位：米
    publishVelocity(target_local);

    // 构造机器人位姿（用于可视化参考）
    geometry_msgs::PoseStamped robot_pose;
    robot_pose.header.frame_id = "odom";
    robot_pose.header.stamp = scan->header.stamp;
    robot_pose.pose.position.x = robot_x;
    robot_pose.pose.position.y = robot_y;
    robot_pose.pose.orientation = robot_tf.transform.rotation;

    // publishVisualization(segments, target_local, robot_pose, scan->header.stamp);
    publishVisualization2(segments, target_local, robot_pose, scan->header.stamp);
  }
  else
  {
    publishVelocity(cv::Point2f(0, 0));
  }
}

// ===== 获取障碍点（局部坐标）=====
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

// ===== 选择目标（返回局部坐标）=====
bool EdgeFollower::findTarget(const std::vector<EdgeSegment> &segments, cv::Point2f &target)
{
  constexpr double MIN_DIST = 0.3;
  constexpr double MAX_DIST = 1.8;
  constexpr double ANGLE_HALF = 75.0 * M_PI / 180.0;

  const EdgeSegment *best = nullptr;
  double best_dist = std::numeric_limits<double>::max();

  // 1. 选择最近的有效线段（在视野内、距离合适）
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

  // 2. 计算线段方向的单位向量
  double dir_norm = cv::norm(best->direction);
  if (dir_norm < 1e-6)
    return false;

  cv::Point2f dir_unit = best->direction / static_cast<float>(dir_norm);

  // >>>>>>>>>> 新增：统一线段方向（朝机器人前方） <<<<<<<<<<
  // 假设机器人朝 x 正方向，强制线段方向 x >= 0
  if (dir_unit.x < 0)
  {
    dir_unit = -dir_unit;
  }

  // 3. 计算垂直于线段的单位向量（左侧 or 右侧）
  cv::Point2f normal;
  if (follow_left_)
  {
    // 左侧：逆时针旋转90° → (-dy, dx)
    normal = cv::Point2f(-dir_unit.y, dir_unit.x);
  }
  else
  {
    // 右侧：顺时针旋转90° → (dy, -dx)
    normal = cv::Point2f(dir_unit.y, -dir_unit.x);
  }

  // 4. 目标点 = 线段中心 + 法向偏移 + （可选）沿切向微调
  //    切向微调用于让小车稍微朝前，避免垂直对齐
  cv::Point2f tangent = dir_unit; // 沿线段方向
  target = best->centroid + safe_distance_ * normal + 0.2f * tangent;

  // 5. 防御性检查
  if (!std::isfinite(target.x) || !std::isfinite(target.y))
    return false;

  return true;
}

// ===== 发布速度（输入：局部坐标 dx, dy）=====
void EdgeFollower::publishVelocity(const cv::Point2f &target)
{
  double dist = cv::norm(target);
  double angle = std::atan2(target.y, target.x);

  geometry_msgs::Twist cmd;
  constexpr double Kp_ang = 1.2, Kp_lin = 0.5;
  constexpr double MAX_ANG = 0.8, MAX_LIN = 0.3;

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

void EdgeFollower::publishVisualization2(
    const std::vector<EdgeSegment> &segments,
    const cv::Point2f &target_local,
    const geometry_msgs::PoseStamped &robot_pose,
    const ros::Time &stamp)
{
  // === 1. 检查目标点是否有效 ===
  if (!std::isfinite(target_local.x) || !std::isfinite(target_local.y))
  {
    ROS_WARN_THROTTLE(1.0, "Target point is NaN/Inf, skipping visualization.");
    return;
  }

  visualization_msgs::MarkerArray marker_array;

  // === 2. 发布边缘线段（红线）===
  for (size_t i = 0; i < segments.size(); ++i)
  {
    const auto &seg = segments[i];

    // 跳过无效线段：长度太小 或 方向向量为零
    if (seg.length < 0.01)
    {
      continue;
    }

    double dir_norm_val = cv::norm(seg.direction);
    if (dir_norm_val < 1e-6)
    {
      continue;
    }

    // 安全归一化
    cv::Point2f dir_norm = seg.direction / static_cast<float>(dir_norm_val);

    // 计算线段端点（局部坐标）
    cv::Point2f p1_local = seg.centroid - dir_norm * (static_cast<float>(seg.length) * 0.5f);
    cv::Point2f p2_local = seg.centroid + dir_norm * (static_cast<float>(seg.length) * 0.5f);

    // 再次检查端点是否有效
    if (!std::isfinite(p1_local.x) || !std::isfinite(p1_local.y) ||
        !std::isfinite(p2_local.x) || !std::isfinite(p2_local.y))
    {
      continue;
    }

    // 构建局部向量
    tf2::Vector3 v1_local(p1_local.x, p1_local.y, 0.0);
    tf2::Vector3 v2_local(p2_local.x, p2_local.y, 0.0);

    // 获取机器人全局位姿变换
    tf2::Transform robot_tf;
    tf2::fromMsg(robot_pose.pose, robot_tf);

    // 变换到全局坐标系
    tf2::Vector3 v1_global = robot_tf * v1_local;
    tf2::Vector3 v2_global = robot_tf * v2_local;

    // 创建 Marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = robot_pose.header.frame_id; // e.g., "odom"
    marker.header.stamp = stamp;
    marker.ns = "edges";
    marker.id = static_cast<int>(i);
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;

    // 必须显式设置 scale 和 color 所有分量
    marker.scale.x = 0.02; // 线宽 2cm
    marker.scale.y = 0.0;
    marker.scale.z = 0.0;

    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f; // alpha 必须 >0

    // 添加两个点（LINE_LIST 需成对）
    geometry_msgs::Point gp1, gp2;
    gp1.x = v1_global.x();
    gp1.y = v1_global.y();
    gp1.z = 0.0;
    gp2.x = v2_global.x();
    gp2.y = v2_global.y();
    gp2.z = 0.0;

    marker.points.push_back(gp1);
    marker.points.push_back(gp2);

    marker_array.markers.push_back(marker);
  }

  // === 3. 发布目标点（绿球）===
  {
    tf2::Vector3 target_local_vec(target_local.x, target_local.y, 0.0);
    tf2::Transform robot_tf;
    tf2::fromMsg(robot_pose.pose, robot_tf);
    tf2::Vector3 target_global = robot_tf * target_local_vec;

    visualization_msgs::Marker target_marker;
    target_marker.header.frame_id = robot_pose.header.frame_id;
    target_marker.header.stamp = stamp;
    target_marker.ns = "target";
    target_marker.id = 0;
    target_marker.type = visualization_msgs::Marker::SPHERE;
    target_marker.action = visualization_msgs::Marker::ADD;

    target_marker.pose.position.x = target_global.x();
    target_marker.pose.position.y = target_global.y();
    target_marker.pose.position.z = 0.0;
    target_marker.pose.orientation.w = 1.0; // 避免未初始化

    target_marker.scale.x = 0.1;
    target_marker.scale.y = 0.1;
    target_marker.scale.z = 0.1;

    target_marker.color.r = 0.0f;
    target_marker.color.g = 1.0f;
    target_marker.color.b = 0.0f;
    target_marker.color.a = 1.0f;

    marker_array.markers.push_back(target_marker);
  }

  // === 4. （可选）发布机器人位姿箭头 ===
  {
    visualization_msgs::Marker pose_marker;
    pose_marker.header.frame_id = robot_pose.header.frame_id;
    pose_marker.header.stamp = stamp;
    pose_marker.ns = "robot_pose";
    pose_marker.id = 0;
    pose_marker.type = visualization_msgs::Marker::ARROW;
    pose_marker.action = visualization_msgs::Marker::ADD;
    pose_marker.pose = robot_pose.pose;

    pose_marker.scale.x = 0.3; // 长度
    pose_marker.scale.y = 0.05;
    pose_marker.scale.z = 0.05;

    pose_marker.color.r = 0.0f;
    pose_marker.color.g = 0.0f;
    pose_marker.color.b = 1.0f;
    pose_marker.color.a = 1.0f;

    marker_array.markers.push_back(pose_marker);
  }

  // === 5. 发布 MarkerArray ===
  // 注意：即使 markers 为空，也可以安全发布
  marker_pub2_.publish(marker_array);
}

// // ===== 可视化调试（关键修复）=====
void EdgeFollower::publishVisualization(
    const std::vector<EdgeSegment> &segments,
    const cv::Point2f &target_local,
    const geometry_msgs::PoseStamped &robot_pose,
    const ros::Time &stamp)
{
  // 发布边缘线段
  visualization_msgs::Marker line_marker;
  line_marker.header.frame_id = "odom";
  line_marker.header.stamp = stamp; // ← 使用激光时间戳
  line_marker.ns = "edges";
  line_marker.id = 0;
  line_marker.type = visualization_msgs::Marker::LINE_LIST;
  line_marker.action = visualization_msgs::Marker::ADD;
  line_marker.scale.x = 0.02;
  line_marker.color.r = 1.0;
  line_marker.color.g = 0.0;
  line_marker.color.b = 0.0;
  line_marker.color.a = 1.0;

  double rx = robot_pose.pose.position.x;
  double ry = robot_pose.pose.position.y;

  for (const auto &seg : segments)
  {
    cv::Point2f p1 = seg.centroid + 0.5f * seg.direction;
    cv::Point2f p2 = seg.centroid - 0.5f * seg.direction;

    // 转换到 odom 坐标系
    geometry_msgs::Point gp1, gp2;
    gp1.x = rx + p1.x;
    gp1.y = ry + p1.y;
    gp1.z = 0.0;
    gp2.x = rx + p2.x;
    gp2.y = ry + p2.y;
    gp2.z = 0.0;

    line_marker.points.push_back(gp1);
    line_marker.points.push_back(gp2);
  }

  if (!line_marker.points.empty())
  {
    marker_pub1_.publish(line_marker);
  }

  // 发布目标点（在 odom 中）
  visualization_msgs::Marker target_marker;
  target_marker.header.frame_id = "odom";
  target_marker.header.stamp = stamp;
  target_marker.ns = "target";
  target_marker.id = 0;
  target_marker.type = visualization_msgs::Marker::SPHERE;
  target_marker.action = visualization_msgs::Marker::ADD;
  target_marker.pose.position.x = rx + target_local.x;
  target_marker.pose.position.y = ry + target_local.y;
  target_marker.pose.position.z = 0.1;
  target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.1;
  target_marker.color.g = 1.0;
  target_marker.color.a = 1.0;

  marker_pub1_.publish(target_marker);
}

// ===== getRobotPose 可保留（但未使用）=====
bool EdgeFollower::getRobotPose(geometry_msgs::PoseStamped &pose)
{
  try
  {
    geometry_msgs::TransformStamped transform =
        tf_buffer_.lookupTransform("odom", "base_footprint", ros::Time(0), ros::Duration(0.1));
    pose.header.frame_id = "odom";
    pose.header.stamp = transform.header.stamp;
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