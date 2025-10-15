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
  marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/edge_following_debug2", 1);
  traj_pub_ = nh.advertise<nav_msgs::Path>("/edge_following_trajectory", 1);
  temp_traj_pub_ = nh.advertise<nav_msgs::Path>("/edge_following_temp_trajectory", 1);
}

// ===== 激光回调 =====
void EdgeFollower::laserCallback(const sensor_msgs::LaserScanConstPtr &scan)
{
  auto local_traj = generateLocalTrajectory(*scan, "odom");
  generateCompareLocalTrajectory(*scan, "odom");            //  生成对比轨迹
  // return;  // TODO

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

    publishVisualization(segments, target_local, robot_pose, scan->header.stamp);
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

void EdgeFollower::publishVisualization(
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
  marker_pub_.publish(marker_array);
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

std::vector<cv::Point2f> EdgeFollower::extractContourPoints(const sensor_msgs::LaserScan &scan)
{
  constexpr float MIN_DIST = 0.2f;
  constexpr float MAX_DIST = 1.2f;
  constexpr float ANGLE_HALF = 65.0f * M_PI / 180.0f;

  std::vector<std::pair<float, cv::Point2f>> angle_point_pairs;

  for (size_t i = 0; i < scan.ranges.size(); ++i)
  {
    float range = scan.ranges[i];
    if (std::isnan(range) || std::isinf(range))
      continue;

    float angle = scan.angle_min + i * scan.angle_increment;
    if (std::abs(angle) > ANGLE_HALF)
      continue;
    if (range < MIN_DIST || range > MAX_DIST)
      continue;

    float x = range * std::cos(angle);
    float y = range * std::sin(angle);

    // 注意：我们希望点列从左到右（y递增）或从右到左（y递减）
    // 但为了统一，我们按 x 排序（从左到右）
    // 或者更稳健地：按角度排序，然后反转如果需要
    angle_point_pairs.emplace_back(angle, cv::Point2f(x, y));
  }

  if (angle_point_pairs.empty())
    return {};

  // 按角度排序（从小到大：从右到左）
  std::sort(angle_point_pairs.begin(), angle_point_pairs.end(),
            [](const auto &a, const auto &b)
            { return a.first < b.first; });

  // 提取点列（此时是右→左）
  std::vector<cv::Point2f> points;
  points.reserve(angle_point_pairs.size());
  for (const auto &ap : angle_point_pairs)
    points.push_back(ap.second);

  // >>>>>>>> 新增：反转点列，使从左到右 <<<<<<<<
  // 如果你想让点列从左到右（x递增），则反转
  std::reverse(points.begin(), points.end());

  return points;
}

// 简单移动平均平滑
std::vector<cv::Point2f> smoothPath(const std::vector<cv::Point2f> &path, int window = 3)
{
  if (path.size() <= window)
    return path;

  std::vector<cv::Point2f> smoothed;
  smoothed.reserve(path.size());

  for (size_t i = 0; i < path.size(); ++i)
  {
    cv::Point2f sum(0, 0);
    int count = 0;
    int start = std::max(0, static_cast<int>(i) - window / 2);
    int end = std::min(static_cast<int>(path.size()) - 1, static_cast<int>(i) + window / 2);
    for (int j = start; j <= end; ++j)
    {
      sum += path[j];
      count++;
    }
    smoothed.push_back(sum / count);
  }
  return smoothed;
}

// 按弧长重采样
std::vector<cv::Point2f> resamplePath(const std::vector<cv::Point2f> &path, float interval = 0.1f)
{
  if (path.size() < 2)
    return path;

  std::vector<float> cumdist;
  cumdist.push_back(0.0f);
  float total = 0.0f;
  for (size_t i = 1; i < path.size(); ++i)
  {
    float d = cv::norm(path[i] - path[i - 1]);
    total += d;
    cumdist.push_back(total);
  }

  if (total < interval)
    return {path.front()};

  std::vector<cv::Point2f> resampled;
  resampled.push_back(path.front());

  float current = interval;
  size_t idx = 0;
  while (current < total && resampled.size() < 20) // 最多20个点（2m）
  {
    while (idx < cumdist.size() - 1 && cumdist[idx + 1] < current)
      idx++;

    if (idx >= cumdist.size() - 1)
      break;

    float t = (current - cumdist[idx]) / (cumdist[idx + 1] - cumdist[idx]);
    cv::Point2f p = path[idx] + t * (path[idx + 1] - path[idx]);
    resampled.push_back(p);
    current += interval;
  }

  return resampled;
}

std::vector<cv::Point2f> EdgeFollower::offsetPoints(
  const std::vector<cv::Point2f>& points,
  float robot_radius)
{
  if (points.size() < 3) return points;

  // 计算平均点间距
  float avg_distance = 0.0f;
  for (size_t i = 1; i < points.size(); ++i) {
      float d = cv::norm(points[i] - points[i-1]);
      avg_distance += d;
  }
  avg_distance /= std::max(1.0f, static_cast<float>(points.size() - 1));

  // 设置窗口大小：确保覆盖 ~1.5m 物理长度
  int window_size = std::min(15, static_cast<int>(std::ceil(1.5 / avg_distance)));
  int half_window = window_size / 2;

  std::vector<cv::Point2f> offset_path;
  offset_path.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i) {
      cv::Point2f p = points[i];

      // 获取邻域点
      std::vector<cv::Point2f> neighbors;
      int start = std::max(0, static_cast<int>(i) - half_window);
      int end = std::min(static_cast<int>(points.size()) - 1, static_cast<int>(i) + half_window);

      for (int j = start; j <= end; ++j) {
          neighbors.push_back(points[j]);
      }

      // PCA 拟合主方向
      cv::Mat data(neighbors.size(), 2, CV_32F);
      for (size_t k = 0; k < neighbors.size(); ++k) {
          data.at<float>(k, 0) = neighbors[k].x;
          data.at<float>(k, 1) = neighbors[k].y;
      }

      cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1);
      cv::Mat eigenvec = pca.eigenvectors.row(0);
      cv::Point2f tangent(eigenvec.at<float>(0), eigenvec.at<float>(1));
      float t_norm = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
      if (t_norm < 1e-4) {
          offset_path.push_back(p);
          continue;
      }
      tangent.x /= t_norm;
      tangent.y /= t_norm;

      // 法向方向
      cv::Point2f normal_left(-tangent.y, tangent.x);
      cv::Point2f normal_right(tangent.y, -tangent.x);

      // 根据 follow_left_ 选择偏移方向
      cv::Point2f offset_dir;
      if (follow_left_) {
          offset_dir = normal_left;
      } else {
          offset_dir = normal_right;
      }

      cv::Point2f offset_point = p + offset_dir * robot_radius;
      offset_path.push_back(offset_point);
  }

  return offset_path;
}

nav_msgs::Path EdgeFollower::generateLocalTrajectory(const sensor_msgs::LaserScan &scan,
                                                     const std::string &global_frame)
{
  nav_msgs::Path path;
  path.header.frame_id = global_frame;
  path.header.stamp = ros::Time::now();

  // 1. 提取原始轮廓点（局部坐标）
  auto raw_points = extractContourPoints(scan);
  if (raw_points.empty())
  {
    traj_pub_.publish(path); // 发布空路径
    return path;
  }

  float robot_radius = 0.3f; // 假设机器人的半径为0.3米
  auto offset_points = offsetPoints(raw_points, robot_radius);

  // 2. 平滑 + 重采样
  auto smoothed = smoothPath(offset_points, 3);
  auto resampled = resamplePath(smoothed, 0.1f); // 0.1m 间隔

  if (resampled.empty())
  {
    traj_pub_.publish(path);
    return path;
  }

  // 3. 转为 ROS Path（全局坐标）
  geometry_msgs::PoseStamped robot_pose;
  auto result = getRobotPose(robot_pose);
  if (!result)
  {
    traj_pub_.publish(path); // 发布空路径
    return path;
  }
  tf2::Transform robot_tf;
  tf2::fromMsg(robot_pose.pose, robot_tf);
  for (const auto &p_local : resampled)
  {
    tf2::Vector3 v_local(p_local.x, p_local.y, 0);
    tf2::Vector3 v_global = robot_tf * v_local;

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = v_global.x();
    pose.pose.position.y = v_global.y();
    pose.pose.orientation.w = 1.0; // 无朝向
    path.poses.push_back(pose);
  }

  // 4. 发布可视化
  traj_pub_.publish(path);

  return path;
}

nav_msgs::Path EdgeFollower::generateCompareLocalTrajectory(const sensor_msgs::LaserScan &scan,
                                                            const std::string &global_frame)
{
  nav_msgs::Path path;
  path.header.frame_id = global_frame;
  path.header.stamp = ros::Time::now();

  // 1. 提取原始轮廓点（局部坐标）
  auto raw_points = extractContourPoints(scan);
  if (raw_points.empty())
  {
    temp_traj_pub_.publish(path); // 发布空路径
    return path;
  }

  // 2. 平滑 + 重采样
  auto smoothed = smoothPath(raw_points, 3);
  auto resampled = resamplePath(smoothed, 0.1f); // 0.1m 间隔

  if (resampled.empty())
  {
    temp_traj_pub_.publish(path);
    return path;
  }

  // 3. 转为 ROS Path（全局坐标）
  geometry_msgs::PoseStamped robot_pose;
  auto result = getRobotPose(robot_pose);
  if (!result)
  {
    temp_traj_pub_.publish(path); // 发布空路径
    return path;
  }
  tf2::Transform robot_tf;
  tf2::fromMsg(robot_pose.pose, robot_tf);
  for (const auto &p_local : resampled)
  {
    tf2::Vector3 v_local(p_local.x, p_local.y, 0);
    tf2::Vector3 v_global = robot_tf * v_local;

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = v_global.x();
    pose.pose.position.y = v_global.y();
    pose.pose.orientation.w = 1.0; // 无朝向
    path.poses.push_back(pose);
  }

  // 4. 发布可视化
  temp_traj_pub_.publish(path);

  return path;
}