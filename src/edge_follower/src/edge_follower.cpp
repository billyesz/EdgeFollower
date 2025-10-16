#include "edge_follower/edge_follower.h"
#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/imgproc.hpp>
#include <tf/transform_datatypes.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>

// ====== 新增：Eigen Spline 支持 ======
#include <unsupported/Eigen/Splines>
typedef Eigen::Spline<float, 2> Spline2d;
typedef Eigen::Matrix<float, 2, Eigen::Dynamic> PointsMatrix;

std::vector<float> computeCurvatures(const std::vector<cv::Point2f> &path)
{
  std::vector<float> curvatures(path.size(), 0.0f);
  if (path.size() < 3)
    return curvatures;

  for (size_t i = 1; i < path.size() - 1; ++i)
  {
    cv::Point2f p0 = path[i - 1];
    cv::Point2f p1 = path[i];
    cv::Point2f p2 = path[i + 1];

    float dx1 = p1.x - p0.x, dy1 = p1.y - p0.y;
    float dx2 = p2.x - p1.x, dy2 = p2.y - p1.y;

    float ds1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    float ds2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

    if (ds1 < 1e-6 || ds2 < 1e-6)
    {
      curvatures[i] = 0.0f;
      continue;
    }

    // 单位切向量
    cv::Point2f t1(dx1 / ds1, dy1 / ds1);
    cv::Point2f t2(dx2 / ds2, dy2 / ds2);

    // 曲率近似：|dT/ds|
    float dtx = t2.x - t1.x;
    float dty = t2.y - t1.y;
    float dT = std::sqrt(dtx * dtx + dty * dty);
    float ds = (ds1 + ds2) / 2.0f;

    curvatures[i] = (ds > 1e-6) ? (dT / ds) : 0.0f;
  }
  // 首尾点曲率设为邻近值
  if (curvatures.size() > 1)
  {
    curvatures[0] = curvatures[1];
    curvatures.back() = curvatures[curvatures.size() - 2];
  }
  return curvatures;
}

std::vector<cv::Point2f> limitCurvature(const std::vector<cv::Point2f> &path, float max_curvature, int max_iter = 3)
{
  if (path.size() < 3)
    return path;
  auto smoothed = path;

  for (int iter = 0; iter < max_iter; ++iter)
  {
    auto curvatures = computeCurvatures(smoothed);
    bool within_limit = true;
    for (float k : curvatures)
    {
      if (std::abs(k) > max_curvature)
      {
        within_limit = false;
        break;
      }
    }
    if (within_limit)
      break;

    // 局部高斯平滑（窗口=3）
    std::vector<cv::Point2f> new_path = smoothed;
    for (size_t i = 1; i < smoothed.size() - 1; ++i)
    {
      if (std::abs(curvatures[i]) > max_curvature)
      {
        new_path[i].x = (smoothed[i - 1].x + smoothed[i].x + smoothed[i + 1].x) / 3.0f;
        new_path[i].y = (smoothed[i - 1].y + smoothed[i].y + smoothed[i + 1].y) / 3.0f;
      }
    }
    smoothed = new_path;
  }
  return smoothed;
}

std::vector<cv::Point2f> filterSidePoints(
    const std::vector<cv::Point2f> &points,
    bool follow_left)
{
  if (points.empty())
    return {};

  std::vector<std::vector<cv::Point2f>> segments;
  std::vector<cv::Point2f> current_seg;

  for (const auto &p : points)
  {
    bool is_target_side = follow_left ? (p.y > 0) : (p.y < 0);
    if (is_target_side)
    {
      current_seg.push_back(p);
    }
    else
    {
      if (!current_seg.empty())
      {
        segments.push_back(current_seg);
        current_seg.clear();
      }
    }
  }
  if (!current_seg.empty())
  {
    segments.push_back(current_seg);
  }

  // 选择最长段
  size_t max_len = 0;
  size_t best_idx = 0;
  for (size_t i = 0; i < segments.size(); ++i)
  {
    if (segments[i].size() > max_len)
    {
      max_len = segments[i].size();
      best_idx = i;
    }
  }
  ROS_WARN("max_len : %zu, best_idx: %zu, segments.size(): %zu", max_len, best_idx, segments.size());
  return (max_len >= 2) ? segments[best_idx] : std::vector<cv::Point2f>();
}

EdgeFollower::EdgeFollower(ros::NodeHandle &nh)
    : tf_listener_(tf_buffer_),
      map_size_px_(static_cast<int>(map_size_m_ / resolution_))
{
  local_map_ = cv::Mat::zeros(map_size_px_, map_size_px_, CV_8UC1);
  laser_sub_ = nh.subscribe("scan", 1, &EdgeFollower::laserCallback, this);
  traj_pub_ = nh.advertise<nav_msgs::Path>("/edge_following_trajectory", 1);
  fitted_contour_pub_ = nh.advertise<sensor_msgs::PointCloud>("/fitted_contour", 1);
}

// ===== 激光回调 =====
void EdgeFollower::laserCallback(const sensor_msgs::LaserScanConstPtr &scan)
{
  auto local_traj = generateLocalTrajectory(*scan, "odom");
  // 不再做其他操作
}

// ===== getRobotPose =====
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

// ===== 提取轮廓点 （局部坐标）=====
std::vector<cv::Point2f> EdgeFollower::extractContourPoints(const sensor_msgs::LaserScan &scan)
{
  constexpr float MIN_DIST = 0.2f;
  constexpr float MAX_DIST = 1.5f; // 修改为 1.5m
  constexpr float ANGLE_HALF = 65.0f * M_PI / 180.0f;
  std::vector<std::pair<float, cv::Point2f>> angle_point_pairs;

  // ROS_INFO("scan.ranges.size : %zu", scan.ranges.size());
  for (size_t i = 0; i < scan.ranges.size(); ++i)
  {
    float range = scan.ranges[i];
    if (std::isnan(range) || std::isinf(range))
      continue;
    float angle = scan.angle_min + i * scan.angle_increment;
    float angle_deg = angle * 180.0f / M_PI;
    if (std::abs(angle) > ANGLE_HALF) // FOV: -65° ~ +65°
      continue;
    if (range < MIN_DIST || range > MAX_DIST)
      continue;
    float x = range * std::cos(angle);
    float y = range * std::sin(angle);
    // ROS_INFO("[%zu] (%f, %f) @ angle %f  angle_deg %f  scan.angle_min %f  scan.angle_increment %f", i, x, y, angle, angle_deg, scan.angle_min, scan.angle_increment);
    angle_point_pairs.emplace_back(angle, cv::Point2f(x, y));
  }

  if (angle_point_pairs.empty())
    return {};

  // 按角度排序，即从-65°到65°依次排序（右→左）
  std::sort(angle_point_pairs.begin(), angle_point_pairs.end(),
            [](const auto &a, const auto &b)
            { return a.first < b.first; });

  std::vector<cv::Point2f> points;
  points.reserve(angle_point_pairs.size());
  for (const auto &ap : angle_point_pairs)
  {
    points.push_back(ap.second);
  }

  // points 永远是从右→左（y 递增），针对不同的场景，后续对点的顺序需要调整，否则路径方向会出错
  // 比如：follow_left_ = true 时，激光点从右→左，路径方向就会反了
  // follow_left_ = false 时，激光点从右→左，路径方向就是对的

  // std::reverse(points.begin(), points.end());
  return points;
}

// ===== 使用 Eigen Spline 拟合光滑轮廓 =====
// 作用：将原始离散的墙轮廓点 raw_points 拟合成一条平滑的三次样条曲线，并在曲线上均匀重采样 num_samples + 1 个点，用于后续轨迹生成。
std::vector<cv::Point2f> EdgeFollower::fitSplineContour(const std::vector<cv::Point2f> &raw_points, int num_samples)
{
  // 关键修复：三次样条至少需要 4 个点
  if (raw_points.size() < 4)
  {
    // 点太少：直接返回原始点（或线性插值）
    if (raw_points.size() <= 1)
      return raw_points;

    // 线性重采样（2~3个点）
    std::vector<cv::Point2f> linear;
    linear.reserve(num_samples + 1);
    for (int i = 0; i <= num_samples; ++i)
    {
      float t = static_cast<float>(i) / num_samples;
      if (raw_points.size() == 2)
      {
        cv::Point2f p = raw_points[0] + t * (raw_points[1] - raw_points[0]);
        linear.push_back(p);
      }
      else if (raw_points.size() == 3)
      {
        // 分段线性：0→1→2
        if (t <= 0.5f)
        {
          cv::Point2f p = raw_points[0] + (t * 2.0f) * (raw_points[1] - raw_points[0]);
          linear.push_back(p);
        }
        else
        {
          cv::Point2f p = raw_points[1] + ((t - 0.5f) * 2.0f) * (raw_points[2] - raw_points[1]);
          linear.push_back(p);
        }
      }
    }
    return linear;
  }

  // 原有样条拟合逻辑（仅当点数 >=4 时执行）
  std::vector<float> u;
  u.push_back(0.0f);
  float total = 0.0f;
  for (size_t i = 1; i < raw_points.size(); ++i)
  {
    total += cv::norm(raw_points[i] - raw_points[i - 1]);
    u.push_back(total);
  }
  if (total < 1e-6f)
    return raw_points; // 所有点重合

  for (auto &val : u)
    val /= total;

  PointsMatrix data(2, raw_points.size());
  for (size_t i = 0; i < raw_points.size(); ++i)
  {
    data(0, i) = raw_points[i].x;
    data(1, i) = raw_points[i].y;
  }

  // 现在安全：raw_points.size() >= 4
  Spline2d spline = Eigen::SplineFitting<Spline2d>::Interpolate(data, 3, Eigen::Map<Eigen::VectorXf>(u.data(), u.size()));

  std::vector<cv::Point2f> fitted;
  fitted.reserve(num_samples + 1);
  for (int i = 0; i <= num_samples; ++i)
  {
    float t = static_cast<float>(i) / num_samples;
    Eigen::Vector2f pt = spline(t);
    fitted.emplace_back(pt.x(), pt.y());
  }
  return fitted;
}

// ===== 偏移轮廓（保持不变）=====
std::vector<cv::Point2f> EdgeFollower::offsetPoints(
    const std::vector<cv::Point2f> &points,
    float robot_radius)
{
  if (points.size() < 3)
    return points;

  // 计算平均点间距，估计点云的空间密度
  // 这是自适应窗口的关键：点密 → 窗口小；点疏 → 窗口大
  float avg_distance = 0.0f;
  for (size_t i = 1; i < points.size(); ++i)
  {
    float d = cv::norm(points[i] - points[i - 1]);
    avg_distance += d;
  }
  avg_distance /= std::max(1.0f, static_cast<float>(points.size() - 1));

  // 动态计算 PCA 窗口大小：确保覆盖 ~1.5m 物理长度
  // 1.5m是一个经验长度，表示 PCA 应该在 1.5 米范围内拟合局部切线
  // ≤15是防止在极稀疏点云中窗口过大（如 avg_distance=0.01 → window=150，计算慢且过平滑）
  // half_window 用于以当前点为中心取邻域
  int window_size = std::min(15, static_cast<int>(std::ceil(1.5f / avg_distance)));
  int half_window = window_size / 2;

  std::vector<cv::Point2f> offset_path;
  offset_path.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i)
  {
    cv::Point2f p = points[i];
    std::vector<cv::Point2f> neighbors;
    int start = std::max(0, static_cast<int>(i) - half_window);
    int end = std::min(static_cast<int>(points.size()) - 1, static_cast<int>(i) + half_window);
    for (int j = start; j <= end; ++j)
    {
      neighbors.push_back(points[j]);
    }

    cv::Mat data(neighbors.size(), 2, CV_32F);
    for (size_t k = 0; k < neighbors.size(); ++k)
    {
      data.at<float>(k, 0) = neighbors[k].x;
      data.at<float>(k, 1) = neighbors[k].y;
    }
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1);
    cv::Mat eigenvec = pca.eigenvectors.row(0);
    cv::Point2f tangent(eigenvec.at<float>(0), eigenvec.at<float>(1));
    float t_norm = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    if (t_norm < 1e-4)
    {
      offset_path.push_back(p);
      continue;
    }
    tangent.x /= t_norm;
    tangent.y /= t_norm;

    cv::Point2f normal_left(-tangent.y, tangent.x);
    cv::Point2f normal_right(tangent.y, -tangent.x);

    cv::Point2f offset_dir = follow_left_ ? normal_right : normal_left;
    // cv::Point2f offset_dir = follow_left_ ? normal_left : normal_right;
    cv::Point2f offset_point = p + offset_dir * robot_radius;
    offset_path.push_back(offset_point);
  }
  return offset_path;
}

// ===== 裁剪轨迹到前方 1.0m =====
std::vector<cv::Point2f> cropTrajectoryToHorizon(const std::vector<cv::Point2f> &path, float horizon = 1.0f)
{
  if (path.empty())
    return path;
  std::vector<cv::Point2f> cropped;
  cropped.push_back(path[0]);
  float dist = 0.0f;
  for (size_t i = 1; i < path.size(); ++i)
  {
    float d = cv::norm(path[i] - path[i - 1]);
    if (dist + d > horizon)
      break;
    dist += d;
    cropped.push_back(path[i]);
  }
  return cropped;
}

// ===== 生成局部轨迹（主逻辑）=====
nav_msgs::Path EdgeFollower::generateLocalTrajectory(const sensor_msgs::LaserScan &scan,
                                                     const std::string &global_frame)
{
  nav_msgs::Path path;
  path.header.frame_id = global_frame;
  path.header.stamp = ros::Time::now();

  auto raw_points = extractContourPoints(scan);

  auto filter_points = filterSidePoints(raw_points, follow_left_);

  if (raw_points.size() < 2)
  { // 至少两个点才有意义
    traj_pub_.publish(path);
    return path;
  }

  // ===== 关键修改：使用 Spline 拟合代替 smoothPath + resamplePath =====
  auto fitted_contour = fitSplineContour(filter_points, 40); // 40 个采样点
  publishPointCloud(fitted_contour, "base_footprint", ros::Time::now(), fitted_contour_pub_);

  float robot_radius = 0.25f; // 根据你的参数
  auto offset_points = offsetPoints(fitted_contour, robot_radius);

  // ===== 新增：曲率限制 =====
  const float max_curvature = 3.33f; // 对应 R_min = 0.3m
  auto curvature_limited = limitCurvature(offset_points, max_curvature);

  // 裁剪到前方 1.0m
  auto final_points = cropTrajectoryToHorizon(curvature_limited, 1.0f);

  if (final_points.empty())
  {
    traj_pub_.publish(path);
    return path;
  }

  // 转为全局坐标
  geometry_msgs::PoseStamped robot_pose;
  if (!getRobotPose(robot_pose))
  {
    traj_pub_.publish(path);
    return path;
  }

  tf2::Transform robot_tf;
  tf2::fromMsg(robot_pose.pose, robot_tf);
  for (const auto &p_local : final_points)
  {
    tf2::Vector3 v_local(p_local.x, p_local.y, 0);
    tf2::Vector3 v_global = robot_tf * v_local;
    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = v_global.x();
    pose.pose.position.y = v_global.y();
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }

  traj_pub_.publish(path);
  return path;
}

void EdgeFollower::publishPointCloud(const std::vector<cv::Point2f> &points,
                                     const std::string &frame_id,
                                     const ros::Time &stamp,
                                     ros::Publisher &pub)
{
  if (points.empty())
    return; // 新增：空则不发布

  sensor_msgs::PointCloud cloud;
  cloud.header.frame_id = frame_id;
  cloud.header.stamp = stamp;
  cloud.points.reserve(points.size());
  for (const auto &p : points)
  {
    geometry_msgs::Point32 pt;
    pt.x = p.x;
    pt.y = p.y;
    pt.z = 0.0f;
    cloud.points.push_back(pt);
  }
  pub.publish(cloud);
}