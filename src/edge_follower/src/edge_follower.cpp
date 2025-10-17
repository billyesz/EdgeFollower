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
#include <queue>
#include <algorithm>

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
  return (max_len >= 2) ? segments[best_idx] : std::vector<cv::Point2f>();
}

// 辅助函数：计算两点距离
inline float dist2D(const cv::Point2f &a, const cv::Point2f &b)
{
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// 辅助函数：计算两个点集之间的最小距离
float minDistanceBetweenClusters(const std::vector<cv::Point2f> &A,
                                 const std::vector<cv::Point2f> &B)
{
  float min_dist = std::numeric_limits<float>::max();
  for (const auto &a : A)
  {
    for (const auto &b : B)
    {
      float d = dist2D(a, b);
      if (d < min_dist)
        min_dist = d;
    }
  }
  return min_dist;
}

// DBSCAN 聚类（简化版，适合小规模点云）
std::vector<std::vector<cv::Point2f>> dbscanClustering(
    const std::vector<cv::Point2f> &points,
    float eps,
    int min_pts)
{
  int n = points.size();
  std::vector<int> labels(n, -1); // -1: noise, >=0: cluster id
  int cluster_id = 0;

  for (int i = 0; i < n; ++i)
  {
    if (labels[i] != -1)
      continue; // already processed

    // Find neighbors
    std::vector<int> neighbors;
    for (int j = 0; j < n; ++j)
    {
      if (dist2D(points[i], points[j]) <= eps)
      {
        neighbors.push_back(j);
      }
    }

    if (static_cast<int>(neighbors.size()) < min_pts)
    {
      labels[i] = -2; // mark as noise (optional)
      continue;
    }

    // Start new cluster
    labels[i] = cluster_id;
    std::queue<int> seeds;
    for (int idx : neighbors)
    {
      if (labels[idx] == -1 || labels[idx] == -2)
      {
        labels[idx] = cluster_id;
        seeds.push(idx);
      }
    }

    while (!seeds.empty())
    {
      int current = seeds.front();
      seeds.pop();

      std::vector<int> current_neighbors;
      for (int j = 0; j < n; ++j)
      {
        if (dist2D(points[current], points[j]) <= eps)
        {
          current_neighbors.push_back(j);
        }
      }

      if (static_cast<int>(current_neighbors.size()) >= min_pts)
      {
        for (int idx : current_neighbors)
        {
          if (labels[idx] == -1 || labels[idx] == -2)
          {
            labels[idx] = cluster_id;
            seeds.push(idx);
          }
        }
      }
    }
    cluster_id++;
  }

  // Build clusters
  std::vector<std::vector<cv::Point2f>> clusters(cluster_id);
  for (int i = 0; i < n; ++i)
  {
    if (labels[i] >= 0)
    {
      clusters[labels[i]].push_back(points[i]);
    }
  }

  ROS_INFO("DBSCAN found %zu clusters", clusters.size());

  return clusters;
}

// 线性插值：在相邻点间距超过 max_gap 时插入中间点
std::vector<cv::Point2f> interpolatePoints(
    const std::vector<cv::Point2f> &points,
    float max_gap = 0.1f) // 单位：米，例如 0.1m = 10cm
{
  if (points.size() <= 1)
  {
    return points;
  }

  std::vector<cv::Point2f> result;
  result.reserve(points.size() * 3); // 预留空间，避免频繁 realloc

  result.push_back(points[0]);

  for (size_t i = 1; i < points.size(); ++i)
  {
    const cv::Point2f &prev = result.back(); // 上一个已插入的点（可能是原点或插值点）
    const cv::Point2f &curr = points[i];

    float dx = curr.x - prev.x;
    float dy = curr.y - prev.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    // 如果两点距离超过 max_gap，进行插值
    if (dist > max_gap && dist > 1e-6f)
    {
      int num_segments = static_cast<int>(std::ceil(dist / max_gap));
      float step = 1.0f / num_segments;

      for (int s = 1; s < num_segments; ++s)
      {
        float t = s * step;
        cv::Point2f interp(
            prev.x + t * dx,
            prev.y + t * dy);
        result.push_back(interp);
      }
    }

    // 添加当前原始点
    result.push_back(curr);
  }

  return result;
}

std::vector<cv::Point2f> clusterAndMergeNearby(
    const std::vector<cv::Point2f> &points,
    float eps = 0.25f,           // DBSCAN 内部距离阈值
    int min_pts = 3,             // 最小点数
    float merge_threshold = 0.3f // 簇间合并距离
)
{
  if (points.size() < min_pts)
    return {};

  // Step 1: DBSCAN 聚类
  auto clusters = dbscanClustering(points, eps, min_pts);
  if (clusters.empty())
    return points;

  // Step 2: 合并邻近簇
  std::vector<std::vector<cv::Point2f>> merged_clusters;
  for (auto &c : clusters)
  {
    bool merged = false;
    for (auto &mc : merged_clusters)
    {
      if (minDistanceBetweenClusters(c, mc) <= merge_threshold)
      {
        mc.insert(mc.end(), c.begin(), c.end());
        merged = true;
        break;
      }
    }
    if (!merged)
    {
      merged_clusters.push_back(c);
    }
  }

  // Step 3: 选择主簇（目标侧 + 最近）
  std::vector<cv::Point2f> best_cluster;
  float best_score = -1.0f;

  for (const auto &cluster : merged_clusters)
  {
    if (cluster.size() < min_pts)
      continue;

    // 判断目标侧（假设 follow_left_ == true → y > 0）
    int target_side_count = 0;
    for (const auto &p : cluster)
    {
      if (p.y > 0)
        target_side_count++;
    }
    float target_ratio = static_cast<float>(target_side_count) / cluster.size();
    if (target_ratio < 0.6f)
      continue;

    // 找最小 x
    auto min_it = std::min_element(cluster.begin(), cluster.end(),
                                   [](const cv::Point2f &a, const cv::Point2f &b)
                                   {
                                     return a.x < b.x;
                                   });
    float min_x = min_it->x;

    float score = -min_x + 0.01f * static_cast<float>(cluster.size());
    if (score > best_score)
    {
      best_score = score;
      best_cluster = cluster;
    }
  }

  return best_cluster;
}

std::vector<cv::Point2f> clusterByLocalDensity(
    const std::vector<cv::Point2f> &points,
    float window_size = 0.5f, // 滑动窗口宽度
    float min_density = 0.1f  // 每米至少 0.1 个点
)
{
  if (points.empty())
    return {};

  // Step 1: 按 x 排序
  std::vector<cv::Point2f> sorted = points;
  std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f &a, const cv::Point2f &b)
            { return a.x < b.x; });

  std::vector<std::vector<cv::Point2f>> clusters;
  int n = sorted.size();

  for (int i = 0; i < n; ++i)
  {
    // 找到以 sorted[i] 为中心、x ∈ [x_i - window_size/2, x_i + window_size/2] 的所有点
    std::vector<cv::Point2f> window_points;
    for (int j = 0; j < n; ++j)
    {
      if (std::abs(sorted[j].x - sorted[i].x) <= window_size / 2)
      {
        window_points.push_back(sorted[j]);
      }
    }

    // 如果窗口内点数 ≥ 3，且密度达标，则视为一个簇
    float length = window_size;
    float density = static_cast<float>(window_points.size()) / length;
    if (window_points.size() >= 3 && density >= min_density)
    {
      // 合并所有点（避免重复）
      std::vector<cv::Point2f> unique_cluster;
      for (auto &p : window_points)
      {
        bool exists = false;
        for (auto &uc : unique_cluster)
        {
          if (dist2D(p, uc) < 0.05f)
          { // 去重
            exists = true;
            break;
          }
        }
        if (!exists)
          unique_cluster.push_back(p);
      }

      // 添加到集群列表
      bool already_added = false;
      for (auto &c : clusters)
      {
        if (c.size() > 0)
        {
          float dist_to_c = dist2D(unique_cluster[0], c[0]);
          if (dist_to_c < 0.1f)
          {
            c.insert(c.end(), unique_cluster.begin(), unique_cluster.end());
            already_added = true;
            break;
          }
        }
      }
      if (!already_added)
      {
        clusters.push_back(unique_cluster);
      }
    }
  }

  // Step 2: 选择主簇（目标侧 + 最近）
  std::vector<cv::Point2f> best_cluster;
  float best_score = -1.0f;

  for (const auto &cluster : clusters)
  {
    if (cluster.size() < 3)
      continue;

    // 判断目标侧（假设 follow_left_ == true → y > 0）
    int target_side_count = 0;
    for (const auto &p : cluster)
    {
      if (p.y > 0)
        target_side_count++;
    }
    float target_ratio = static_cast<float>(target_side_count) / cluster.size();
    if (target_ratio < 0.6f)
      continue;

    // 找最小 x
    auto min_it = std::min_element(cluster.begin(), cluster.end(),
                                   [](const cv::Point2f &a, const cv::Point2f &b)
                                   {
                                     return a.x < b.x;
                                   });
    float min_x = min_it->x;

    float score = -min_x + 0.01f * static_cast<float>(cluster.size());
    if (score > best_score)
    {
      best_score = score;
      best_cluster = cluster;
    }
  }

  return best_cluster;
}

std::vector<cv::Point2f> mergeNearbyClustersByDensity(
    const std::vector<cv::Point2f> &points,
    float max_distance = 0.3f,
    float min_density = 0.1f)
{
  if (points.empty())
    return {};

  // Step 1: 按 x 排序（假设机器人朝 +x）
  std::vector<cv::Point2f> sorted = points;
  std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f &a, const cv::Point2f &b)
            { return a.x < b.x; });

  // Step 2: 合并邻近点
  std::vector<std::vector<cv::Point2f>> clusters;
  std::vector<cv::Point2f> current;

  for (const auto &p : sorted)
  {
    if (current.empty())
    {
      current.push_back(p);
    }
    else
    {
      float dx = p.x - current.back().x;
      float dy = p.y - current.back().y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist <= max_distance)
      {
        current.push_back(p);
      }
      else
      {
        // 密度过滤：长度 > 0.1m 且密度达标
        float length = current.back().x - current.front().x;
        if (length > 0.1f && static_cast<float>(current.size()) / length >= min_density)
        {
          clusters.push_back(current);
        }
        current.clear();
        current.push_back(p);
      }
    }
  }

  // 添加最后一个簇
  if (!current.empty())
  {
    float length = current.back().x - current.front().x;
    if (length > 0.1f && static_cast<float>(current.size()) / length >= min_density)
    {
      clusters.push_back(current);
    }
  }

  // Step 3: 选择主簇（目标侧 + 最近）
  std::vector<cv::Point2f> best_cluster;
  float best_score = -1.0f;

  for (const auto &cluster : clusters)
  {
    if (cluster.size() < 3)
      continue;

    // 判断目标侧（假设 follow_left_ == true → y > 0）
    int target_side_count = 0;
    for (const auto &p : cluster)
    {
      if (p.y > 0)
        target_side_count++; // ← 请根据你的 follow_left_ 逻辑调整
    }
    float target_ratio = static_cast<float>(target_side_count) / cluster.size();
    if (target_ratio < 0.6f)
      continue;

    // 找最小 x（最近点）
    auto min_it = std::min_element(cluster.begin(), cluster.end(),
                                   [](const cv::Point2f &a, const cv::Point2f &b)
                                   {
                                     return a.x < b.x;
                                   });
    float min_x = min_it->x;

    float score = -min_x + 0.01f * static_cast<float>(cluster.size());
    if (score > best_score)
    {
      best_score = score;
      best_cluster = cluster;
    }
  }

  return best_cluster;
}

// 主函数：聚类 + 合并 + 选择主簇
std::vector<cv::Point2f> clusterAndSelectMain(
    const std::vector<cv::Point2f> &points,
    bool follow_left,
    float cluster_eps,
    int min_cluster_size,
    float merge_threshold)
{

  if (points.size() < static_cast<size_t>(min_cluster_size))
  {
    return points; // fallback
  }

  // Step 1: DBSCAN 聚类
  auto clusters = dbscanClustering(points, cluster_eps, min_cluster_size);
  if (clusters.empty())
  {
    return points;
  }

  // Step 2: 合并邻近簇
  std::vector<std::vector<cv::Point2f>> merged_clusters;
  for (auto cluster : clusters)
  {
    bool merged = false;
    for (auto &mc : merged_clusters)
    {
      if (minDistanceBetweenClusters(cluster, mc) <= merge_threshold)
      {
        mc.insert(mc.end(), cluster.begin(), cluster.end());
        merged = true;
        break;
      }
    }
    if (!merged)
    {
      merged_clusters.push_back(cluster);
    }
  }

  // Step 3: 选择主簇
  // 标准：1. 在目标侧；2. 离机器人最近（x 最小）；3. 点数最多
  std::vector<cv::Point2f> best_cluster;
  float best_score = -1.0f;

  for (const auto &cluster : merged_clusters)
  {
    if (cluster.size() < static_cast<size_t>(min_cluster_size))
      continue;

    // 判断是否在目标侧（简单策略：多数点满足）
    int target_side_count = 0;
    for (const auto &p : cluster)
    {
      bool is_target = follow_left ? (p.y > 0) : (p.y < 0);
      if (is_target)
        target_side_count++;
    }
    float target_ratio = static_cast<float>(target_side_count) / cluster.size();

    // 如果目标侧占比太低，跳过
    if (target_ratio < 0.6f)
      continue;

    // 计算最近点的 x（假设机器人在原点，朝 +x 方向）
    float min_x = std::numeric_limits<float>::max();
    for (const auto &p : cluster)
    {
      if (p.x < min_x)
        min_x = p.x;
    }

    // 打分：-min_x（越小越好） + 0.01 * size（辅助）
    float score = -min_x + 0.01f * static_cast<float>(cluster.size());

    if (score > best_score)
    {
      best_score = score;
      best_cluster = cluster;
    }
  }

  // 如果没找到合适的，返回原始点（fallback）
  if (best_cluster.empty())
  {
    return points;
  }

  return best_cluster;
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

  for (size_t i = 0; i < scan.ranges.size(); ++i)
  {
    float range = scan.ranges[i];
    if (std::isnan(range) || std::isinf(range))
      continue;
    float angle = scan.angle_min + i * scan.angle_increment;
    if (std::abs(angle) > ANGLE_HALF) // FOV: -65° ~ +65°
      continue;
    if (range < MIN_DIST || range > MAX_DIST)
      continue;
    float x = range * std::cos(angle);
    float y = range * std::sin(angle);
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

  // 过滤点，使每个点至少保持 0.05m 的间距
  constexpr float min_distance = 0.05f;
  cv::Point2f last_accepted_point(1e9, 1e9); // 初始化为一个很大的值，确保第一个点总是被接受

  for (const auto &ap : angle_point_pairs)
  {
    // points.push_back(ap.second);
    if (cv::norm(ap.second - last_accepted_point) >= min_distance)
    {
      points.push_back(ap.second);
      last_accepted_point = ap.second;
    }
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

// ===== 偏移轮廓 =====
std::vector<cv::Point2f> EdgeFollower::offsetPoints(
    const std::vector<cv::Point2f> &points,
    float offset_value)
{
  // PCA 需要至少 2 个点才能计算方向，但 2 个点时窗口可能退化。保守起见，少于 3 点直接返回
  if (points.size() < 3)
    return points;

  // 计算平均的点间距，估计点云的空间密度
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
    // 构建局部邻域（滑动窗口）
    // 注意：这是按索引取邻域，不是按距离！
    // 潜在问题：如果原始点序不连续（如跳变），邻域可能包含无关点
    std::vector<cv::Point2f> neighbors;
    int start = std::max(0, static_cast<int>(i) - half_window);
    int end = std::min(static_cast<int>(points.size()) - 1, static_cast<int>(i) + half_window);
    for (int j = start; j <= end; ++j)
    {
      neighbors.push_back(points[j]);
    }

    // 对局部邻域点集执行 PCA 获取局部切线方向
    cv::Mat data(neighbors.size(), 2, CV_32F);
    for (size_t k = 0; k < neighbors.size(); ++k)
    {
      data.at<float>(k, 0) = neighbors[k].x;
      data.at<float>(k, 1) = neighbors[k].y;
    }
    // PCA 原理：找数据方差最大的方向 → 局部切线方向
    // CV_PCA_DATA_AS_ROW：每行是一个点
    // 1：只保留第一主成分（切线），忽略法向（噪声）
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1);
    cv::Mat eigenvec = pca.eigenvectors.row(0);
    cv::Point2f tangent(eigenvec.at<float>(0), eigenvec.at<float>(1));
    // 切线归一化（得到只表示方向的单位向量tangent）
    float t_norm = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    if (t_norm < 1e-4)
    {
      // 如果所有邻居点都重合（比如激光打到一个点反复返回），PCA 会返回一个零向量 (0, 0)。
      // 长度是 0 → 无法归一化（除以 0 会崩溃）
      // 所以直接跳过偏移，保留原点
      offset_path.push_back(p);
      continue;
    }
    tangent.x /= t_norm;
    tangent.y /= t_norm;

    // 计算法向量（关键）
    // 在 2D 中，把一个向量 (x, y) 逆时针旋转 90°，得到 (-y, x)，顺时针旋转 90°，得到 (y, -x)
    cv::Point2f normal_left(-tangent.y, tangent.x);  // 逆时针90° → 左法向
    cv::Point2f normal_right(tangent.y, -tangent.x); // 顺时针90° → 右法向

    // 根据 follow_left_ 选择偏移方向，并生成偏移点
    cv::Point2f offset_dir = follow_left_ ? normal_right : normal_left;
    cv::Point2f offset_point = p + offset_dir * offset_value;
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
  // auto fitted_contour = fitSplineContour(filter_points, 40); // 40 个采样点
  // publishPointCloud(fitted_contour, "base_footprint", ros::Time::now(), fitted_contour_pub_);

  // auto main_cluster = mergeNearbyClustersByDensity(filter_points, 0.25f, 0.1f);
  // auto main_cluster = clusterAndSelectMain(filter_points, follow_left_, 0.4f, 3, 0.1f);
  // auto main_cluster = clusterByLocalDensity(filter_points, 0.5f, 0.1f);
  auto main_cluster = clusterAndMergeNearby(filter_points, 0.25f, 3, 0.3f);

  // 1. 排序：按 x 从大到小（近 → 远）
  std::vector<cv::Point2f> sorted_cluster = main_cluster;
  std::sort(sorted_cluster.begin(), sorted_cluster.end(),
            [](const cv::Point2f &a, const cv::Point2f &b)
            {
              return a.x > b.x; // 机器人前方 x 更大
            });

  // 2. 插值：填补大于 0.1m 的空隙
  // auto dense_cluster = interpolatePoints(sorted_cluster, 0.03f); // 每 10cm 至少一个点
  publishPointCloud(sorted_cluster, "base_footprint", ros::Time::now(), fitted_contour_pub_);

  float robot_radius = 0.25f; // 根据你的参数
  auto offset_points = offsetPoints(sorted_cluster, robot_radius);

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