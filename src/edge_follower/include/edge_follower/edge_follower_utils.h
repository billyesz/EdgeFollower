#ifndef FLSLAM_SRC_EDGE_FOLLOWER_INCLUDE_EDGE_FOLLOWER_EDGE_FOLLOWER_UTILS_H_
#define FLSLAM_SRC_EDGE_FOLLOWER_INCLUDE_EDGE_FOLLOWER_EDGE_FOLLOWER_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

// 假设 'points' 是从 extractContourPoints 函数中获得的原始轮廓点集
std::vector<cv::Point2f> offsetPoints(const std::vector<cv::Point2f>& points, float robot_radius);



#endif // FLSLAM_SRC_EDGE_FOLLOWER_INCLUDE_EDGE_FOLLOWER_EDGE_FOLLOWER_UTILS_H_