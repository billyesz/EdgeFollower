#include "edge_follower/edge_follower_utils.h"



// 假设 'points' 是从 extractContourPoints 函数中获得的原始轮廓点集
std::vector<cv::Point2f> offsetPoints(const std::vector<cv::Point2f>& points, float robot_radius)
{
    std::vector<cv::Point2f> offset_path;
    if (points.size() < 2) return points; // 如果少于两个点，则直接返回

    for (size_t i = 0; i < points.size(); ++i)
    {
        cv::Point2f current_point = points[i];
        cv::Point2f prev_point = i > 0 ? points[i-1] : current_point;
        cv::Point2f next_point = i < points.size()-1 ? points[i+1] : current_point;

        // 计算当前点的切线向量（next - prev）
        cv::Point2f tangent = next_point - prev_point;

        // 法线向量（逆时针旋转90度）并归一化
        cv::Point2f normal(-tangent.y, tangent.x);
        float length = std::sqrt(normal.x * normal.x + normal.y * normal.y);
        if (length > 0) // 防止除零错误
        {
            normal.x /= length;
            normal.y /= length;
        }

        // 根据机器人半径偏移
        cv::Point2f offset_point = current_point + normal * robot_radius;
        offset_path.push_back(offset_point);
    }

    return offset_path;
}