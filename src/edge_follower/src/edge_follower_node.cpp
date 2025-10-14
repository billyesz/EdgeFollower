#include "edge_follower/edge_follower.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "edge_follower_node");
  ros::NodeHandle nh;
  EdgeFollower follower(nh);
  ros::spin();
  return 0;
}