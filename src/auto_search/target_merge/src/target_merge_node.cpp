#include <ros/ros.h>
#include "target_merge/target_merge.h"

using namespace target_merge;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "target_merge_node");
  Target_Merge tm;
  tm.init();
  ros::spin();
  return 0;
}