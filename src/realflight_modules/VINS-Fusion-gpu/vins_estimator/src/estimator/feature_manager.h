/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "../utility/tic_toc.h"

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        is_stereo = false;
    }
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    bool is_stereo;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    int start_frame_local;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }
    FeaturePerId(int _feature_id, int _start_frame, int _start_frame_local)
    : feature_id(_feature_id), start_frame(_start_frame), start_frame_local(_start_frame_local),
      used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame_local();
    int endFrame_global(const vector<int> &global_frame_idx);
};

class FeatureManager
{
  public:
    //FeatureManager(Matrix3d _Rs[]);

    void init(int _frontId, bool _is_stereo, Matrix3d _Rs[]);
    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    int fromGlobalIndex2Local(const vector<int> &local_window, int global_index);
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td, const vector<int> &global_frame_idx);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], const vector<int> &global_frame_idx);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], const vector<int> &global_frame_idx);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P, const vector<int> &updated_global_frame_idx);
    void removeBack(const vector<int> &updated_global_frame_idx);
    void removeBackNoMerge();
    void removeFront(int merge_frame_idx_global, int merge_frame_idx_local);
    void removeFrontNoMerge(int merge_frame_idx);
    void removeOutlier(set<int> &outlierIndex);
    list<FeaturePerId> feature;
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;//当前帧追踪到的连续特征点（被追踪到4帧以上）数量
    bool is_stereo;
    int cam_num;
    int frontId;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count, const vector<int> &global_frame_idx);
    const Matrix3d *Rs;
    Matrix3d ric[2];
};

#endif