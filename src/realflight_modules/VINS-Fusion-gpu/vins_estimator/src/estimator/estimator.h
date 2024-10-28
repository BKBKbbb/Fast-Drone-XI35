/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(int frontId, double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(int frontId, double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const vector<pair<int, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> &images, const double header);
    void processMeasurements();
    //adjust multicamera
    pair<bool, int> isFeatureBufAvailable();
    bool isFeatureBufAvailable();
    int getFrontIdFromBuf();

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew(set<int> mergeFrontId, vector<int> local_idx);
    void slideWindowOld(set<int> mergeFrontId);
    vector<int> getLastFrontId();
    void multiCamIdxPushBack(vector<int> add_id, int add_frame_idx);
    vector<int> multiCamIdxSlideOld();
    pair<vector<int>, vector<int>> multiCamIdxSlideNew(int merge_frame_idx);
    int getExParamIdx(int frontId) const;
    int getWindowSize(int frontId) const;
    int fromGlobalIndex2Local(const vector<int> &local_window, int global_index);
    void printMultiCamIdx();
    void optimization(const vector<int> &frontId);
    void vector2double(const vector<int> &frontId);
    void double2vector(const vector<int> &frontId);
    bool failureDetection(int frontId);
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame(int frontId);
    void outliersRejection(int frontId, set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates(int latest_frontId);
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    void frontStateInit();
    vector<int> getGoodFrontId();
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    //记录前端的状态信息
    struct FrontState
    {
        FrontState(bool _is_stereo):is_stereo(_is_stereo)
        {
            if(_is_stereo)
                usable = true;//双目初始化为可用
            else
                usable = false;
            initialized = false;
        }
        void reset()
        {
            usable = false;
            if(!is_stereo)
                initialized = false;
        }
        bool is_stereo;
        bool usable;
        bool initialized;//单目是否完成初始化
    };
    vector<FrontState> frontStateVec;
    vector<double> lastProcessTime;//记录各前端上一次被处理的时间
    int usable2InitializedThresh;//可用到完成初始化的连续特征点数量阈值
    int resetThresh;//重置前端的连续特征点数量阈值

    std::mutex mBuf;//多个前端共用一个mutex
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    vector<queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > > featureBuf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    vector<FeatureTracker> featureTracker;//FRONTEND_NUM个前端

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[4];//预留到4个
    Vector3d tic[4];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    
    vector<double> td;
    
    //多前端索引映射矩阵
    vector<vector<int>> multicam_frame_idx;
    //同步帧索引矩阵 
    vector<vector<int>> sync_frame_idx;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;
    vector<int> inputImageCntPerCam;//记录接收的图像帧数
    float sum_t_feature;
    int begin_time_count;

    vector<FeatureManager> f_manager;//FRONTEND_NUM个特征点管理器
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[3][NUM_OF_F][SIZE_FEATURE];//支持3个前端
    double para_Ex_Pose[4][SIZE_POSE];//1个双目+2个单目
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[3][1];//支持3个前端
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    int mergeNewFrameIdx;//边缘化次新帧的全局id

    double first_image_time[3];
    bool first_image_flag[3] = {false};

};
