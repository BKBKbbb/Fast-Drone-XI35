/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator()
{
    ROS_INFO("init begins");
    clearState();
    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    // sum_t_feature = 0.0;
    // begin_time_count = 10;
    initFirstPoseFlag = false;
}

void Estimator::setParameter()
{
    featureBuf = vector<queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > >(FRONTEND_NUM);
    featureTracker = vector<FeatureTracker>(FRONTEND_NUM);
    f_manager = vector<FeatureManager>(FRONTEND_NUM);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    //initialize f_manager and featureTracker
    for(int i = 0, idx = 0; i < FRONTEND_NUM; i++)
    {
        if(i < STEREO_NUM)
        {
            f_manager[i].init(i, true, Rs);
            f_manager[i].setRic(ric+idx);
            vector<string> cali_file = vector<string>(CAM_NAMES.begin() + idx, CAM_NAMES.begin() + idx + 2);
            featureTracker[i].readIntrinsicParameter(cali_file);
            idx +=  2;
        }
        else
        {
            f_manager[i].init(i, false, Rs);
            f_manager[i].setRic(ric+idx);
            vector<string> cali_file = vector<string>(CAM_NAMES.begin() + idx, CAM_NAMES.begin() + idx + 1);
            featureTracker[i].readIntrinsicParameter(cali_file);
            idx += 1;
        }
    }
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    key_poses = vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(WINDOW_SIZE+1);
    td = vector<double>(FRONTEND_NUM, TD);
    lastProcessTime = vector<double>(FRONTEND_NUM, 0);
    g = G;
    cout << "set g " << g.transpose() << endl;

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    //初始化frontState
    usable2InitializedThresh = INITIALIZED_THRESH;
    resetThresh = RESET_THRESH;
    frontStateInit();
    multicam_frame_idx = vector<vector<int>>(FRONTEND_NUM);

    if (MULTIPLE_THREAD)
    {
        processThread   = std::thread(&Estimator::processMeasurements, this);
    }
    inputImageCntPerCam = vector<int>(FRONTEND_NUM, 1);
}

void Estimator::frontStateInit()
{
    for(int i = 0; i < FRONTEND_NUM; i++)
    {
        if(i < STEREO_NUM)//双目初始为可用
            frontStateVec.emplace_back(true);
        else
            frontStateVec.emplace_back(false);
    }
}

int Estimator::fromGlobalIndex2Local(const vector<int> &local_window, int global_index)
{
    for(int i = 0; i < local_window.size(); i++)
    {
        if(local_window[i] == global_index)
            return i;
    }
    return -1;//not found
}

//TODO:控制featureFrame的push频率,目前是图像发布频率的一半
void Estimator::inputImage(int frontId, double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    if(frontStateVec[frontId].usable && !first_image_flag[frontId])
    {
        first_image_flag[frontId] = true;
        first_image_time[frontId] = t;
        return;
    }
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    if(_img1.empty())
        featureFrame = featureTracker[frontId].trackImage(t, _img);
    else
        featureFrame = featureTracker[frontId].trackImage(t, _img, _img1);
    
    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker[frontId].getTrackImage();
        pubTrackImage(frontId, imgTrack, t);
    }

    if(solver_flag == INITIAL)
    {
        if( !frontStateVec[frontId].is_stereo)//初始化阶段只处理双目帧
            return;
        else
        {
            if(inputImageCntPerCam[frontId] > 5 && featureTracker[frontId].getCurrentLongTrackFeatCnt() < USABLE_THRESH_FEATCNT)
                ROS_WARN("Feature track quality is Bad at initail stage!");
        }
    }
    else
    {
        if( !frontStateVec[frontId].usable)//若处于不可用，追踪特征点较多后才可用
        {
            if(featureTracker[frontId].getCurrentLongTrackFeatCnt() >= USABLE_THRESH_FEATCNT)
            {
                frontStateVec[frontId].usable = true;
                ROS_INFO("FrontEnd-%d is usable!", frontId);
            }
            else
            {
                first_image_time[frontId] = t;
                inputImageCntPerCam[frontId] = 1;//不可用状态下持续刷新first_image_time和inputImageCntPerCam，避免长时间不可用导致频率计算结果很小
                ROS_WARN("FrontEnd-%d is unusable due to bad feature quality!", frontId);
                return;
            }
        }
    }

    if(MULTIPLE_THREAD)  
    {     
        //控制加入到后端的频率
        int FREQ_CTRL = frontStateVec[frontId].is_stereo? STEREO_FREQ:MONO_FREQ;
        bool push_enable = false;
        ROS_ASSERT(t > first_image_time[frontId]);
        double cur_pub_freq = round(1.0 * inputImageCntPerCam[frontId] / (t - first_image_time[frontId]));
        ROS_INFO("FrontEnd-%d push frequency: %lf", frontId, cur_pub_freq);
        if(cur_pub_freq <= FREQ_CTRL)
        {
            push_enable = true;
            if(abs(1.0 * inputImageCntPerCam[frontId] / (t - first_image_time[frontId]) < 0.01 * FREQ_CTRL))
            {
                first_image_time[frontId] = t;
                inputImageCntPerCam[frontId] = 0;
            }
        }

        if(push_enable)
        {
            inputImageCntPerCam[frontId] ++;
            mBuf.lock();
            featureBuf[frontId].push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf[frontId].push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    fastPredictIMU(t, linearAcceleration, angularVelocity);
    if (solver_flag == NON_LINEAR)
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
}

void Estimator::inputFeature(int frontId, double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf[frontId].push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}
//return-<buf是否用待处理前端，时间戳最早的前端ID>
pair<bool, int> Estimator::isFeatureBufAvailable()
{
    pair<bool, int> res(false, 0);
    double earliest_time;
    unique_lock<mutex> lck(mBuf);
    for(int i = 0; i < FRONTEND_NUM; i++)
    {
        if(!featureBuf[i].empty())
        {
            double cur_time = featureBuf[i].front().first + td[i];
            if(!res.first)
            {//第一个不为空的buf
                res.first = true;
                earliest_time = cur_time;
                res.second = i;
            }
            else
            {
                if(cur_time < earliest_time)
                {
                    res.second = i;
                    earliest_time = cur_time;
                }
            }
        }
    }
    return res;
}
//确认是否满足调度条件
bool Estimator::isProcessAvailable()
{
    std::unique_lock<std::mutex> lck(mBuf);
    //只要有一个前端的buf达到2帧以上就满足调度条件
    for(auto vec : featureBuf)
    {
        if(vec.size() >= 3)
            return true;
    }
    return false;
}
//根据调度规则返回将被处理的前端id，并剔除被处理帧附近的帧
int Estimator::getFrontIdFromBuf()
{
    double earliest_time = 0;
    std::unique_lock<std::mutex> lck(mBuf);
    //找到最早的时间戳
    int idx = 0;
    for(auto f_vec : featureBuf)
    {
        if(!f_vec.empty())
        {
            if(earliest_time == 0)
                earliest_time = f_vec.front().first + td[idx];
            else
                earliest_time = std::min(f_vec.front().first + td[idx], earliest_time);
        }
        idx++;
    }
    //判断待处理的前端id
    double time_thresh = earliest_time + PROCESS_INTERVAL_THRESH;
    double max_interval = 0;
    std::unordered_map<double, int> interval_map;
    idx = 0;
    for(auto f_vec : featureBuf)
    {
        if(!f_vec.empty())
        {
            double cur_time = f_vec.front().first + td[idx];
            if(cur_time < time_thresh)
            {//该帧在处理间隔阈值内
                double process_interval = cur_time - lastProcessTime[idx];
                max_interval = std::max(max_interval, process_interval);
                interval_map[process_interval] = idx;
            }
        }
        idx++;
    }
    //将待处理帧附近的其他前端帧进行pop
    for(auto it = interval_map.begin(); it != interval_map.end(); it++)
    {
        if(it->first != max_interval)
            featureBuf[it->second].pop();
    }
    return interval_map[max_interval];
}
//adjust multi camera
void Estimator::processMeasurements()
{
    while (1)//500hz
    {
        //printf("process measurments\n");
        if(isProcessAvailable())
        {
            TicToc t_process;
            pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
            vector<pair<int, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > multiFeatures;//同步帧容器
            vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
            vector<int> syncFeatureIdx;//同步帧id
            int baseIdx = getFrontIdFromBuf();//待处理前端id
            ROS_DEBUG("Current baseIdx is %d", baseIdx);
            syncFeatureIdx.push_back(baseIdx);
            mBuf.lock();
            feature = std::move(featureBuf[baseIdx].front());
            featureBuf[baseIdx].pop();
            mBuf.unlock();
            multiFeatures.emplace_back(baseIdx, feature.second);

            double baseTime = feature.first;
            curTime = baseTime + td[baseIdx];
            if(curTime < prevTime)
            {//当前处理的帧时间戳小于上一时刻的，重置
                ROS_ERROR("processMeasurements fatal error: curTime %lf is less than preTime %lf! will throw current frame[frontId-%d]!", curTime, prevTime, baseIdx);
                continue;
            }
            while(1)
            {//等待imu覆盖基础帧
                if ((!USE_IMU  || IMUAvailable(curTime)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }

            mBuf.lock();
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);//初始化第一帧imu的姿态
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);//预积分
                }
            }
            ROS_INFO("Process image:%d, the timestamp is %lf", baseIdx, curTime);
            processImage(multiFeatures, baseTime);
            prevTime = curTime;
            lastProcessTime[baseIdx] = curTime;
            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(baseIdx, *this, header);
            pubPointCloud(baseIdx, *this, header);
            //pubKeyframe(baseIdx, *this);
            pubTF(*this, header);
            printf("process measurement time: %f\n", t_process.toc());
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

//初始化第一帧imu的姿态
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();
    for(int i = 0; i < FRONTEND_NUM; i++)
        f_manager[i].clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}
//检测各前端可用状态，返回能用于后端优化的前端id
vector<int> Estimator::getGoodFrontId()
{
    vector<int> res;
    for(int i = 0; i < FRONTEND_NUM; i++)
    {
        if(frontStateVec[i].is_stereo)
        {//双目
            if(frontStateVec[i].usable)
            {
                int featureCounts = f_manager[i].getFeatureCount();
                if(featureCounts < RESET_THRESH)
                {//连续特征点过少，重置为不可用
                    //f_manager[i].clearState();
                    frontStateVec[i].reset();
                    ROS_WARN("Front_%d's long-tracking feature %d is less than threshold %d, Reset it to unusable!", i, featureCounts, RESET_THRESH);
                }
                else
                    res.push_back(i);   
            }
        }
        else
        {//单目
            if(frontStateVec[i].usable)
            {
                if(!frontStateVec[i].initialized)
                {//未完成初始化
                    if(f_manager[i].getMonoFeatureInitialCount() >= usable2InitializedThresh)
                    {
                        frontStateVec[i].initialized = true;
                        ROS_INFO("Front-%d initialized successfully!", i);
                        res.push_back(i);
                    }
                }
                else
                {//已完成初始化，需要检查是否可用
                    int featureCounts = f_manager[i].getFeatureCount();
                    if(featureCounts < RESET_THRESH)
                    {//连续特征点过少，重置为不可用和未初始化
                        //f_manager[i].clearState();
                        frontStateVec[i].reset();

                        ROS_WARN("Front_%d's long-tracking feature %d is less than threshold %d, Reset it to unusable!", i, featureCounts, RESET_THRESH);
                    }
                    else
                        res.push_back(i);
                }
            }
        }
    }
    return res;
}

void Estimator::processImage(const vector<pair<int, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> &images, const double header)
{
    int frame_size = images.size();
    int baseFrameIdx = images[0].first;
    ROS_DEBUG("---------new image coming, sync frame'num is %d, the base frame is %d---------", frame_size, baseFrameIdx);

    vector<int> frontId;
    bool marge_old = false;
    for(auto &img : images)
    {
        frontId.push_back(img.first);
        //update the multicam_idx
        multicam_frame_idx[img.first].push_back(frame_count);
        //add feature and judge merge type
        marge_old |= f_manager[img.first].addFeatureCheckParallax(frame_count, img.second, td[img.first], multicam_frame_idx[img.first]);
    }
    if (marge_old)
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        ROS_ASSERT(multicam_frame_idx[baseFrameIdx].size() > 2);
        mergeNewFrameIdx = *(multicam_frame_idx[baseFrameIdx].end()-2);//边缘化基础帧的次新帧,得到其全局id
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    for(auto &img : images)
        ROS_DEBUG("number of feature: %d", f_manager[img.first].getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(images, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    ROS_INFO("MultiCamIdx before merge, merge type: %d:", static_cast<int>(marginalization_flag));
    printMultiCamIdx();
    // if(ESTIMATE_EXTRINSIC == 2)
    // {//这里默认第一个是双目，第二个是单目
    //     ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    //     if (frame_count != 0)
    //     {
    //         for(auto &img : images)
    //         {
    //             int frame_count_l_local = fromGlobalIndex2Local(multicam_frame_idx[img.first], frame_count-1);
    //             int frame_count_r_local = fromGlobalIndex2Local(multicam_frame_idx[img.first], frame_count);
    //             vector<pair<Vector3d, Vector3d>> corres = f_manager[img.first].getCorresponding(frame_count_l_local, frame_count_r_local);
    //             Matrix3d calib_ric;
    //             if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
    //             {
    //                 ROS_WARN("initial extrinsic rotation calib success");
    //                 ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
    //                 ric[0] = calib_ric;
    //                 RIC[0] = calib_ric;
    //                 ESTIMATE_EXTRINSIC = 1;
    //             }
    //         }
    //     }
    // }
    vector<int> all_fronts;
    for(int i = 0; i < FRONTEND_NUM; i++)
        all_fronts.push_back(i);
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    solver_flag = NON_LINEAR;
                    optimization(all_fronts);
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {//默认frontId为0的是双目，初始化阶段只采用双目帧
            f_manager[baseFrameIdx].initFramePoseByPnP(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);
            f_manager[baseFrameIdx].triangulate(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization(vector<int>{0});//初始化阶段只优化双目
                slideWindow();
                ROS_INFO("Initialization finish!");
                solver_flag = NON_LINEAR;
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager[baseFrameIdx].initFramePoseByPnP(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);
            f_manager[baseFrameIdx].triangulate(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);
            optimization(all_fronts);

            if(frame_count == WINDOW_SIZE)
            {
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {//非线性优化
        TicToc t_solve;
        if(!USE_IMU)
            f_manager[baseFrameIdx].initFramePoseByPnP(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);
        for(auto &img : images)
            f_manager[img.first].triangulate(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[baseFrameIdx]);

        vector<int> goodFrontID = getGoodFrontId();//检测能用的前端
        if(goodFrontID.empty())
        {
            ROS_WARN("There's no front end can be used in optimazation!");
        }
        optimization(goodFrontID);
        //剔除外点
        set<int> removeIndex;
        for(auto front_id : goodFrontID)
        {
            outliersRejection(front_id, removeIndex);
            f_manager[front_id].removeOutlier(removeIndex);
            if(! MULTIPLE_THREAD)
            {
                featureTracker[front_id].removeOutliers(removeIndex);
                predictPtsInNextFrame(front_id);
            }
            int remove_counts = removeIndex.size();
            ROS_INFO("f_manager-%d remove %d features.", front_id, remove_counts);
            removeIndex.clear();
        }
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection(baseFrameIdx))
        {//错误检测，实际没用
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        //滑窗操作
        slideWindow();
        ROS_INFO("MultiCamIdx after merge, merge type: %d:", static_cast<int>(marginalization_flag));
        printMultiCamIdx();
        for(int front_id = 0; front_id < FRONTEND_NUM; front_id++)
            f_manager[front_id].removeFailures();
        ROS_INFO("removeFailures successfully!");
        // prepare output of VINS
        //key_poses.clear();
        // for (int i = 0; i <= WINDOW_SIZE; i++)
        //     key_poses[i] = Ps[i];

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates(baseFrameIdx);
        ROS_INFO("updateLatestStates successfully!");
    }  
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager[0].feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points[0].second)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager[0].clearDepth();
    f_manager[0].triangulate(frame_count, Ps, Rs, tic, ric, multicam_frame_idx[0]);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager[0].getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double(const vector<int> &frontId)
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    for(auto id : frontId)
    {
        VectorXd dep = f_manager[id].getDepthVector();
        for (int i = 0; i < f_manager[id].getFeatureCount(); i++)
            para_Feature[id][i][0] = dep(i);
        para_Td[id][0] = td[id];
    }
}

void Estimator::double2vector(const vector<int> &frontId)
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }
    for(auto id : frontId)
    {
        VectorXd dep = f_manager[id].getDepthVector();
        //ROS_INFO("f_manager's feature counts is %d", f_manager[id].getFeatureCount());
        for (int i = 0; i < f_manager[id].getFeatureCount(); i++)
            dep(i) = para_Feature[id][i][0];
        f_manager[id].setDepth(dep);

        if(USE_IMU)
            td[id] = para_Td[id][0];
    }

}

bool Estimator::failureDetection(int frontId)
{
    return false;
    if (f_manager[frontId].last_track_num < 2)
    {
        ROS_INFO("FrontEnd-%d has little feature %d", frontId, f_manager[frontId].last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}
//frontId:参与优化的前端ID
void Estimator::optimization(const vector<int> &frontId)
{
    TicToc t_whole, t_prepare;
    vector2double(frontId);

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    //位姿、速度、零偏
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);
    //外参
    for(auto id : frontId)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        if(id < STEREO_NUM)
        {//双目
            int left_idx = id*2;
            int right_idx= left_idx + 1;
            problem.AddParameterBlock(para_Ex_Pose[left_idx], SIZE_POSE, local_parameterization);
            problem.AddParameterBlock(para_Ex_Pose[right_idx], SIZE_POSE, local_parameterization);
            if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
            {
                //ROS_INFO("estimate extinsic param");
                openExEstimation = 1;
            }
            else
            {
                //ROS_INFO("fix extinsic param");
                problem.SetParameterBlockConstant(para_Ex_Pose[left_idx]);
                problem.SetParameterBlockConstant(para_Ex_Pose[right_idx]);
            }
        }
        else
        {//单目
            int mono_idx = STEREO_NUM + id;
            problem.AddParameterBlock(para_Ex_Pose[mono_idx], SIZE_POSE, local_parameterization);
            if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
            {
                //ROS_INFO("estimate extinsic param");
                openExEstimation = 1;
            }
            else
            {
                //ROS_INFO("fix extinsic param");
                problem.SetParameterBlockConstant(para_Ex_Pose[mono_idx]);
            }
        }
    }
    //时间差
    for(auto id : frontId)
    {
        problem.AddParameterBlock(para_Td[id], 1);
        if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
            problem.SetParameterBlockConstant(para_Td[id]);
    }

    // ------------------------在问题中添加约束,构造残差函数---------------------------------- 
    //添加上一次边缘化形成的先验信息约束
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    //添加IMU约束
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }
    //添加视觉重投影约束 
    int f_m_cnt = 0;
    for(auto id : frontId)
    {
        bool is_stereo = id < STEREO_NUM;
        int param_ex_id = is_stereo? 2*id : STEREO_NUM + id;//外参左(单)目param索引
        int feature_index = -1;
        for (auto &it_per_id : f_manager[id].feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
    
            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            int imu_i_local = it_per_id.start_frame_local, imu_j_local = imu_i_local - 1;
            
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j_local++;
                imu_j = multicam_frame_idx[id][imu_j_local];
                if (imu_i_local != imu_j_local)//左(单)目帧到左(单)目帧的重投影误差
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[param_ex_id], para_Feature[id][feature_index], para_Td[id]);
                }

                if(is_stereo && it_per_frame.is_stereo)
                {                
                    Vector3d pts_j_right = it_per_frame.pointRight;
                    if(imu_i_local != imu_j_local)
                    {//左目到右目的重投影误差
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[param_ex_id], para_Ex_Pose[param_ex_id+1], para_Feature[id][feature_index], para_Td[id]);
                    }
                    else
                    {//首帧左目到右目的重投影误差
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Ex_Pose[param_ex_id], para_Ex_Pose[param_ex_id+1], para_Feature[id][feature_index], para_Td[id]);
                    }
                
                }
                f_m_cnt++;
            }
        }
    }
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector(frontId);
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;

    // -----------------------------marginalization ------------------------------------
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {//边缘化最旧帧
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double(frontId);
        //marge信息：上一时刻的先验因子
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        //marge信息：imu预积分因子
        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        //marge信息：视觉重投影误差约束
        {
            auto merge_front = getLastFrontId();
            for(auto id : merge_front)//边缘化将遍历指定前端的特征点
            {
                bool is_stereo = id < STEREO_NUM;
                int param_ex_id = is_stereo? 2*id : STEREO_NUM + id;//外参左(单)目param索引
                int feature_index = -1;
                for (auto &it_per_id : f_manager[id].feature)
                {
                    it_per_id.used_num = it_per_id.feature_per_frame.size();
                    if (it_per_id.used_num < 4)
                        continue;

                    ++feature_index;

                    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                    int imu_i_local = it_per_id.start_frame_local;
                    if (imu_i != 0 || imu_i_local != 0)
                        continue;
                    int imu_j_local = imu_i_local - 1;
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    for (auto &it_per_frame : it_per_id.feature_per_frame)
                    {
                        imu_j_local++;
                        imu_j = multicam_frame_idx[id][imu_j_local];
                        if(imu_i_local != imu_j_local)
                        {
                            Vector3d pts_j = it_per_frame.point;
                            ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[param_ex_id], para_Feature[id][feature_index], para_Td[id]},
                                                                                            vector<int>{0, 3});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        if(is_stereo && it_per_frame.is_stereo)
                        {
                            Vector3d pts_j_right = it_per_frame.pointRight;
                            if(imu_i_local != imu_j_local)
                            {
                                ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[param_ex_id], para_Ex_Pose[param_ex_id+1], para_Feature[id][feature_index], para_Td[id]},
                                                                                            vector<int>{0, 4});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                            else
                            {
                                ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                            vector<double *>{para_Ex_Pose[param_ex_id], para_Ex_Pose[param_ex_id+1], para_Feature[id][feature_index], para_Td[id]},
                                                                                            vector<int>{2});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        for(int i = 0; i < FRONTEND_NUM; i++)
            addr_shift[reinterpret_cast<long>(para_Td[i])] = para_Td[i];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {//边缘化次新帧
        if (last_marginalization_info &&//具有上一时刻的先验因子并且涉及到的参数包含需要边缘化的次新帧
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[mergeNewFrameIdx]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double(frontId);
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[mergeNewFrameIdx]);//先验信息涉及的参数块仅包含最旧帧位姿和速度零偏、中间帧位姿、外参等，不包含中间帧速度零偏
                    if (last_marginalization_parameter_blocks[i] == para_Pose[mergeNewFrameIdx])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            //调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == mergeNewFrameIdx)
                    continue;
                else if (i > mergeNewFrameIdx)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            for(int i = 0; i < FRONTEND_NUM; i++)
                addr_shift[reinterpret_cast<long>(para_Td[i])] = para_Td[i];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {//滑窗最旧帧
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                if(it_0 != all_image_frame.end())
                {
                    delete it_0->second.pre_integration;
                    all_image_frame.erase(all_image_frame.begin(), it_0);
                }
            }
            vector<int> mergeFronteId_vec = multiCamIdxSlideOld();//更新索引矩阵
            set<int> mergeFronteId_set;
            for(auto id : mergeFronteId_vec)
                mergeFronteId_set.insert(id);
            slideWindowOld(mergeFronteId_set);//更新特征点
        }
    }
    else
    {//滑窗次新帧
        if (frame_count == WINDOW_SIZE)
        {
            for(int i = mergeNewFrameIdx; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);//交换
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    if( i == mergeNewFrameIdx)
                    {//保证预积分连贯性
                        for (unsigned int j = 0; j < dt_buf[mergeNewFrameIdx+1].size(); j++)
                        {
                            double tmp_dt = dt_buf[mergeNewFrameIdx+1][j];
                            Vector3d tmp_linear_acceleration = linear_acceleration_buf[mergeNewFrameIdx+1][j];
                            Vector3d tmp_angular_velocity = angular_velocity_buf[mergeNewFrameIdx+1][j];

                            pre_integrations[mergeNewFrameIdx]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                            dt_buf[mergeNewFrameIdx].push_back(tmp_dt);
                            linear_acceleration_buf[mergeNewFrameIdx].push_back(tmp_linear_acceleration);
                            angular_velocity_buf[mergeNewFrameIdx].push_back(tmp_angular_velocity);
                        }
                    }
                    else
                        std::swap(pre_integrations[i], pre_integrations[i + 1]);
                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }  
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            auto pair_res = multiCamIdxSlideNew(mergeNewFrameIdx);//更新索引矩阵
            set<int> mergeFronteId_set;
            for(auto id : pair_res.first)
                mergeFronteId_set.insert(id);
            slideWindowNew(mergeFronteId_set, pair_res.second);//更新特征点
        }
    }
}
//adjust the multicam
void Estimator::slideWindowNew(set<int> mergeFrontId, vector<int> local_idx)
{
    sum_of_front++;
    int local_counts = 0;
    for(int id = 0; id < FRONTEND_NUM; id++)
    {
        if(mergeFrontId.find(id) != mergeFrontId.end())
        {//被边缘化的前端
            f_manager[id].removeFront(mergeNewFrameIdx, local_idx[local_counts]);
            local_counts++;
        }
        else
            f_manager[id].removeFrontNoMerge(mergeNewFrameIdx);
    }
}
//adjust the multicam
void Estimator::slideWindowOld(set<int> mergeFrontId)
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    for(int id = 0; id < FRONTEND_NUM; id++)
    {
        if(mergeFrontId.find(id) != mergeFrontId.end())
        {//被边缘化的前端
            int ex_id = getExParamIdx(id);
            int new_last_id = multicam_frame_idx[id][0];
            if (shift_depth)
            {
                Matrix3d R0, R1;
                Vector3d P0, P1;
                R0 = back_R0 * ric[ex_id];
                R1 = Rs[new_last_id] * ric[ex_id];
                P0 = back_P0 + back_R0 * tic[ex_id];
                P1 = Ps[new_last_id] + Rs[new_last_id] * tic[ex_id];
                f_manager[id].removeBackShiftDepth(R0, P0, R1, P1, multicam_frame_idx[id]);
            }
            else
                f_manager[id].removeBack(multicam_frame_idx[id]);
        }
        else//未被边缘化的前端
            f_manager[id].removeBackNoMerge();
    }

}

//merge_frame_idx:被边缘化次新帧的全局idx,返回被边缘化的前端ID
//pair<vector<int>, vector<int>> first:被边缘化的前端ID，second：被边缘化帧的局部ID
pair<vector<int>, vector<int>> Estimator::multiCamIdxSlideNew(int merge_frame_idx)
{
    pair<vector<int>, vector<int>> res;
    for(int i = 0; i < multicam_frame_idx.size(); i++)
    {   int local_idx = 0;
        for(auto it = multicam_frame_idx[i].begin(); it != multicam_frame_idx[i].end();)
        {
            if(*it == merge_frame_idx)
            {
                res.first.push_back(i);
                res.second.push_back(local_idx);
                multicam_frame_idx[i].erase(it);//earse后it指向最后一个元素
            }
            else if(*it > merge_frame_idx)
            {
                (*it)--;
                it++;
            }
            else
            {
                it++;
            }
            local_idx++;
        }
    }
    return res;
}
//边缘化最旧帧更新索引映射矩阵,返回要边缘化的前端ID,考虑同步帧被边缘化，所以返回vector<int>类型
vector<int> Estimator::multiCamIdxSlideOld()
{
    vector<int> merge_id;
    for(int i = 0; i < multicam_frame_idx.size(); i++)
    {
        for(auto it = multicam_frame_idx[i].begin(); it != multicam_frame_idx[i].end();)
        {   
            if(*it == 0)
            {
                merge_id.push_back(i);
                multicam_frame_idx[i].erase(it);
            }
            else
            {
                (*it)--;
                it++;
            }
        }
    }
    return merge_id;
}
//返回具有第0帧的前端ID
vector<int> Estimator::getLastFrontId()
{
    vector<int> res;
    int id = 0;
    for(auto &vec : multicam_frame_idx)
    {
        if(!vec.empty() && vec.front() == 0)
            res.push_back(id);
        id++;
    }
    return res;
}
//add_id：同步帧前端id
void Estimator::multiCamIdxPushBack(vector<int> add_id, int add_frame_idx)
{
    for(int i = 0; i < add_id.size(); i++)
    {
        multicam_frame_idx[add_id[i]].push_back(add_frame_idx);
    }
}
//输入前端id，输出该前端的外参索引
int Estimator::getExParamIdx(int frontId) const
{
    if(frontId < STEREO_NUM)
        return 2*frontId;
    else
        return frontId + STEREO_NUM;
}
//返回指定前端的滑窗大小
int Estimator::getWindowSize(int frontId) const
{
    return multicam_frame_idx[frontId].size();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame(int frontId)
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager[frontId].feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker[frontId].setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}
//剔除重投影误差大的特征点
void Estimator::outliersRejection(int frontId, set<int> &removeIndex)
{
    //return;
    auto &f_manager_cur = f_manager[frontId];
    int exparam_idx = getExParamIdx(frontId);
    int feature_index = -1;
    for (auto &it_per_id : f_manager_cur.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        int imu_i_local = it_per_id.start_frame_local, imu_j_local = imu_i_local - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j_local++;
            imu_j = multicam_frame_idx[frontId][imu_j_local];
            if (imu_i_local != imu_j_local)
            {//帧间左（单）目重投影误差
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[exparam_idx], tic[exparam_idx], 
                                                    Rs[imu_j], Ps[imu_j], ric[exparam_idx], tic[exparam_idx],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {//双目增加帧间左目到右目的重投影误差和帧内的左目到右目重投影误差
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i_local != imu_j_local)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[exparam_idx], tic[exparam_idx], 
                                                        Rs[imu_j], Ps[imu_j], ric[exparam_idx+1], tic[exparam_idx+1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {//帧内左目到右目的重投影误差
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[exparam_idx], tic[exparam_idx], 
                                                        Rs[imu_j], Ps[imu_j], ric[exparam_idx+1], tic[exparam_idx+1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);//R  
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;//P
    latest_V = latest_V + dt * un_acc;//V
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates(int latest_frontId)
{
    latest_time = Headers[frame_count] + td[latest_frontId];//最新帧的时间戳
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mBuf.unlock();
}

void Estimator::printMultiCamIdx()
{
    ROS_INFO("--------MultiCamIdx--------");
    std::string ss;
    for(int i = 0; i < multicam_frame_idx.size(); i++)
    {
        std::string s;
        int last_idx = -1;
        for(auto cur_idx : multicam_frame_idx[i])
        {
            int space_num = cur_idx - last_idx - 1;
            while(space_num--)
            {
                s.append(std::string(" _"));
            }
            s.append(" "+std::to_string(cur_idx));
            last_idx = cur_idx;
        }
        s.append("\n");
        ss.append(s);
    }
    ROS_INFO("%s", ss.c_str());
}
