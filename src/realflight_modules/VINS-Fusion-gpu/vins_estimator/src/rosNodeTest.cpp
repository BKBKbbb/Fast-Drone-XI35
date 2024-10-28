/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <ros/master.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include "thread_pool/ThreadPool.h"

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
//multi
vector<queue<sensor_msgs::ImageConstPtr>> img_buf;
vector<std::mutex> m_buf;

class Image_Callback
{
public:
    Image_Callback(){}
    Image_Callback(int _frontend_id, int _cam_id): frontend_id(_frontend_id), cam_id(_cam_id){}

    void init(int _frontend_id, int _cam_id)
    {
        frontend_id = _frontend_id;
        cam_id = _cam_id;
    }
    void operator()(const sensor_msgs::ImageConstPtr &img_msg)
    {
        m_buf[frontend_id].lock();
        img_buf[cam_id].push(img_msg);
        m_buf[frontend_id].unlock();
    }
private:
    int frontend_id;
    int cam_id;
};

void img_callback(int frontend_id, int cam_id, const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf[frontend_id].lock();
    img_buf[cam_id].push(img_msg);
    m_buf[frontend_id].unlock();
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf[0].lock();
    img0_buf.push(img_msg);
    m_buf[0].unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf[0].lock();
    img1_buf.push(img_msg);
    m_buf[0].unlock();
}

//sensor_msgs::Image ==> cv::Mat
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
//adjust multi camera
void sync_process(bool is_stereo, int frontEndId)
{
    while(1)
    {
        if(is_stereo)
        {
            int left_idx = frontEndId * 2;
            int right_idx = left_idx + 1;
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf[frontEndId].lock();
            if (!img_buf[left_idx].empty() && !img_buf[right_idx].empty())
            {
                double time0 = img_buf[left_idx].front()->header.stamp.toSec();
                double time1 = img_buf[right_idx].front()->header.stamp.toSec();
                if(time0 < time1)
                {
                    img_buf[left_idx].pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1)
                {
                    img_buf[right_idx].pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img_buf[left_idx].front()->header.stamp.toSec();
                    header = img_buf[left_idx].front()->header;
                    image0 = getImageFromMsg(img_buf[left_idx].front());
                    img_buf[left_idx].pop();
                    image1 = getImageFromMsg(img_buf[right_idx].front());
                    img_buf[right_idx].pop();
                    //printf("find img0 and img1\n");
                }
            }
            m_buf[frontEndId].unlock();
            if(!image0.empty())
            {
                ROS_DEBUG("Input stereo images!");
                estimator.inputImage(frontEndId, time, image0, image1);
                
            }
        }
        else
        {
            int mono_idx = STEREO_NUM + frontEndId;
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf[frontEndId].lock();
            if(!img_buf[mono_idx].empty())
            {
                time = img_buf[mono_idx].front()->header.stamp.toSec();
                header = img_buf[mono_idx].front()->header;
                image = getImageFromMsg(img_buf[mono_idx].front());
                img_buf[mono_idx].pop();
            }
            m_buf[frontEndId].unlock();
            if(!image.empty())
            {
                ROS_DEBUG("Input mono images!");
                estimator.inputImage(frontEndId, time, image);
                
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(0, t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf[0].lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf[0].unlock();
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

    img_buf = vector<queue<sensor_msgs::ImageConstPtr>>(NUM_OF_CAM);
    m_buf = vector<std::mutex>(FRONTEND_NUM);

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 20000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    
    //initialize sub_img
    ros::Subscriber sub_img[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
    {
        int frontend_id;
        if(i < STEREO_NUM*2)
            frontend_id = i/2;
        else
            frontend_id = i - STEREO_NUM;
        sub_img[i] = n.subscribe<sensor_msgs::Image>(IMAGES_TOPIC[i], 2000, boost::bind(&img_callback, frontend_id, i, _1));
    }
    //init thread_pool
    ThreadPool pool(FRONTEND_NUM);
    pool.init();
    for(int i = 0; i < FRONTEND_NUM; i++)
    {
        if(i < STEREO_NUM)
            pool.submit(sync_process, true, i);
        else
            pool.submit(sync_process, false, i);
    }
    ros::spin();
    pool.shutdown();
    return 0;
}
