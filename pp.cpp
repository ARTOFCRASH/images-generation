#define _USE_MATH_DEFINES

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/io/png_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <string>
#include <vector>
#include <thread>


#ifdef _WIN32
    #include <direct.h> // For _mkdir on Windows
#else
    #include <sys/stat.h> // For mkdir on Linux/macOS
    #include <sys/types.h>
#endif


typedef pcl::PointXYZRGB PointT;

using namespace std::chrono_literals;

PointT picked_point;

// distance:200   expand times:*10


//事件回调函数：屏幕选点
void pointPickingCallback(const pcl::visualization::PointPickingEvent&);

// 旋转矩阵函数：返回值为旋转矩阵
Eigen::Matrix4f rot_mat(const Eigen::Vector3f&, const Eigen::Vector3f&, const float);

// 返回距离相机最近的点
float findMinValue(std::vector<float>&);

// 返回距离相机最远的点
float findMaxValue(std::vector<float>& depthArray);


std::string getBaseNameNoExt(const std::string& path) {
    size_t slash_pos = path.find_last_of("/\\");
    std::string filename = (slash_pos == std::string::npos)
                         ? path
                         : path.substr(slash_pos + 1);

    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(0, dot_pos);
    }
}


void ensureOutputDir(const std::string& dirName) {
#ifdef _WIN32
    _mkdir(dirName.c_str()); // Windows
#else
    mkdir(dirName.c_str(), 0777);
#endif
}


int main(int argc, char** argv) {

    std::string cloud_path = argv[1];
    std::string base_name = getBaseNameNoExt(cloud_path);
    ensureOutputDir(base_name);

    pcl::PointCloud<PointT>::Ptr point_cloud_ptr (new pcl::PointCloud<PointT>);

    pcl::PointCloud<PointT>& point_cloud = *point_cloud_ptr;

    pcl::io::loadPLYFile (argv[1], point_cloud);
    
    
    
    pcl::PassThrough<PointT> pass;              // 直通滤波器
    pcl::NormalEstimation<PointT, pcl::Normal> ne;  // 法线估算对象
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;   // 分割器
    pcl::PCDWriter writer;                          // PCD文件写出对象
    pcl::ExtractIndices<PointT> extract;            // 点提取对象
    pcl::ExtractIndices<pcl::Normal> extract_normals;   // 法线提取对象
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);            
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
    
    
      // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(point_cloud_ptr);
    ne.setKSearch(500);
    ne.compute(*cloud_normals);
    
    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients(true);          
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(10);
    seg.setInputCloud(point_cloud_ptr);
    seg.setInputNormals(cloud_normals);
    
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
    
    // Extract the planar inliers from the input cloud
    extract.setInputCloud(point_cloud_ptr);
    extract.setIndices(inliers_plane);
    extract.setNegative(true);   //这里false代表过滤出persimmon
    pcl::PointCloud<PointT>::Ptr cloud_persimmon(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_persimmon);                                              // pointer: cloud_persimmon (the object)
    Eigen::Vector3f plane_normal(coefficients_plane->values[0], coefficients_plane->values[1], coefficients_plane->values[2]);        // normal vector of the plane
    std::cout<<"The normal vector of the plane is: "<<plane_normal<<std::endl;
    

    
    
      // 根据平面法向量定义柿子自身的坐标系统
    Eigen::Vector3f ux (0, -plane_normal[2], plane_normal[1]);   // x-axis
    ux = ux.normalized();
    Eigen::Vector3f uy = plane_normal.cross(ux).normalized();    // y-axis
    Eigen::Vector3f uz = plane_normal.normalized();           // z-axis
    // 在可视化窗口选择点
    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    viewer.addPointCloud<PointT>(cloud_persimmon, "cloud");
    
    viewer.initCameraParameters ();
    
    viewer.registerPointPickingCallback(pointPickingCallback);
    
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce (100);
    }
    
    std::cout<<"picked point: "<<picked_point.x<<"  "<<picked_point.y<<"  "<<picked_point.z<<std::endl;
    
    
    
    
    
      // camera_transform  相机的变换矩阵
    double distance = 100;
    Eigen::Affine3f camera_transform = Eigen::Affine3f::Identity();
    camera_transform.linear().col(0) = ux;
    camera_transform.linear().col(1) = uy;
    // camera_transform.linear().col(2) = -plane_normal;
    camera_transform.linear().col(2) = uz;
    Eigen::Vector3f mass_center(picked_point.x, picked_point.y, picked_point.z);
    Eigen::Vector3f camera_center = mass_center - distance * plane_normal;
    camera_transform.translation() = camera_center;
    
    
    
    
    
    
    
    
/* pcl::visualization::PCLVisualizer viewer2 ("3D Viewer");
    viewer2.addPointCloud<PointT>(cloud_persimmon, "cloud");
    
    viewer2.initCameraParameters ();
    
    Eigen::Vector3f pos_vector = camera_transform.translation();
    Eigen::Vector3f look_at_vector = camera_transform.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector = camera_transform.rotation() * Eigen::Vector3f(0, 1, 0);
    
    
    viewer2.setCameraPosition(pos_vector[0], pos_vector[1], pos_vector[2],
                              look_at_vector[0], look_at_vector[1], look_at_vector[2],
                              up_vector[0], up_vector[1], up_vector[2]);
    
    
        while (!viewer2.wasStopped ())
    {
        viewer2.spinOnce (100);
    }
    
    */
    
    
    
    
    
    
    
    
    
    
    
    
    /*
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cloud_persimmon, *transformed_cloud, camera_transform);
    
    
        pcl::visualization::PCLVisualizer viewer2 ("3D Viewer");
    viewer2.addPointCloud<PointT>(cloud_persimmon, "cloud1");
    viewer2.addPointCloud<PointT>(transformed_cloud, "cloud2");
    viewer2.initCameraParameters ();
    
    
    while (!viewer2.wasStopped ())
    {
        viewer2.spinOnce (100);
    }
    
    
    */
    
    
    // 相机内参 IntelRealSense D405
    
    float fx = 390.056, fy = 389.777;
    float cx = 320.50, cy = 240.072;
    int width = 640, height = 480;
    

    // 相机外参: camera_transform
    Eigen::Matrix4f cameraTransform = camera_transform.matrix(); 
      
//for (int z = 50; z <=50; z++){

    //float yaw_angle = z * M_PI / 2;
    //Eigen::Matrix4f yaw_matrix = rot_mat(mass_center,uz,yaw_angle);
    
//  pcl::PointCloud<PointT>::Ptr yawed_cloud (new pcl::PointCloud<PointT> ());
//   pcl::transformPointCloud (*cloud_persimmon, *yawed_cloud, yaw_matrix);
    
      for(int i = 10; i <= 10; i++){
      
         for (int j = -10; j <= 12; j++){
         
         
         
                 float roll_angle = i * M_PI / 180;
                 float pitch_angle = j * M_PI / 180;
                   
                   
                   Eigen::Matrix4f roll_matrix = rot_mat(mass_center,uy,roll_angle);
                   
                   // pitch matrix after rolled
                   Eigen::Vector3f center_point = roll_matrix.block<3,3>(0,0) * mass_center + roll_matrix.block<3,1>(0,3);
                   Eigen::Vector3f transformed_ux = roll_matrix.block<3,3>(0,0) * ux;
                   Eigen::Matrix4f pitch_matrix = rot_mat(center_point, transformed_ux, pitch_angle);
                   
                   
                   
                  pcl::PointCloud<PointT>::Ptr transformed_cloud (new pcl::PointCloud<PointT> ());
                  pcl::transformPointCloud (*cloud_persimmon, *transformed_cloud, pitch_matrix*roll_matrix);
                   
                   
                    std::stringstream color_filename;
                    color_filename <<base_name<<"/"<<base_name<<"_"<<i<<"_"<<j<<"_"<<"_color"<<".png";
                                        std::stringstream depth_filename;
                    depth_filename <<base_name<<"/"<<base_name<<"_"<<i<<"_"<<j<<"_"<<"_depth"<<".png";
                                                             
                                                             // 1. 创建深度
                   cv::Mat depthFloat(height, width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::infinity()));

                 // 创建和初始化彩色图像
                 cv::Mat colorImage(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
                 
           
           Eigen::Matrix4f camInv = cameraTransform.inverse();

                   for (const auto& point : *transformed_cloud) {

                   // 世界坐标 -> 相机坐标
                   Eigen::Vector4f Pw(point.x, point.y, point.z, 1.0f);
                   Eigen::Vector4f Pc = camInv * Pw;

                   // 投影到图像平面
                   float X = Pc[0];
                   float Y = Pc[1];
                   float Z = Pc[2];
                   
                   if (Z <= 0.0f) continue;
                   float uf = fx * (X / Z) + cx;
                       float vf = fy * (Y / Z) + cy;

                       int px = static_cast<int>(std::floor(uf));
                       int py = static_cast<int>(std::floor(vf));
if (px < 0 || px >= width || py < 0 || py >= height) continue;
float& stored_depth = depthFloat.at<float>(py, px);
    if (Z < stored_depth) {
        stored_depth = Z;
        cv::Vec3b& pixel = colorImage.at<cv::Vec3b>(py, px);
        pixel[0] = point.b;
        pixel[1] = point.g;
        pixel[2] = point.r;
  }
}

cv::Mat depthImage(height, width, CV_16UC1);
for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
        float z = depthFloat.at<float>(yy, xx);

        if (std::isinf(z)) {
            depthImage.at<uint16_t>(yy, xx) = 0;
        } else {
            uint16_t depth_mm = static_cast<uint16_t>(z);
            depthImage.at<uint16_t>(yy, xx) = depth_mm;
        }
    }
}

cv::imwrite(color_filename.str(), colorImage);
cv::imwrite(depth_filename.str(), depthImage);

         }
      }
// }

    
    
    return 0;
}


//事件回调函数：屏幕选点
void pointPickingCallback(const pcl::visualization::PointPickingEvent& event)
{
    if (event.getPointIndex() == -1)
        return;

    float x, y, z;
    event.getPoint(x, y, z);

std::cout<<"select: "<<"x:"<<x<<"  y:"<<y<<"   z:"<<z<<std::endl;


    picked_point.x = x;
    picked_point.y = y;
    picked_point.z = z;

}




// 旋转矩阵函数：返回值为旋转矩阵
//  point:旋转轴通过的点   
// vector:旋转轴的方向向量，模长为1  
//      t:旋转角度
Eigen::Matrix4f rot_mat(const Eigen::Vector3f& point, const Eigen::Vector3f& vector, const float t)
{
    float u = vector(0);
    float v = vector(1);
    float w = vector(2);
    float a = point(0);
    float b = point(1);
    float c = point(2);
 
    Eigen::Matrix4f matrix;
    matrix<<u*u + (v*v + w*w)*cos(t), u*v*(1 - cos(t)) - w*sin(t), u*w*(1 - cos(t)) + v*sin(t), (a*(v*v + w*w) - u*(b*v + c*w))*(1 - cos(t)) + (b*w - c*v)*sin(t),
        u*v*(1 - cos(t)) + w*sin(t), v*v + (u*u + w*w)*cos(t), v*w*(1 - cos(t)) - u*sin(t), (b*(u*u + w*w) - v*(a*u + c*w))*(1 - cos(t)) + (c*u - a*w)*sin(t),
        u*w*(1 - cos(t)) - v*sin(t), v*w*(1 - cos(t)) + u*sin(t), w*w + (u*u + v*v)*cos(t), (c*(u*u + v*v) - w*(a*u + b*v))*(1 - cos(t)) + (a*v - b*u)*sin(t),
        0, 0, 0, 1;
    return matrix;
}


// 返回距离相机最近的点
float findMinValue(std::vector<float>& depthArray){


 float min_value = std::numeric_limits<float>::infinity();

for (float depth : depthArray) {
         if (depth != FLT_MAX && depth < min_value) {
             min_value = depth;
         }
}
return min_value;
}

// 返回距离相机最远的点
float findMaxValue(std::vector<float>& depthArray){

float maxVal = -FLT_MAX; // 初始化为极小值
     for (float depth : depthArray) {
         if (depth != FLT_MAX) {
             maxVal = std::max(maxVal, depth);
         }
}

return maxVal;

}