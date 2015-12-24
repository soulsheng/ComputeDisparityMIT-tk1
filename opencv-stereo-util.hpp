/*
 * Utility functions for opencv-stereo
 *
 * Copyright 2013, Andrew Barry <abarry@csail.mit.edu>
 *
 */

#ifndef OPENCV_STEREO_UTIL
#define OPENCV_STEREO_UTIL

#include <cv.h>
#include <highgui.h>
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/opencv.hpp"
#include "pushbroom-stereo-def.hpp"


#include <string>

#define CALIBEXE

extern "C"
{
    #include <stdio.h>
    #include <stdint.h>
    #include <stdlib.h>
    //#include <inttypes.h>

}


using namespace cv;

struct OpenCvStereoConfig
{
    uint64 guidLeft;
    uint64 guidRight;

    string lcmUrl;
    string stereoControlChannel;
    string calibrationDir;
    float calibrationUnitConversion;
    int lastValidPixelRow;
    string videoSaveDir;
    string fourcc;

    bool usePGM;

    string stereo_replay_channel;
    string baro_airspeed_channel;
    string pose_channel;
    string gps_channel;
    string battery_status_channel;
    string servo_out_channel;
    string optotrak_channel;
    string cpu_info_channel1;
    string cpu_info_channel2;
    string cpu_info_channel3;

    string log_size_channel1;
    string log_size_channel2;
    string log_size_channel3;


    int disparity;
    int infiniteDisparity;
    int interestOperatorLimit;
    int blockSize;
    int sadThreshold;
    float horizontalInvarianceMultiplier;

    int displayOffsetX;
    int displayOffsetY;

};



#ifdef CALIBEXE
struct CalibParams
	{
		Mat mx1, my1, mx2, my2;
		Mat P1, P2, Q;
		Mat R1, T1, R2, T2;
		Mat M1, M2, D1, D2;
		int image_chop_up;
		int image_chop_down;
		int image_chop_left;
		int image_chop_right;
	};
bool readCalibrationFiles(CalibParams &params);
#endif



struct OpenCvStereoCalibration
{
    Mat mx1fp;
    Mat mx2fp;
    Mat qMat;

    Mat M1;
    Mat D1;
    Mat R1;
    Mat P1;

    Mat M2;
    Mat D2;
    Mat R2;
    Mat P2;
};

bool LoadCalibration(string calibrationDir, OpenCvStereoCalibration *stereoCalibration);
bool TransCalibration(CalibParams &params, OpenCvStereoCalibration *stereoCalibration);

void Draw3DPointsOnImage(Mat camera_image, vector<Point3f> *points_list_in, Mat cam_mat_m, Mat cam_mat_d, Mat cam_mat_r, Scalar outline_color = 128, Scalar inside_color = 255, Point2d box_top = Point2d(-1, -1), Point2d box_bottom = Point2d(-1, -1), vector<int> *points_in_box = NULL, float min_z = 0, float max_z = 0, int box_size = 4);

int GetDisparityForDistance(double distance, const OpenCvStereoCalibration &calibration, int *inf_disparity = NULL);


void DrawLines(Mat leftImg, Mat rightImg, Mat stereoImg, 
	int lineX, int lineY, int disparity, int inf_disparity);

int configCD(OpenCvStereoCalibration& stereoCalibration, PushbroomStereoState& state);

#endif
