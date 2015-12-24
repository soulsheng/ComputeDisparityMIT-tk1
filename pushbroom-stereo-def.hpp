/**
 * Implements pushbroom stereo, a fast, single-disparity stereo algorithm.
 *
 * Copyright 2013-2015, Andrew Barry <abarry@csail.mit.edu>
 *
 */

#ifndef PUSHBROOM_STEREO_DEF_HPP
#define PUSHBROOM_STEREO_DEF_HPP

#include "opencv2/opencv.hpp"
#include <cv.h>
#include <iostream>

#include <math.h>
#include <random> // for debug random generator


#define NUM_THREADS 1
//#define NUM_REMAP_THREADS 8
#define NUMERIC_CONST 333 // just a constant that we multiply the score by to make

using namespace cv;
using namespace std;

enum ThreadWorkType { REMAP, INTEREST_OP, STEREO };

struct PushbroomStereoState
{
    int disparity;
    int zero_dist_disparity;
    int sobelLimit;
    int blockSize;
    int sadThreshold;
    float horizontalInvarianceMultiplier;

    int lastValidPixelRow;

    Mat mapxL;

    Mat mapxR;

    Mat Q;

    bool show_display, check_horizontal_invariance;

    float random_results;

    float debugJ, debugI, debugDisparity;

    //cv::vector<Point3f> *localHitPoints; // this is an array of cv::vector<Point3f>'s
};

struct PushbroomStereoStateThreaded {
    PushbroomStereoState state;

    Mat remapped_left;
    Mat remapped_right;

    Mat laplacian_left;
    Mat laplacian_right;

    cv::vector<Point3f> *pointVector3d;
    cv::vector<Point3i> *pointVector2d;
    cv::vector<uchar> *pointColors;

    int row_start;
    int row_end;


};

struct RemapThreadState {
    Mat left_image;
    Mat right_image;

    Mat sub_remapped_left_image;
    Mat sub_remapped_right_image;

    Mat sub_laplacian_left;
    Mat sub_laplacian_right;

    Mat submapxL;
    Mat submapxR;
};

struct InterestOpState {
    Mat left_image;
    Mat right_image;

    Mat sub_laplacian_left;
    Mat sub_laplacian_right;

    int row_start;
    int row_end;
};



#endif
