// ComputeDisparityMIT.cpp : 定义控制台应用程序的入口点。
//


#include "opencv2/opencv.hpp"
using namespace cv;

#include "pushbroom-stereo.hpp"
#include "opencv-stereo-util.hpp"

#include "helper_timer.h"

int main( )
{
	Mat matL = imread("Data/11_L.jpg", CV_8UC1);
	Mat matR = imread("Data/11_R.jpg", CV_8UC1);

	cv::vector<Point3f> pointVector3d;
	cv::vector<uchar> pointColors;
	cv::vector<Point3i> pointVector2d; // for display
	cv::vector<Point3i> pointVector2d_inf; // for display


	OpenCvStereoCalibration stereoCalibration;
	PushbroomStereoState state; // HACK
	configCD( stereoCalibration, state);

	PushbroomStereo pushbroom_stereo;


	StopWatchInterface	*timer;
	sdkCreateTimer( &timer );

	sdkResetTimer( &timer );
	sdkStartTimer( &timer );

	pushbroom_stereo.ProcessImages(matL, matR, &pointVector3d, &pointColors, &pointVector2d, state);

	sdkStopTimer( &timer );
	printf("total timer: %.2f ms \n", sdkGetTimerValue( &timer) );

	cout << pointVector2d.size() << "points " <<  endl;

	// output
	Mat matDisp, remapL, remapR;
#if 1
	remapL = matL;
	remapR = matR;
	remapL.copyTo(matDisp);
#else
	if (state.show_display) {
		// we remap again here because we're just in display
		Mat remapLtemp(matL.rows, matL.cols, matL.depth());
		Mat remapRtemp(matR.rows, matR.cols, matR.depth());

		remapL = remapLtemp;
		remapR = remapRtemp;

		remap(matL, remapL, stereoCalibration.mx1fp, Mat(), INTER_NEAREST);
		remap(matR, remapR, stereoCalibration.mx2fp, Mat(), INTER_NEAREST);

		remapL.copyTo(matDisp);

		//process LCM until there are no more messages
		// this allows us to drop frames if we are behind
	} // end show_display
#endif

	// global for where we are drawing a line on the image
	bool visualize_stereo_hits = true;
	bool show_unrectified = false;

	if (state.show_display) {

		for (unsigned int i=0;i<pointVector2d.size();i++) {
			int x2 = pointVector2d[i].x;
			int y2 = pointVector2d[i].y;
			//int sad = pointVector2d[i].z;
			//rectangle(matDisp, Point(x2,y2), Point(x2+state.blockSize, y2+state.blockSize), 0,  CV_FILLED);
				rectangle(matL, Point(x2,y2), Point(x2+state.blockSize, y2+state.blockSize), 0,  CV_FILLED);
			//rectangle(matDisp, Point(x2+1,y2+1), Point(x2+state.blockSize-1, y2-1+state.blockSize), 255);
				rectangle(matL, Point(x2+1,y2+1), Point(x2+state.blockSize-1, y2-1+state.blockSize), 255);
		}

		if (visualize_stereo_hits == true) {

			// draw the points on the unrectified image (to see these
			// you must pass the -u flag)
			Draw3DPointsOnImage(matL, &pointVector3d, stereoCalibration.M1, stereoCalibration.D1, stereoCalibration.R1, 128);

		}

	}

	if (show_unrectified == false) {

		//imshow("matDisp", matDisp);
			imshow("matL", matL);
	} else {
		imshow("matL", matL);
	}

	waitKey();

	return 0;
}

