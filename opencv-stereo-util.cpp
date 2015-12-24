/*
 * Utility functions for opencv-stereo
 *
 * Copyright 2013, Andrew Barry <abarry@csail.mit.edu>
 *
 */

#include "opencv-stereo-util.hpp"

#define ConfigFile	"Data/aaazzz.conf"
#define CalibrationDir	"..\\..\\calib-02-20-2014"

/**
 * Loads XML stereo calibration files and reamps them.
 *
 * @param calibrationDir directory the calibration files are in
 * @param stereoCalibration calibration structure to fill in
 *
 * @retval true on success, false on falure.
 *
 */
#ifdef CALIBEXE
bool readCalibrationFiles(CalibParams &params)
{
	Mat &mx1 = params.mx1;
	Mat &my1 = params.my1;
	Mat &mx2 = params.mx2;
	Mat &my2 = params.my2;

	// Load calibration file 
	FileStorage fs;
	fs.open("Data/calib_para.yml",FileStorage::READ);
	if (fs.isOpened())
	{
		Mat validroi1,validroi2;
		validroi1.create(1,4,CV_32S);
		validroi2.create(1,4,CV_32S);
		fs["MX1"] >> mx1;
		fs["MX2"] >> mx2;
		fs["MY1"] >> my1;
		fs["MY2"] >> my2;

		fs.release();

		fs.open("Data/validRoi.yml",FileStorage::READ);
		if (fs.isOpened())
		{
			fs["validRoi1"] >> validroi1;
			fs["validRoi2"] >> validroi2;
		}
		params.image_chop_up = MAX(validroi1.at<int>(0,1),validroi2.at<int>(0,1));
		params.image_chop_down = MAX(validroi1.at<int>(0,3),validroi2.at<int>(0,3));
		params.image_chop_left = MAX(validroi1.at<int>(0,0),validroi2.at<int>(0,0));

		//params.image_chop_left = MAX(params.image_chop_left, MAX_DISP);

		params.image_chop_right = MAX(validroi1.at<int>(0,2),validroi2.at<int>(0,2));

		//Mat P1, P2, Q, R_left;
		fs.open("Data/extrinsics.yml",FileStorage::READ);
		fs["P1"] >> params.P1;
		fs["P2"] >> params.P2;
		fs["Q"] >> params.Q;
		//fs["R1"] >> R_left;
		fs.release();
	//	Mat R1, T1, R2, T2;
		fs.open("Data/rt_vectors.yml", FileStorage::READ);
		if (fs.isOpened())
		{
			fs["R1"] >> params.R1;
			fs["T1"] >> params.T1;
			fs["R2"] >> params.R2;
			fs["T2"] >> params.T2;
		}
		fs.release();

		fs.open("Data/intrinsics.yml", FileStorage::READ);
		if (fs.isOpened())
		{
			fs["M1"] >> params.M1;
			fs["D1"] >> params.D1;
			fs["M2"] >> params.M2;
			fs["D2"] >> params.D2;
		}
		fs.release();


		return true;
	}
	else
	{
		//cout<<"NO Calibration File Found! "<<endl;
		fs.release();
		return false;
	}
}
#endif
bool TransCalibration(CalibParams &params, OpenCvStereoCalibration *stereoCalibration)
{
    Mat qMat, mx1Mat, my1Mat, mx2Mat, my2Mat, m1Mat, d1Mat, r1Mat, p1Mat, r2Mat, p2Mat, m2Mat, d2Mat;


    qMat = Mat(params.Q);
	mx1Mat = Mat(params.mx1);
	my1Mat = Mat(params.my1);
	mx2Mat = Mat(params.mx2);
	my2Mat = Mat(params.my2);

	m1Mat = Mat(params.M1);
	d1Mat = Mat(params.D1);
	r1Mat = Mat(params.R1);
	p1Mat = Mat(params.P1);

	m2Mat = Mat(params.M2);
	d2Mat = Mat(params.D2);
	r2Mat = Mat(params.R2);
	p2Mat = Mat(params.P2);

    Mat mx1fp, empty1, mx2fp, empty2;

    // this will convert to a fixed-point notation
    convertMaps(mx1Mat, my1Mat, mx1fp, empty1, CV_16SC2, true);
    convertMaps(mx2Mat, my2Mat, mx2fp, empty2, CV_16SC2, true);

    stereoCalibration->qMat = qMat;
    stereoCalibration->mx1fp = mx1fp;
    stereoCalibration->mx2fp = mx2fp;

    stereoCalibration->M1 = m1Mat;
    stereoCalibration->D1 = d1Mat;
    stereoCalibration->R1 = r1Mat;
    stereoCalibration->P1 = p1Mat;

    stereoCalibration->M2 = m2Mat;
    stereoCalibration->D2 = d2Mat;
    stereoCalibration->R2 = r2Mat;
    stereoCalibration->P2 = p2Mat;


    return true;
}


bool LoadCalibration(string calibrationDir, OpenCvStereoCalibration *stereoCalibration)
{
    Mat qMat, mx1Mat, my1Mat, mx2Mat, my2Mat, m1Mat, d1Mat, r1Mat, p1Mat, r2Mat, p2Mat, m2Mat, d2Mat;

    CvMat *Q = (CvMat *)cvLoad((calibrationDir + "/Q.xml").c_str(),NULL,NULL,NULL);

    if (Q == NULL)
    {
        std::cerr << "Error: failed to read " + calibrationDir + "/Q.xml." << std::endl;
        return false;
    }

    CvMat *mx1 = (CvMat *)cvLoad((calibrationDir + "/mx1.xml").c_str(),NULL,NULL,NULL);

    if (mx1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/mx1.xml." << std::endl;
        return false;
    }

    CvMat *my1 = (CvMat *)cvLoad((calibrationDir + "/my1.xml").c_str(),NULL,NULL,NULL);

    if (my1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/my1.xml." << std::endl;
        return false;
    }

    CvMat *mx2 = (CvMat *)cvLoad((calibrationDir + "/mx2.xml").c_str(),NULL,NULL,NULL);

    if (mx2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/mx2.xml." << std::endl;
        return false;
    }

    CvMat *my2 = (CvMat *)cvLoad((calibrationDir + "/my2.xml").c_str(),NULL,NULL,NULL);

    if (my2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/my2.xml." << std::endl;
        return false;
    }

    CvMat *m1 = (CvMat *)cvLoad((calibrationDir + "/M1.xml").c_str(),NULL,NULL,NULL);

    if (m1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/M1.xml." << std::endl;
        return false;
    }

    CvMat *d1 = (CvMat *)cvLoad((calibrationDir + "/D1.xml").c_str(),NULL,NULL,NULL);

    if (d1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/D1.xml." << std::endl;
        return false;
    }

    CvMat *r1 = (CvMat *)cvLoad((calibrationDir + "/R1.xml").c_str(),NULL,NULL,NULL);

    if (r1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/R1.xml." << std::endl;
        return false;
    }

    CvMat *p1 = (CvMat *)cvLoad((calibrationDir + "/P1.xml").c_str(),NULL,NULL,NULL);

    if (p1 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/P1.xml." << std::endl;
        return false;
    }

    CvMat *m2 = (CvMat *)cvLoad((calibrationDir + "/M2.xml").c_str(),NULL,NULL,NULL);

    if (m2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/M2.xml." << std::endl;
        return false;
    }

    CvMat *d2 = (CvMat *)cvLoad((calibrationDir + "/D2.xml").c_str(),NULL,NULL,NULL);

    if (d2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/D2.xml." << std::endl;
        return false;
    }

    CvMat *r2 = (CvMat *)cvLoad((calibrationDir + "/R2.xml").c_str(),NULL,NULL,NULL);

    if (r2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/R2.xml." << std::endl;
        return false;
    }

    CvMat *p2 = (CvMat *)cvLoad((calibrationDir + "/P2.xml").c_str(),NULL,NULL,NULL);

    if (p2 == NULL)
    {
        std::cerr << "Error: failed to read " << calibrationDir << "/P2.xml." << std::endl;
        return false;
    }



    qMat = Mat(Q, true);
    mx1Mat = Mat(mx1,true);
    my1Mat = Mat(my1,true);
    mx2Mat = Mat(mx2,true);
    my2Mat = Mat(my2,true);

    m1Mat = Mat(m1,true);
    d1Mat = Mat(d1,true);
    r1Mat = Mat(r1,true);
    p1Mat = Mat(p1,true);

    m2Mat = Mat(m2,true);
    d2Mat = Mat(d2,true);
    r2Mat = Mat(r2,true);
    p2Mat = Mat(p2,true);

    Mat mx1fp, empty1, mx2fp, empty2;

    // this will convert to a fixed-point notation
    convertMaps(mx1Mat, my1Mat, mx1fp, empty1, CV_16SC2, true);
    convertMaps(mx2Mat, my2Mat, mx2fp, empty2, CV_16SC2, true);

    stereoCalibration->qMat = qMat;
    stereoCalibration->mx1fp = mx1fp;
    stereoCalibration->mx2fp = mx2fp;

    stereoCalibration->M1 = m1Mat;
    stereoCalibration->D1 = d1Mat;
    stereoCalibration->R1 = r1Mat;
    stereoCalibration->P1 = p1Mat;

    stereoCalibration->M2 = m2Mat;
    stereoCalibration->D2 = d2Mat;
    stereoCalibration->R2 = r2Mat;
    stereoCalibration->P2 = p2Mat;


    return true;
}




/**
 * Draws 3D points onto a 2D image when given the camera calibration.
 *
 * @param camera_image image to draw onto
 * @param points_list_in vector<Point3f> of 3D points to draw. Likely obtained from Get3DPointsFromStereoMsg
 * @param cam_mat_m camera calibration matrix (usually M1.xml)
 * @param cam_mat_d distortion calibration matrix (usually D1.xml)
 * @param cam_mat_r rotation calibration matrix (usually R1.xml)
 * @param outline_color color to draw the box outlines (default: 128)
 * @param inside_color color to draw the inside of the boxes (default: 255). Set to -1 for no fill.
 * @param box_top if you only want to draw points inside a box, this specifies one coordinate of the box
 * @param box_bottom the second coordinate of the box
 * @param points_in_box if you pass box_top and box_bottom, this will be filled with the indicies of
 *          the points inside the box.
 * @param min_z minimum z value allowable to draw the point
 * @param max_z maximum z value allowable to draw the point
 * @param box_size size of the box (default = 4)
 */
void Draw3DPointsOnImage(Mat camera_image, vector<Point3f> *points_list_in, Mat cam_mat_m, Mat cam_mat_d, Mat cam_mat_r, Scalar outline_color, Scalar inside_color, Point2d box_top, Point2d box_bottom, vector<int> *points_in_box,
float min_z, float max_z, int box_size) {
    vector<Point3f> &points_list = *points_list_in;

    if (points_list.size() <= 0)
    {
        //std::cout << "Draw3DPointsOnimage: zero sized points list" << std::endl;
        return;
    }


    vector<Point2f> img_points_list;

    projectPoints(points_list, cam_mat_r.inv(), Mat::zeros(3, 1, CV_32F), cam_mat_m, cam_mat_d, img_points_list);


    int min_x = min(box_top.x, box_bottom.x);
    int min_y = min(box_top.y, box_bottom.y);
    int max_x = max(box_top.x, box_bottom.x);
    int max_y = max(box_top.y, box_bottom.y);
    bool box_bounding = false;

    if (box_top.x != -1 || box_top.y != -1 || box_bottom.x != -1 || box_bottom.y != -1) {
        box_bounding = true;
    }

    int thickness = CV_FILLED;
    if (inside_color[0] == -1) {
        thickness = 1;
    }

    // now draw the points onto the image
    for (int i=0; i<int(img_points_list.size()); i++)
    {

        //line(camera_image, Point(img_points_list[i].x, 0), Point(img_points_list[i].x, camera_image.rows), color);
        //line(camera_image, Point(0, img_points_list[i].y), Point(camera_image.cols, img_points_list[i].y), color);

        bool flag = false;

        if (box_bounding) {

            if (img_points_list[i].x >= min_x && img_points_list[i].x <= max_x &&
                img_points_list[i].y >= min_y && img_points_list[i].y <= max_y) {

                if (points_in_box) {
                    points_in_box->push_back(i);
                }

                flag = true;
            }
        }


        if (box_bounding == false || flag == true) {

            if (min_z == 0 || points_list[i].z >= min_z) {

                if (max_z == 0 || points_list[i].z <= max_z) {

                    rectangle(camera_image, Point(img_points_list[i].x - box_size, img_points_list[i].y - box_size),
                        Point(img_points_list[i].x + box_size, img_points_list[i].y + box_size), outline_color, thickness);

                    if (inside_color[0] != -1) {
                        rectangle(camera_image, Point(img_points_list[i].x - 2, img_points_list[i].y - box_size/2),
                            Point(img_points_list[i].x + box_size/2, img_points_list[i].y + box_size/2), inside_color, thickness);
                    }

                }
            }
        }

    }
}

int GetDisparityForDistance(double distance, const OpenCvStereoCalibration &calibration, int *inf_disparity) {

    int min_search = -100;
    int max_search = 100;

    cv::vector<Point3f> disparity_candidates;
    cv::vector<int> disparity_values;
    for (int i = min_search; i <= max_search; i++) {
        disparity_candidates.push_back(Point3f(0, 0, -i)); // note the negative! it is correct!
        disparity_values.push_back(i);
    }
    cv::vector<Point3f> vector_3d_out;

    perspectiveTransform(disparity_candidates, vector_3d_out, calibration.qMat);

    int best_disparity = 0;
    double best_dist_abs = -1;
    double max_dist = -1000;
    int max_disparity = 0;

    for (int i = 0; i < (int)vector_3d_out.size(); i++) {
        double this_dist_abs = fabs(vector_3d_out.at(i).z - distance);
        //std::cout << "Disp: " << disparity_values.at(i) << " ----> " << vector_3d_out.at(i).z << " (dist = " << this_dist_abs << ")" << std::endl;
        if (best_dist_abs == -1 || this_dist_abs < best_dist_abs) {
            best_disparity = disparity_values.at(i);
            best_dist_abs = this_dist_abs;
        }

        if (vector_3d_out.at(i).z > max_dist) {
            max_dist = vector_3d_out.at(i).z;
            max_disparity = disparity_values.at(i);
        }
    }

    if (inf_disparity != NULL) {
        *inf_disparity = max_disparity;
    }

    return best_disparity;
}

/**
 * Draws lines on the images for stereo debugging.
 *
 * @param rightImg right image
 * @param stereoImg stereo image
 * @param lineX x position of the line to draw
 * @param lineY y position of the line to draw
 * @param disparity disparity move the line on the right image over by
 * @param inf_disparity disparity corresponding to "infinite distance"
 *  used to filter out false-positives.  Usually availible in
 *   state.zero_dist_disparity.
 */
void DrawLines(Mat leftImg, Mat rightImg, Mat stereoImg, int lineX, int lineY, int disparity, int inf_disparity) {
    int lineColor = 128;
    if (lineX >= 0)
    {
        // print out the values of the pixels where they clicked
        //cout << endl << endl << "Left px: " << (int)leftImg.at<uchar>(lineY, lineX)
        //    << "\tRight px: " << (int)rightImg.at<uchar>(lineY, lineX + disparity)
        //    << endl;

        line(leftImg, Point(lineX, 0), Point(lineX, leftImg.rows), lineColor);
        line(stereoImg, Point(lineX, 0), Point(lineX, leftImg.rows), lineColor);
        line(rightImg, Point(lineX + disparity, 0), Point(lineX + disparity, rightImg.rows), lineColor);

        line(rightImg, Point(lineX + inf_disparity, 0), Point(lineX + inf_disparity, rightImg.rows), lineColor);

        line(leftImg, Point(0, lineY), Point(leftImg.cols, lineY), lineColor);
        line(stereoImg, Point(0, lineY), Point(leftImg.cols, lineY), lineColor);
        line(rightImg, Point(0, lineY), Point(rightImg.cols, lineY), lineColor);
    }
}

int configCD(OpenCvStereoCalibration& stereoCalibration, PushbroomStereoState& state) 
{
	
	
    // load calibration from calibexe
	CalibParams calibparam;
	if( !readCalibrationFiles(calibparam) )
	{
		printf("Cannot find calibration file!\n");
		//EtronDI_CloseDevice(m_hEtronDI);
		return 0 ;
	}
	//OpenCvStereoCalibration stereoCalibration;
	
		if (TransCalibration(calibparam, &stereoCalibration) != true)
	{
		cerr << "Error: failed to read calibration files. Quitting." << endl;
		return -1;
	}
/*
	if (LoadCalibration(CalibrationDir, &stereoCalibration) != true)
	{
		cerr << "Error: failed to read calibration files. Quitting." << endl;
		return -1;
	}
	*/
	




	int inf_disparity_tester, disparity_tester;
	//para of Distance
	disparity_tester = GetDisparityForDistance(2000, stereoCalibration, &inf_disparity_tester);

	//std::cout << "computed disparity is = " << disparity_tester << ", inf disparity = " << inf_disparity_tester << std::endl;

	float random_results = -1.0;
	bool show_display = true;

	// sensors\stereo\aaazzz.conf 
	//state.disparity = -105;
	//state.zero_dist_disparity = -95;
	state.disparity = disparity_tester;		// -41
	state.zero_dist_disparity = inf_disparity_tester;			// -1
	state.sobelLimit = 860;
	state.horizontalInvarianceMultiplier = 0.5;
	state.blockSize = 5;
	state.random_results = random_results;
	state.check_horizontal_invariance = true;

	if (state.blockSize > 10 || state.blockSize < 1)
	{
		fprintf(stderr, "Warning: block size is very large "
			"or small (%d).  Expect trouble.\n", state.blockSize);
	}

	state.sadThreshold = 54;

	state.mapxL = stereoCalibration.mx1fp;
	state.mapxR = stereoCalibration.mx2fp;
	state.Q = stereoCalibration.qMat;
	state.show_display = show_display;

	state.lastValidPixelRow =  -1;

	return 0;
}
