/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Darina Kalina
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"


int demo_image_stitching(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	const auto main_wnd = "orig";
	const auto demo_wnd = "demo";
	const auto dbg_wnd = "dbg";

	cv::namedWindow(main_wnd);
	cv::namedWindow(demo_wnd);
	cv::namedWindow(dbg_wnd);

	int corner_threshold = 16;
	int max_distance = 26;
	const int max_blur = 100;
	const double blur_coeff = 5.0;
	int blur_value = 20;

	cvlib::Stitcher stitcher;

	cv::createTrackbar("thresh", demo_wnd, &corner_threshold, 255);
	cv::createTrackbar("max dist", demo_wnd, &max_distance, 150);
	cv::createTrackbar("blur %", demo_wnd, &blur_value, 100);

	cv::Mat orig_frame;
	cv::Mat demo_frame;
	cv::Mat dbg_frame;
	utils::fps_counter fps;

	int pressed_key = 0;
	while (pressed_key != 27) // ESC
	{
		cap >> orig_frame;
		double blur_sigma = static_cast<double>(blur_value) / max_blur * blur_coeff;
		cv::imshow(main_wnd, orig_frame);
		cv::GaussianBlur(orig_frame, orig_frame, cv::Size(5, 5), blur_sigma);
		stitcher.set_params(corner_threshold, max_distance);

		dbg_frame = stitcher.get_debug_info(orig_frame);
		cv::imshow(dbg_wnd, dbg_frame);

		demo_frame = stitcher.get_stiched_image();
		if (!demo_frame.empty())
			cv::imshow(demo_wnd, demo_frame);

		pressed_key = cv::waitKey(30);
		if (pressed_key == ' ')
			stitcher.make_stiched_image(orig_frame);
		else if (pressed_key == 'z')
			stitcher.cancel_last();
	}

	cv::destroyWindow(main_wnd);
	cv::destroyWindow(demo_wnd);
	cv::destroyWindow(dbg_wnd);

	return 0;
}



