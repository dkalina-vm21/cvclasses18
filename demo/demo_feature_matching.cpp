/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

namespace
{
	void detector_trackbar_callback_func(int threshold, void* obj)
	{
		cvlib::corner_detector_fast* detector = (cvlib::corner_detector_fast*)obj;
		detector->set_threshold(threshold);
	}

	void ratio_trackbar_callback_func(int value, void* ptr)
	{
		((cvlib::descriptor_matcher*)(ptr))->set_ratio(float(value / 100));
	}

} //namespace

int demo_feature_matching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

	auto detector = cvlib::corner_detector_fast::create();
	auto threshold = 20;
	cv::createTrackbar("thresh", demo_wnd, &threshold, 50, detector_trackbar_callback_func, (void*)detector);
	detector->set_threshold(threshold);

	auto ratio = 50;
    auto matcher = cvlib::descriptor_matcher(ratio);
	cv::createTrackbar("ratio SSD", demo_wnd, &ratio, 100, ratio_trackbar_callback_func, (void*)&matcher);
	matcher.set_ratio(ratio/100);

	auto max_distance = 1000;
	cv::createTrackbar("max_dist", demo_wnd, &max_distance, 5000);

    /// \brief helper struct for tidy code
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

    img_features ref;
    img_features test;
    std::vector<std::vector<cv::DMatch>> pairs;

    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> test.img;
        detector->detect(test.img, test.corners);
        cv::drawKeypoints(test.img, test.corners, main_frame);
        cv::imshow(main_wnd, main_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            ref.img = test.img.clone();
            detector->detectAndCompute(ref.img, cv::Mat(), ref.corners, ref.descriptors);
        }

        if (ref.corners.empty())
        {
            continue;
        }

        detector->compute(test.img, test.corners, test.descriptors);
        matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, max_distance);
        cv::drawMatches(test.img, test.corners, ref.img, ref.corners, pairs, demo_frame);

        utils::put_fps_text(demo_frame, fps);
        cv::imshow(demo_wnd, demo_frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
