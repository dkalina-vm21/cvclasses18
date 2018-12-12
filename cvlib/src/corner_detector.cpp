/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Darina Kalina
 */

#include "cvlib.hpp"
#include <ctime>

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
	cv::Ptr<corner_detector_fast> ptr = cv::makePtr<corner_detector_fast>();
	ptr->make_test_points();
	return ptr;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();

	if (image.empty())
		return;

	else if (image.channels() == 1)
		image.copyTo(image_);

	else if (image.channels() == 3)
		cv::cvtColor(image, image_, cv::COLOR_BGR2GRAY);

	for (auto y = 3; y < image_.rows - 3; ++y)
		for (auto x = 3; x < image_.cols - 3; ++x)
			if (is_keypoint(cv::Point(x, y), 4, 3) &&
				is_keypoint(cv::Point(x, y), 1, 12))
				keypoints.push_back(cv::KeyPoint(cv::Point(x, y), 1.f));
}

void corner_detector_fast::set_threshold(int thresh)
{
	threshold_ = thresh;
}

brightness_check_result corner_detector_fast::check_brightness(unsigned int circle_point_num, cv::Point center)
{
	uchar center_brightness = image_.at<uchar>(center);
	uchar circle_point_brightness = image_.at<uchar>(center + circle_template_[circle_point_num]);
	if (circle_point_brightness <= center_brightness - static_cast<uchar>(threshold_))
		return brightness_check_result::darker;

	if(center_brightness + static_cast<uchar>(threshold_) <= circle_point_brightness)
		return brightness_check_result::brighter;

	if ((circle_point_brightness > center_brightness - static_cast<uchar>(threshold_)) &&
		(circle_point_brightness < center_brightness + static_cast<uchar>(threshold_)))
		return brightness_check_result::similar;
}

bool corner_detector_fast::is_keypoint(cv::Point center, unsigned int step, unsigned int num)
{
	unsigned int brighter_count = 0;
	unsigned int darker_count = 0;

	for (int i = 0; i < 16; i += step)
	{
		if (check_brightness(i, center) == brightness_check_result::brighter)
			++brighter_count;
		else if (check_brightness(i, center) == brightness_check_result::darker)
			++darker_count;
	}
	return (brighter_count >= num) || (darker_count >= num);
}

void corner_detector_fast::make_test_points()
{
	cv::RNG rng;
	cv::Point point1, point2;

	for (int i = 0; i < descriptor_length_; ++i)
	{
		point1.x = abs(cvRound(rng.gaussian(patch_size_ / 5.0)));
		point1.y = abs(cvRound(rng.gaussian(patch_size_ / 5.0)));
		point2.x = abs(cvRound(rng.gaussian(patch_size_ / 5.0)));
		point2.y = abs(cvRound(rng.gaussian(patch_size_ / 5.0)));
		test_points_pairs_.push_back(std::make_pair(point1, point2));
	}
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
	if (image.empty())
		return;

	else if (image.channels() == 1)
		image.copyTo(image_);

	else if (image.channels() == 3)
		cv::cvtColor(image, image_, cv::COLOR_BGR2GRAY);

	cv::GaussianBlur(image_, blured_image_, cv::Size(9,9), 2.75, 2.75, cv::BORDER_REPLICATE);
	cv::Mat descr(keypoints.size(), descriptor_length_ / 8, CV_8U, cv::Scalar(0));
	cv::copyMakeBorder(blured_image_, blured_image_, patch_size_, patch_size_, patch_size_, patch_size_,
		cv::BORDER_REPLICATE);
	cv::Point2i point1, point2;
	bool test_res;

	for (int i = 0; i < keypoints.size(); ++i)
	{
		for (int byte_num = 0; byte_num < descriptor_length_ / 8; ++byte_num)
		{
			descr.row(i).at<uchar>(byte_num) = 0;

			for (int bit_num = 0; bit_num < 8; ++bit_num)
			{
				point1 = cv::Point2i(keypoints[i].pt) + test_points_pairs_[byte_num * 8 + bit_num].first;
				point2 = cv::Point2i(keypoints[i].pt) + test_points_pairs_[byte_num * 8 + bit_num].second;

				 test_res = blured_image_.at<uchar>(point1) < blured_image_.at<uchar>(point2);
				descr.row(i).at<uchar>(byte_num) |= (test_res << bit_num);
			}
		}
	}
	descriptors.assign(descr);
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
	detect(image, keypoints);
	compute(image, keypoints, descriptors);
}
} // namespace cvlib
