/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Darina Kalina
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

namespace
{
	enum brightness_check_result
	{
		darker = -1,
	    similar,
		brighter
	};
}; //namespace

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \return segmented image
cv::Mat split_and_merge(const cv::Mat& image, double stddev);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    /// \brief ctor

	motion_segmentation():is_initialized_(false), current_frame_(0) {};

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = 0.05) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override
    {
        backgroundImage.assign(bg_model_);
    }

	/// \brief check for background initialization
	bool isinitialized() const
	{
		return is_initialized_;
	}

	/// \brief set variance threshold
	void setVarThreshold(double threshold);

    private:
    cv::Mat bg_model_;
	cv::Mat variance_;
	double var_threshold_;
	unsigned int current_frame_;
	bool is_initialized_;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;
	
	void set_threshold(int thresh);

	/// \see Feature2d::compute
	virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

	/// \see Feature2d::getDefaultName
	virtual cv::String getDefaultName() const override
	{
		return "FAST_Binary";
	}

	private:
	brightness_check_result check_brightness(unsigned int circle_point_num, cv::Point center);
	bool is_keypoint(cv::Point center, unsigned int step, unsigned int num);
	void make_test_points();

	const cv::Point circle_template_[16] = { cv::Point(0, -3), cv::Point(1, -3),  cv::Point(2, -2),  cv::Point(3, -1),
											 cv::Point(3, 0),  cv::Point(3, 1),   cv::Point(2, 2),   cv::Point(1, 3),
			                                 cv::Point(0, 3),  cv::Point(-1, 3),  cv::Point(-2, 2),  cv::Point(-3, 1),
			                                 cv::Point(-3, 0), cv::Point(-3, -1), cv::Point(-2, -2), cv::Point(-1, -3) };

	const int patch_size_ = 48;
	const int descriptor_length_ = 256;

	int threshold_;
	cv::Mat image_;
	cv::Mat blured_image_;

	typedef std::vector<std::pair<cv::Point2i, cv::Point2i>> PointsPairs;
	PointsPairs test_points_pairs_;
};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    /// \brief ctor
    descriptor_matcher(float ratio = 1.5) : ratio_(ratio)
    {
    }

    /// \brief setup ratio threshold for SSD filtering
    void set_ratio(float r)
    {
        ratio_ = r;
    }

    protected:
    /// \see cv::DescriptorMatcher::knnMatchImpl
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::radiusMatchImpl
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual bool isMaskSupported() const override
    {
        return false;
    }

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override
    {
        cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
        if (emptyTrainData)
        {
            copy->clear();
        }
        return copy;
    }

    private:
    float ratio_;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
	/// \todo design and implement
public:
	Stitcher() : count_(0)
	{
		detector_ = cv::ORB::create();
		matcher_ = cv::BFMatcher::create();
		set_params(16, 26);
	}

	bool make_stiched_image(cv::Mat new_image);
	cv::Mat get_stiched_image();
	void cancel_last();
	void set_params(int detector_threshold, int max_match_distance);
	cv::Mat get_debug_info(cv::Mat new_image);

private:
	int count_;

	cv::Mat stitched_image_;
	std::vector<cv::KeyPoint> keypoints_;
	cv::Mat descriptors_;

	cv::Mat prev_stitched_image_;
	std::vector<cv::KeyPoint> prev_keypoints_;
	cv::Mat prev_descriptors_;

	cv::Ptr<cv::ORB> detector_;
	cv::Ptr<cv::BFMatcher> matcher_;

	int max_match_distance_ = 26;

	bool make_new_stiched_image(cv::Mat new_image);
	void update_keypoints(cv::Point2f offset, cv::Matx33f transform_mat, 
		const std::vector<cv::KeyPoint>& new_keypoints, cv::Mat new_descriptors,
		const std::vector<std::vector<cv::DMatch>>& pairs);
	
	// \brief helping functions
	void merge_images(cv::Mat src, cv::Mat dst);
	cv::Point2f transform_point(cv::Matx33f mat, cv::Point2f point);
	void get_expanded_size_and_offset(cv::Mat new_image, cv::Matx33f homo_mat, 
		cv::Mat prev_image, cv::Size& result_size, cv::Point2f& result_offset);
	cv::Matx33f translate_matrix(cv::Point2f vector);
};
} // namespace cvlib

#endif // __CVLIB_HPP__
