/*
 * GradientOrientationFilter.hpp
 *
 *  Created on: 09.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_GRADIENTORIENTATIONFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_GRADIENTORIENTATIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/filtering/GradientMagnitudeFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <array>

namespace imageprocessing {
namespace filtering {

/**
 * Filter that computes gradient orientation and magnitude given an image containing gradients for x and y.
 *
 * The input image is expected to have an even number of channels, where the odd channels are gradients of x and
 * the even channels are gradients of y. If the input image has more than two channels, meaning that the gradients
 * were computed over several channels, then the strongest (biggest magnitude) of these gradients is used per
 * pixel. The depth of the input image can be CV_8U (with a value of 127 indicating no gradient at all) or CV_32F.
 *
 * The resulting image is of type CV_32FC2 with the first channel containing the orientation angle and the second
 * channel containing the magnitude. The orientation angle is in the range [0,2*pi) in case of full gradients and
 * [0,pi) in case of half gradients. The magnitude may be normalized. In that case, it is divided by the smoothed
 * magnitude plus a small constant (to avoid division by zero). The smoothed magnitude is obtained by convolving
 * with a triangular filter kernel.
 */
class GradientOrientationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient orientation filter.
	 *
	 * @param[in] full Flag that indicates whether the orientations should cover the full range of [0,2*pi) instead of only [0,pi).
	 * @param[in] normalizationRadius Radius of the magnitude's triangular smoothing filter (filter size is 2 * radius + 1).
	 * @param[in] normalizationConstant Small constant that is added to the smoothed magnitude to prevent division by zero.
	 */
	explicit GradientOrientationFilter(bool full = true, int normalizationRadius = 0, double normalizationConstant = 0.005);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * @return The upper bound (exclusive) of the gradient orientations, 2*pi for full gradients and pi for half gradients.
	 */
	float getUpperBound() const {
		return half ? PI : TWO_PI;
	}

	static constexpr float PI = static_cast<float>(M_PI); ///< Upper bound (exclusive) of half gradient orientations.
	static constexpr float TWO_PI = static_cast<float>(2 * M_PI); ///< Upper bound (exclusive) of full gradient orientations.

private:

	void createGradientLut();

	void computeOrientationAndMagnitude(const cv::Mat& image, cv::Mat& filtered) const;

	void computeOrientationAndMagnitudeForUchar(const cv::Mat& image, cv::Mat& filtered) const;

	void computeOrientationAndMagnitudeForUcharN(const cv::Mat& image, int originalChannels, cv::Mat& filtered) const;

	void computeOrientationAndMagnitudeForFloat(const cv::Mat& image, cv::Mat& filtered) const;

	void computeOrientationAndMagnitudeForFloatN(const cv::Mat& image, int originalChannels, cv::Mat& filtered) const;

public:

	float computeOrientation(float gradientX, float gradientY) const;

	float computeMagnitude(float gradientX, float gradientY) const;

	float computeSquaredMagnitude(float gradientX, float gradientY) const;

private:

	bool isNormalizing() const;

	void normalizeMagnitude(cv::Mat& orientationAndMagnitude) const;

	cv::Mat extractMagnitude(const cv::Mat& orientationAndMagnitude) const;

	void normalizeMagnitude(cv::Mat& orientationAndMagnitude, const cv::Mat& smoothedMagnitude) const;

	bool half; ///< Flag that indicates whether the orientations should only cover the range [0,pi) instead of [0,2*pi).
	std::array<cv::Vec2f, 256 * 256> gradientLut; ///< Look-up table for gradient orientation and magnitude of CV_8U gradient images.
	GradientMagnitudeFilter magnitudeFilter; ///< Gradient magnitude filter that is only used for computing the smoothed magnitudes.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_GRADIENTORIENTATIONFILTER_HPP_ */
