/*
 * GradientBinningFilter.hpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#ifndef GRADIENTBINNINGFILTER_HPP_
#define GRADIENTBINNINGFILTER_HPP_

#include "imageprocessing/BinningFilter.hpp"
#include <array>

namespace imageprocessing {

/**
 * Image filter that expects an image of type CV_8UC2, where the first channel corresponds to the gradient of x and
 * the second channel corresponds to the gradient of y.
 *
 * Without interpolation, the resulting image is of type CV_8UC2, where the first channel corresponds to the histogram
 * bins (according to the gradient direction) and the second channel is the corresponding weight (according to the
 * magnitude of the gradient).
 *
 * With interpolation, the resulting image is of type CV_8UC4, where channels one and three correspond to histogram
 * bins (according to the gradient direction) and channels two and four are the corresponding weights (according to
 * the magnitude of the gradient and the distance of the gradient to the center of the bin).
 */
class GradientBinningFilter : public BinningFilter {
public:

	/**
	 * Constructs a new gradient histogram filter.
	 *
	 * @param[in] bins The amount of bins.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 * @param[in] interpolate Flag that indicates whether the bin weight should be divided between the closest bins.
	 */
	explicit GradientBinningFilter(unsigned int bins, bool signedGradients = false, bool interpolate = false);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	unsigned int getBinCount() const;

private:

	unsigned int bins;                   ///< The amount of bins.
	bool interpolate;                    ///< Flag that indicates whether the bin weight should be divided between the closest bins.
	std::array<cv::Vec2b, 256 * 256> oneBinCodes; ///< The look-up table of bin codes (no interpolation), the gradient codes are used as the index.
	std::array<cv::Vec4b, 256 * 256> twoBinCodes; ///< The look-up table of bin codes (with interpolation), the gradient codes are used as the index.
	int resultType;                      ///< The type of the filtered images.
};

} /* namespace imageprocessing */
#endif /* GRADIENTBINNINGFILTER_HPP_ */
