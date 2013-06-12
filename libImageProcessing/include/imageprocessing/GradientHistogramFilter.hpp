/*
 * GradientHistogramFilter.hpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#ifndef GRADIENTHISTOGRAMFILTER_HPP_
#define GRADIENTHISTOGRAMFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"
#include <array>

using cv::Vec2b;
using std::array;

namespace imageprocessing {

/**
 * Image filter that expects an image of type CV_8UC2, where the first channel corresponds to the gradient of x and
 * the second channel corresponds to the gradient of y. The resulting image has the same type, where the first
 * channel corresponds to the histogram bin (according to the gradient direction) and the second channel corresponds
 * to the weight (according to the magnitude of the gradient).
 */
class GradientHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new gradient histogram filter.
	 *
	 * @param[in] bins The amount of bins to use.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 * @param[in] offset Lower boundary of the first bin.
	 */
	explicit GradientHistogramFilter(unsigned int bins, bool signedGradients = false, double offset = 0);

	~GradientHistogramFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

	unsigned int getBinCount() const;

private:

	unsigned int bins; ///< The amount of bins.
	double offset; ///< Lower boundary of the first bin.
	array<Vec2b, 256 * 256> binCodes; ///< The look-up tables of the bin codes, the gradient codes are used as the index.
};

} /* namespace imageprocessing */
#endif /* GRADIENTHISTOGRAMFILTER_HPP_ */
