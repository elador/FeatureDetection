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

using cv::Vec4b;
using std::array;

namespace imageprocessing {

/**
 * Image filter that expects an image of type CV_8UC2, where the first channel corresponds to the gradient of x and
 * the second channel corresponds to the gradient of y. The resulting image is of type CV_8UC4, where channels one
 * and three correspond to histogram bins (according to the gradient direction) and channels two and four are the
 * corresponding weights (according to the magnitude of the gradient and the distance of the gradient to the center
 * of the bin).
 */
class GradientHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new gradient histogram filter.
	 *
	 * @param[in] bins The amount of bins to use.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 */
	explicit GradientHistogramFilter(unsigned int bins, bool signedGradients = false);

	~GradientHistogramFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

	unsigned int getBinCount() const;

private:

	unsigned int bins;                ///< The amount of bins.
	array<Vec4b, 256 * 256> binCodes; ///< The look-up tables of the bin codes, the gradient codes are used as the index.
};

} /* namespace imageprocessing */
#endif /* GRADIENTHISTOGRAMFILTER_HPP_ */
