/*
 * ConversionFilter.hpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#ifndef CONVERSIONFILTER_HPP_
#define CONVERSIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Filter that converts the image to another type, rescaling the values. The amount of channels will remain unchanged.
 */
class ConversionFilter : public ImageFilter {
public:

	/**
	 * Constructs a new conversion filter.
	 *
	 * @param[in] type The OpenCV type (depth) of the filtered images. If negative, the type remains unchanged.
	 * @param[in] alpha The optional scaling factor.
	 * @param[in] beta The optional delta that is added to the scaled values.
	 */
	explicit ConversionFilter(int type, double alpha = 1, double beta = 0);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;

private:

	int type;     ///< The type (depth) of the filtered images.
	double alpha; ///< The scaling factor.
	double beta;  ///< The delta that is added to the scaled values.
};

} /* namespace imageprocessing */
#endif /* CONVERSIONFILTER_HPP_ */
