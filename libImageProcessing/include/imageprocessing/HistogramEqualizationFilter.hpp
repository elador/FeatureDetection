/*
 * HistogramEqualizationFilter.hpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#ifndef HISTOGRAMEQUALIZATIONFILTER_HPP_
#define HISTOGRAMEQUALIZATIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that equalizes the histogram using OpenCV's equalizeHist function. The images must
 * be of type 8-bit single channel (CV_8U / CV_8UC1). The output image will be CV_8U as well.
 */
class HistogramEqualizationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new histogram equalization filter.
	 */
	HistogramEqualizationFilter();

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;
};

} /* namespace imageprocessing */
#endif /* HISTOGRAMEQUALIZATIONFILTER_HPP_ */
