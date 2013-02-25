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
 * be of type 8-bit single channel (CV_8UC1).
 */
class HistogramEqualizationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new histogram equalization filter.
	 */
	HistogramEqualizationFilter();

	~HistogramEqualizationFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);
};

} /* namespace imageprocessing */
#endif /* HISTOGRAMEQUALIZATIONFILTER_HPP_ */
