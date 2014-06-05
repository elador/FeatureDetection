/*
 * HistogramEqualizationFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;

namespace imageprocessing {

HistogramEqualizationFilter::HistogramEqualizationFilter() {}

Mat HistogramEqualizationFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::equalizeHist(image, filtered);
	return filtered;
}

void HistogramEqualizationFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
