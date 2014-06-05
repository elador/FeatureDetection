/*
 * GrayscaleFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/GrayscaleFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;
using cv::cvtColor;

namespace imageprocessing {

GrayscaleFilter::GrayscaleFilter() {}

Mat GrayscaleFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		cvtColor(image, filtered, CV_BGR2GRAY);
	else
		image.copyTo(filtered);
	return filtered;
}

void GrayscaleFilter::applyInPlace(Mat& image) const {
	if (image.channels() > 1) {
		Mat tmp;
		cvtColor(image, tmp, CV_BGR2GRAY);
		image = tmp;
	}
}

} /* namespace imageprocessing */
