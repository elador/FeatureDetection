/*
 * GammaCorrectionFilter.cpp
 *
 *  Created on: 30.07.2013
 *      Author: poschmann
 */

#include "imageprocessing/GammaCorrectionFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

GammaCorrectionFilter::GammaCorrectionFilter(double gamma) : gamma(gamma), lut(1, 256, CV_8U) {
	uchar* ptr = lut.ptr();
	for (int i = 0; i < 256; ++i)
		ptr[i] = cvRound(255 * pow(static_cast<double>(i) / 255.0, gamma));
}

Mat GammaCorrectionFilter::applyTo(const Mat& image, Mat& filtered) const {
	switch (image.type()) {
	case CV_8U: LUT(image, lut, filtered); return filtered;
	case CV_32F: return applyGammaCorrection<float>(image, filtered);
	case CV_64F: return applyGammaCorrection<double>(image, filtered);
	}
	throw invalid_argument("GammaCorrectionFilter: The image must be of type CV_8U, CV_32F or CV_64F");
}

void GammaCorrectionFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
