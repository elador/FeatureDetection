/*
 * GradientFilter.cpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

GradientFilter::GradientFilter(int kernelSize, int blurKernelSize) : kernelSize(kernelSize), blurKernelSize(blurKernelSize) {
	if (kernelSize != 1 && kernelSize != 3 && kernelSize != 5 && kernelSize != 7 && kernelSize != CV_SCHARR)
		throw invalid_argument("GradientFilter: the kernel size must be 1, 3, 5, 7 or CV_SCHARR");
}

double GradientFilter::getScale(int kernelSize) const {
	if (kernelSize == 1)
		return 1. / 2;
	if (kernelSize == CV_SCHARR)
		return 1. / 32;
	return 1. / (1 << (2 * kernelSize - 3));
}

double GradientFilter::getDelta(int imageType) const {
	if (imageType == CV_8U)
		return 127;
	if (imageType == CV_16U)
		return 65535;
	return 0;
}

Mat GradientFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		throw invalid_argument("GradientFilter: the image must have exactly one channel");

	Mat blurred;
	if (blurKernelSize > 0) {
		blurred.create(image.rows, image.cols, image.type());
		cv::blur(image, blurred, cv::Size(blurKernelSize, blurKernelSize));
	} else {
		blurred = image;
	}

	int imageType = image.type();
	double scale = getScale(kernelSize);
	double delta = getDelta(imageType);
	Mat gradientX, gradientY;
	cv::Sobel(blurred, gradientX, imageType, 1, 0, kernelSize, scale, delta);
	cv::Sobel(blurred, gradientY, imageType, 0, 1, kernelSize, scale, delta);

	cv::merge({ gradientX, gradientY }, filtered);
	return filtered;
}

void GradientFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
