/*
 * GradientFilter.cpp
 *
 *  Created on: 08.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/GradientFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;
using std::vector;

namespace imageprocessing {
namespace filtering {

GradientFilter::GradientFilter(int kernelSize) : kernelSize(kernelSize), kernelScale(1) {
	bool kernelSizeIsOdd = kernelSize % 2 == 1;
	bool kernelSizeIsPositive = kernelSize > 0;
	if (kernelSize != CV_SCHARR && !(kernelSizeIsOdd && kernelSizeIsPositive))
		throw invalid_argument(
				"GradientFilter: the kernel size must be positive and odd or CV_SCHARR, but was " + std::to_string(kernelSize));
	computeKernelScale();
}

void GradientFilter::computeKernelScale() {
	if (kernelSize == 1)
		kernelScale = 1. / 2;
	else if (kernelSize == CV_SCHARR)
		kernelScale = 1. / 32;
	else
		kernelScale = 1. / (1 << (2 * kernelSize - 3));
}

Mat GradientFilter::applyTo(const Mat& image, Mat& filtered) const {
	vector<Mat> gradients = computeGradients(image);
	mergeGradients(gradients, filtered);
	return filtered;
}

vector<Mat> GradientFilter::computeGradients(const Mat& image) const {
	double delta = getDelta(image.depth());
	Mat gradientX, gradientY;
	cv::Sobel(image, gradientX, -1, 1, 0, kernelSize, kernelScale, delta, cv::BORDER_REPLICATE);
	cv::Sobel(image, gradientY, -1, 0, 1, kernelSize, kernelScale, delta, cv::BORDER_REPLICATE);
	return { gradientX, gradientY };
}

double GradientFilter::getDelta(int imageDepth) const {
	if (imageDepth == CV_8U)
		return 127;
	else if (imageDepth == CV_16U)
		return 32767;
	return 0;
}

void GradientFilter::mergeGradients(const vector<Mat>& gradients, Mat& filtered) const {
	int sourceChannels = gradients[0].channels();
	int destinationChannels = 2 * sourceChannels;
	filtered.create(gradients[0].rows, gradients[0].cols, CV_MAKETYPE(gradients[0].depth(), destinationChannels));
	vector<int> from_to(2 * destinationChannels);
	for (int k = 0; k < sourceChannels; ++k) {
		from_to[4 * k]     = k; // dx from
		from_to[4 * k + 1] = 2 * k; // dx to
		from_to[4 * k + 2] = sourceChannels + k; // dy from
		from_to[4 * k + 3] = 2 * k + 1; // dy to
	}
	cv::mixChannels(gradients, { filtered }, from_to);
}

} /* namespace filtering */
} /* namespace imageprocessing */
