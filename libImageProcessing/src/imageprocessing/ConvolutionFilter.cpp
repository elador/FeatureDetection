/*
 * ConvolutionFilter.cpp
 *
 *  Created on: 13.01.2014
 *      Author: poschmann
 */

#include "imageprocessing/ConvolutionFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using std::vector;
using std::invalid_argument;

namespace imageprocessing {

ConvolutionFilter::ConvolutionFilter(const Mat& kernel, double delta, int depth) : delta(delta), depth(depth) {}

ConvolutionFilter::ConvolutionFilter(int depth) : kernels(), delta(0), depth(depth) {}

Mat ConvolutionFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::Point anchor = cv::Point(-1, -1);
	vector<Mat> channels;
	cv::split(image, channels);
	if (channels.size() != kernels.size())
		throw invalid_argument("the amount of channels of the kernel and the image have to be the same");
	filtered.create(image.rows, image.cols, depth);
	filtered = delta;
	Mat tmp;
	for (size_t i = 0; i < channels.size(); ++i) {
		cv::filter2D(channels[i], tmp, depth, kernels[i], anchor, 0, cv::BORDER_CONSTANT);
		filtered += tmp;
	}
	return filtered;
}

void ConvolutionFilter::setKernel(const cv::Mat& kernel) {
	anchor = cv::Point(kernel.cols / 2, kernel.rows / 2);
	kernels.clear();
	cv::split(kernel, kernels);
}

void ConvolutionFilter::setDelta(double delta) {
	this->delta = delta;
}

} /* namespace imageprocessing */
