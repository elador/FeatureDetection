/*
 * ConvolutionFilter.cpp
 *
 *  Created on: 13.01.2014
 *      Author: poschmann
 */

#include "imageprocessing/ConvolutionFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Point;
using std::vector;
using std::invalid_argument;

namespace imageprocessing {

ConvolutionFilter::ConvolutionFilter(const Mat& kernel, Point anchor, double delta, int depth) :
		anchor(anchor), delta(delta), depth(depth) {
	setKernel(kernel);
}

ConvolutionFilter::ConvolutionFilter(int depth) : kernels(), anchor(-1, -1), delta(0), depth(depth) {}

Mat ConvolutionFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.empty()) {
		filtered.create(0, 0, filtered.type());
		return filtered;
	}
	vector<Mat> channels;
	cv::split(image, channels);
	if (channels.size() != kernels.size())
		throw invalid_argument("ConvolutionFilter: the amount of channels of the kernel and the image have to be the same");
	filtered.create(image.rows, image.cols, depth);
	filtered = delta;
	Mat tmp;
	for (size_t i = 0; i < channels.size(); ++i) {
		cv::filter2D(channels[i], tmp, depth, kernels[i], anchor, 0, cv::BORDER_CONSTANT);
		filtered += tmp;
	}
	return filtered;
}

void ConvolutionFilter::setKernel(const Mat& kernel) {
	kernels.clear();
	cv::split(kernel, kernels);
}

void ConvolutionFilter::setAnchor(Point anchor) {
	this->anchor = anchor;
}

void ConvolutionFilter::setDelta(double delta) {
	this->delta = delta;
}

} /* namespace imageprocessing */
