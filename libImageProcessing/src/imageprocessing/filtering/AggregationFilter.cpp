/*
 * AggregationFilter.cpp
 *
 *  Created on: 14.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/AggregationFilter.hpp"
#include "imageprocessing/filtering/BoxConvolutionFilter.hpp"
#include "imageprocessing/filtering/TriangularConvolutionFilter.hpp"

using cv::FilterEngine;
using cv::Mat;
using cv::Point;
using cv::Ptr;
using cv::Rect;
using cv::Size;
using std::unique_ptr;

namespace imageprocessing {
namespace filtering {

AggregationFilter::AggregationFilter(int cellSize, bool interpolate, bool normalize) {
	int downScaling = cellSize;
	float alpha = normalize ? 1 : (cellSize * cellSize);
	if (interpolate) {
		bool isCellSizeEven = cellSize % 2 == 0;
		int filterSize = isCellSizeEven ? (2 * cellSize) : (2 * cellSize - 1);
		downsamplingConvolutionFilter = unique_ptr<TriangularConvolutionFilter>(new TriangularConvolutionFilter(filterSize, downScaling, alpha));
	} else {
		downsamplingConvolutionFilter = unique_ptr<BoxConvolutionFilter>(new BoxConvolutionFilter(cellSize, downScaling, alpha));
	}
}

Mat AggregationFilter::applyTo(const Mat& image, Mat& filtered) const {
	return downsamplingConvolutionFilter->applyTo(image, filtered);
}

} /* namespace filtering */
} /* namespace imageprocessing */
