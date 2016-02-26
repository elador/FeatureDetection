/*
 * AggregationFilter.cpp
 *
 *  Created on: 14.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/AggregationFilter.hpp"
#include "imageprocessing/filtering/BoxConvolutionFilter.hpp"
#include "imageprocessing/filtering/TriangularConvolutionFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;
using std::unique_ptr;

namespace imageprocessing {
namespace filtering {

AggregationFilter::AggregationFilter(int cellSize, bool interpolate, bool normalize) {
	if (cellSize < 1)
		throw invalid_argument("AggregationFilter: cellSize must be bigger than zero, but was " + std::to_string(cellSize));
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
