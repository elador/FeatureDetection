/*
 * GridSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "condensation/GridSampler.hpp"
#include "condensation/Sample.hpp"
#include <algorithm>
#include <stdexcept>

using cv::Mat;
using std::min;
using std::max;
using std::vector;
using std::shared_ptr;
using std::invalid_argument;
using std::make_shared;

namespace condensation {

GridSampler::GridSampler(int minSize, int maxSize, float sizeScale, float stepSize) :
		minSize(minSize), maxSize(maxSize), sizeScale(sizeScale), stepSize(stepSize) {
	if (minSize < 1)
		throw invalid_argument("GridSampler: the minimum size must be greater than zero");
	if (maxSize < minSize)
		throw invalid_argument("GridSampler: the maximum size must not be smaller than the minimum size");
	if (sizeScale <= 1)
		throw invalid_argument("GridSampler: The scale factor of the size must be greater than one");
	if (sizeScale <= 0)
		throw invalid_argument("GridSampler: The step size must be greater than zero");
}

void GridSampler::init(const Mat& image) {}

void GridSampler::sample(const vector<shared_ptr<Sample>>& samples, vector<shared_ptr<Sample>>& newSamples,
		const Mat& image, const shared_ptr<Sample> target) {
	newSamples.clear();
	for (int size = minSize; size <= maxSize; size *= sizeScale) {
		int halfSize = size / 2;
		int minX = halfSize;
		int minY = halfSize;
		int maxX = image.cols - size + halfSize;
		int maxY = image.rows - size + halfSize;
		int step = (int)(stepSize * size + 0.5f);
		for (int x = minX; x < maxX; x += step) {
			for (int y = minY; y < maxY; y += step) {
				newSamples.push_back(make_shared<Sample>(x, y, size));
			}
		}
	}
}

} /* namespace condensation */
