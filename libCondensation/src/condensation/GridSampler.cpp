/*
 * GridSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "tracking/GridSampler.h"
#include "tracking/Sample.h"
#include <algorithm>

using std::min;
using std::max;

namespace tracking {

GridSampler::GridSampler(float minSize, float maxSize, float sizeScale, float stepSize) :
		minSize(minSize), maxSize(maxSize), sizeScale(sizeScale), stepSize(stepSize) {
	minSize = min(1.0f, max(0.0f, minSize));
	maxSize = min(1.0f, max(0.0f, maxSize));
	sizeScale = max(1.05f, sizeScale);
	stepSize = max(0.0f, stepSize);
}

GridSampler::~GridSampler() {}

void GridSampler::sample(const vector<Sample>& samples, const vector<double>& offset,
			const Mat& image, vector<Sample>& newSamples) {
	newSamples.clear();
	int minSize = (int)(this->minSize * min(image.cols, image.rows));
	int maxSize = (int)(this->maxSize * min(image.cols, image.rows));
	Sample newSample;
	for (int size = minSize; size <= maxSize; size *= sizeScale) {
		newSample.setSize(size);
		int halfSize = size / 2;
		int minX = halfSize;
		int minY = halfSize;
		int maxX = image.cols - size + halfSize;
		int maxY = image.rows - size + halfSize;
		int step = (int)(stepSize * size + 0.5f);
		for (int x = minX; x < maxX; x += step) {
			newSample.setX(x);
			for (int y = minY; y < maxY; y += step) {
				newSample.setY(y);
				newSamples.push_back(newSample);
			}
		}
	}
}

} /* namespace tracking */
