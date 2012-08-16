/*
 * GridSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "tracking/GridSampler.h"
#include "FdImage.h"
#include <algorithm>

namespace tracking {

GridSampler::GridSampler(float minSize, float maxSize, float sizeScale, float stepSize) :
		minSize(minSize), maxSize(maxSize), sizeScale(sizeScale), stepSize(stepSize) {
	minSize = std::min(1.0f, std::max(0.0f, minSize));
	maxSize = std::min(1.0f, std::max(0.0f, maxSize));
	sizeScale = std::max(1.05f, sizeScale);
	stepSize = std::max(0.0f, stepSize);
}

GridSampler::~GridSampler() {}

void GridSampler::sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
			const FdImage* image, std::vector<Sample>& newSamples) {
	newSamples.clear();
	int minSize = this->minSize * std::min(image->w, image->h);
	int maxSize = this->maxSize * std::min(image->w, image->h);
	Sample newSample;
	for (int size = minSize; size <= maxSize; size *= sizeScale) {
		newSample.setSize(size);
		int halfSize = size / 2;
		int minX = halfSize;
		int minY = halfSize;
		int maxX = image->w - size + halfSize;
		int maxY = image->h - size + halfSize;
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
