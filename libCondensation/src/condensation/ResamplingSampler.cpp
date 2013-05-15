/*
 * ResamplingSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "condensation/ResamplingSampler.hpp"
#include "condensation/Sample.hpp"
#include "condensation/ResamplingAlgorithm.hpp"
#include "condensation/TransitionModel.hpp"
#include <algorithm>
#include <ctime>
#include <stdexcept>

using std::invalid_argument;

namespace condensation {

ResamplingSampler::ResamplingSampler(unsigned int count, double randomRate,
		shared_ptr<ResamplingAlgorithm> resamplingAlgorithm, shared_ptr<TransitionModel> transitionModel, int minSize, int maxSize) :
				count(count),
				randomRate(randomRate),
				resamplingAlgorithm(resamplingAlgorithm),
				transitionModel(transitionModel),
				minSize(minSize),
				maxSize(maxSize),
				generator(boost::mt19937(time(0))),
				distribution(boost::uniform_int<>()) {
	if (minSize < 1)
		throw invalid_argument("ResamplingSampler: the minimum size must be greater than zero");
	if (maxSize < minSize)
		throw invalid_argument("ResamplingSampler: the maximum size must not be smaller than the minimum size");
	setRandomRate(randomRate);
}

ResamplingSampler::~ResamplingSampler() {}

void ResamplingSampler::sample(const vector<Sample>& samples, const Mat& image, vector<Sample>& newSamples) {
	unsigned int count = this->count;
	resamplingAlgorithm->resample(samples, (int)((1 - randomRate) * count), newSamples);
	// predict the samples
	for (auto sample = newSamples.begin(); sample != newSamples.end(); ++sample) {
		transitionModel->predict(*sample);
		if (!isValid(*sample, image))
			sampleValid(*sample, image);
	}
	// add new random samples
	Sample newSample;
	while (newSamples.size() < count) {
		sampleValid(newSample, image);
		newSamples.push_back(newSample);
	}
}

bool ResamplingSampler::isValid(const Sample& sample, const Mat& image) {
	int halfSize = sample.getSize() / 2;
	int x = sample.getX() - halfSize;
	int y = sample.getY() - halfSize;
	return sample.getSize() >= minSize && sample.getSize() <= maxSize
			&& x >= 0 && x + sample.getSize() <= image.cols
			&& y >= 0 && y + sample.getSize() <= image.rows;
}

void ResamplingSampler::sampleValid(Sample& sample, const Mat& image) {
	int size = distribution(generator, maxSize - minSize) + minSize;
	int halfSize = size / 2;
	sample.setSize(size);
	sample.setX(distribution(generator, image.cols - size) + halfSize);
	sample.setY(distribution(generator, image.rows - size) + halfSize);
	sample.setVx(0);
	sample.setVy(0);
	sample.setVSize(0);
}

} /* namespace condensation */
