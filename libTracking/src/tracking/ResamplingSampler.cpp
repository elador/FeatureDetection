/*
 * ResamplingSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "tracking/ResamplingSampler.h"
#include "tracking/ResamplingAlgorithm.h"
#include "tracking/TransitionModel.h"
#include "FdImage.h"
#include <algorithm>

namespace tracking {

ResamplingSampler::ResamplingSampler(unsigned int count, double randomRate,
		ResamplingAlgorithm* resamplingAlgorithm, TransitionModel* transitionModel) :
				count(count),
				randomRate(randomRate),
				resamplingAlgorithm(resamplingAlgorithm),
				transitionModel(transitionModel),
				generator(boost::mt19937(time(0))),
				distribution(boost::uniform_int<>()) {
	randomRate = std::max(0.0, std::min(1.0, randomRate));
}

ResamplingSampler::~ResamplingSampler() {
	delete resamplingAlgorithm;
	delete transitionModel;
}

void ResamplingSampler::sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
			const FdImage* image, std::vector<Sample>& newSamples) {
	resamplingAlgorithm->resample(samples, (1 - randomRate) * count, newSamples);
	// predict the samples
	for (std::vector<Sample>::iterator sit = newSamples.begin(); sit < newSamples.end(); ++sit) {
		Sample& sample = *sit;
		transitionModel->predict(sample, offset);
		if (!isValid(sample, image))
			sampleValid(sample, image);
	}
	// add new random samples
	Sample newSample;
	while (newSamples.size() < count) {
		sampleValid(newSample, image);
		newSamples.push_back(newSample);
	}
}

bool ResamplingSampler::isValid(const Sample& sample, const FdImage* image) {
	int minSize = 0.1 * std::min(image->w, image->h);
	int maxSize = 0.8 * std::min(image->w, image->h);
	int halfSize = sample.getSize() / 2;
	int x = sample.getX() - halfSize;
	int y = sample.getY() - halfSize;
	return sample.getSize() >= minSize && sample.getSize() <= maxSize
			&& x >= 0 && x + sample.getSize() <= image->w
			&& y >= 0 && y + sample.getSize() <= image->h;
}

void ResamplingSampler::sampleValid(Sample& sample, const FdImage* image) {
	int minSize = 0.1 * std::min(image->w, image->h);
	int maxSize = 0.8 * std::min(image->w, image->h);
	int size = distribution(generator, maxSize - minSize) + minSize;
	int halfSize = size / 2;
	sample.setSize(size);
	sample.setX(distribution(generator, image->w - size) + halfSize);
	sample.setY(distribution(generator, image->h - size) + halfSize);
}

} /* namespace tracking */
