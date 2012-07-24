/*
 * CondensationTracker.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "tracking/CondensationTracker.h"

#include "tracking/ResamplingAlgorithm.h"
#include "tracking/TransitionModel.h"
#include "tracking/MeasurementModel.h"
#include "tracking/PositionExtractor.h"
#include "FdImage.h"
#include <algorithm>
#include <iostream>

namespace tracking {

CondensationTracker::CondensationTracker(unsigned int count, double randomRate, ResamplingAlgorithm* resamplingAlgorithm,
		TransitionModel* transitionModel, MeasurementModel* measurementModel, PositionExtractor* extractor) :
				count(count),
				randomRate(randomRate),
				samples(count),
				oldSamples(count),
				offset(3),
				resamplingAlgorithm(resamplingAlgorithm),
				transitionModel(transitionModel),
				measurementModel(measurementModel),
				extractor(extractor),
				generator(boost::mt19937(time(0))),
				distribution(boost::uniform_int<>()) {
	randomRate = std::max(0.0, std::min(1.0, randomRate));
	samples.clear();
	oldSamples.clear();
	offset.push_back(0);
	offset.push_back(0);
	offset.push_back(0);
}

CondensationTracker::~CondensationTracker() {
	delete resamplingAlgorithm;
	delete transitionModel;
	delete measurementModel;
	delete extractor;
}

boost::optional<Rectangle> CondensationTracker::process(FdImage* image) {
	oldSamples = samples;
	resamplingAlgorithm->resample(oldSamples, (1 - randomRate) * count, samples);
	// predict the samples
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		transitionModel->predict(sample, this->offset);
		if (!isValid(sample, image))
			sampleValid(sample, image);
	}
	// add new random samples
	Sample newSample;
	while (samples.size() < count) {
		sampleValid(newSample, image);
		samples.push_back(newSample);
	}
	// evaluate samples and extract position
	measurementModel->evaluate(image, samples);
	boost::optional<Sample> position = extractor->extract(samples);
	if (position)
		return boost::optional<Rectangle>(position->getBounds());
	return boost::optional<Rectangle>();
}

bool CondensationTracker::isValid(const Sample& sample, const FdImage* image) {
	int minSize = 0.1 * std::min(image->w, image->h);
	int maxSize = 0.8 * std::min(image->w, image->h);
	int halfSize = sample.getSize() / 2;
	int x = sample.getX() - halfSize;
	int y = sample.getY() - halfSize;
	return sample.getSize() >= minSize && sample.getSize() <= maxSize
			&& x >= 0 && x + sample.getSize() <= image->w
			&& y >= 0 && y + sample.getSize() <= image->h;
}

void CondensationTracker::sampleValid(Sample& sample, const FdImage* image) {
	int minSize = 0.1 * std::min(image->w, image->h);
	int maxSize = 0.8 * std::min(image->w, image->h);
	int size = distribution(generator, maxSize - minSize) + minSize;
	int halfSize = size / 2;
	sample.setSize(size);
	sample.setX(distribution(generator, image->w - size) + halfSize);
	sample.setY(distribution(generator, image->h - size) + halfSize);
}

} /* namespace tracking */
