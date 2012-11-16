/*
 * ResamplingSampler.cpp
 *
 *  Created on: 15.08.2012
 *      Author: poschmann
 */

#include "tracking/ResamplingSampler.h"
#include "tracking/ResamplingAlgorithm.h"
#include "tracking/TransitionModel.h"
#include <algorithm>
#include <ctime>

namespace tracking {

ResamplingSampler::ResamplingSampler(unsigned int count, double randomRate,
		shared_ptr<ResamplingAlgorithm> resamplingAlgorithm, shared_ptr<TransitionModel> transitionModel,
		float minSize, float maxSize) :
				count(count),
				randomRate(randomRate),
				resamplingAlgorithm(resamplingAlgorithm),
				transitionModel(transitionModel),
				minSize(minSize),
				maxSize(maxSize),
				generator(boost::mt19937(time(0))),
				distribution(boost::uniform_int<>()) {
	setRandomRate(randomRate);
}

ResamplingSampler::~ResamplingSampler() {}

void ResamplingSampler::sample(const std::vector<Sample>& samples, const std::vector<double>& offset,
			const cv::Mat& image, std::vector<Sample>& newSamples) {
	unsigned int count = this->count;
	resamplingAlgorithm->resample(samples, (int)((1 - randomRate) * count), newSamples);
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

bool ResamplingSampler::isValid(const Sample& sample, const cv::Mat& image) {
	int minSize = (int)(this->minSize * std::min(image.cols, image.rows));
	int maxSize = (int)(this->maxSize * std::min(image.cols, image.rows));
	int halfSize = sample.getSize() / 2;
	int x = sample.getX() - halfSize;
	int y = sample.getY() - halfSize;
	return sample.getSize() >= minSize && sample.getSize() <= maxSize
			&& x >= 0 && x + sample.getSize() <= image.cols
			&& y >= 0 && y + sample.getSize() <= image.rows;
}

void ResamplingSampler::sampleValid(Sample& sample, const cv::Mat& image) {
	int minSize = (int)(this->minSize * std::min(image.cols, image.rows));
	int maxSize = (int)(this->maxSize * std::min(image.cols, image.rows));
	int size = distribution(generator, maxSize - minSize) + minSize;
	int halfSize = size / 2;
	sample.setSize(size);
	sample.setX(distribution(generator, image.cols - size) + halfSize);
	sample.setY(distribution(generator, image.rows - size) + halfSize);
}

} /* namespace tracking */
