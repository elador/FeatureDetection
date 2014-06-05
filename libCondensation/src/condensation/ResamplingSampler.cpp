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

using cv::Mat;
using std::vector;
using std::shared_ptr;
using std::invalid_argument;
using std::make_shared;

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
				intDistribution(boost::uniform_int<>()),
				realDistribution(boost::uniform_real<>()) {
	if (minSize < 1)
		throw invalid_argument("ResamplingSampler: the minimum size must be greater than zero");
	if (maxSize < minSize)
		throw invalid_argument("ResamplingSampler: the maximum size must not be smaller than the minimum size");
	setRandomRate(randomRate);
}

void ResamplingSampler::init(const Mat& image) {
	if (minSize > image.cols || minSize > image.rows)
		throw invalid_argument("ResamplingSampler: the minimum size must not be greater than the width and height of the image");
	if (maxSize > image.cols || maxSize > image.rows)
		maxSize = std::min(image.cols, image.rows);
	transitionModel->init(image);
}

void ResamplingSampler::sample(const vector<shared_ptr<Sample>>& samples, vector<shared_ptr<Sample>>& newSamples,
		const Mat& image, const shared_ptr<Sample> target) {
	resamplingAlgorithm->resample(samples, (int)((1 - randomRate) * count), newSamples);
	transitionModel->predict(newSamples, image, target);
	while (newSamples.size() < count) {
		shared_ptr<Sample> newSample = make_shared<Sample>();
		sampleValues(*newSample, image);
		newSamples.push_back(newSample);
	}
}

void ResamplingSampler::sampleValues(Sample& sample, const Mat& image) {
	double sizeFactor = realDistribution(generator) * (static_cast<double>(maxSize) / static_cast<double>(minSize) - 1.0) + 1.0;
	int size = cvRound(sizeFactor * minSize);
	int halfSize = size / 2;
	sample.setSize(size);
	sample.setX(intDistribution(generator, image.cols - size) + halfSize);
	sample.setY(intDistribution(generator, image.rows - size) + halfSize);
	sample.setVx(0);
	sample.setVy(0);
	sample.setVSize(1);
}

} /* namespace condensation */
