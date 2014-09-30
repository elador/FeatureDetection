/*
 * CondensationTracker.cpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#include "condensation/CondensationTracker.hpp"
#include "condensation/Sample.hpp"
#include "condensation/TransitionModel.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/ResamplingAlgorithm.hpp"
#include "condensation/StateExtractor.hpp"
#include "condensation/StateValidator.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include <stdexcept>

using imageprocessing::VersionedImage;
using cv::Mat;
using cv::Rect;
using boost::optional;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::runtime_error;
using std::invalid_argument;

namespace condensation {

CondensationTracker::CondensationTracker(
		std::shared_ptr<TransitionModel> transitionModel, std::shared_ptr<MeasurementModel> measurementModel,
		std::shared_ptr<ResamplingAlgorithm> resamplingAlgorithm, std::shared_ptr<StateExtractor> extractor,
		size_t count, double randomRate, int minSize, int maxSize) :
				count(count),
				randomRate(randomRate),
				minSize(minSize),
				maxSize(maxSize),
				samples(),
				oldSamples(),
				state(),
				adapted(false),
				image(make_shared<VersionedImage>()),
				transitionModel(transitionModel),
				measurementModel(measurementModel),
				resamplingAlgorithm(resamplingAlgorithm),
				extractor(extractor),
				validators(),
				generator(boost::mt19937(time(0))),
				intDistribution(boost::uniform_int<>()),
				realDistribution(boost::uniform_real<>()) {
	if (minSize < 1)
		throw invalid_argument("ResamplingSampler: the minimum size must be greater than zero");
	if (maxSize < minSize)
		throw invalid_argument("ResamplingSampler: the maximum size must not be smaller than the minimum size");
	setRandomRate(randomRate);
	shared_ptr<StateValidator> validator = std::dynamic_pointer_cast<StateValidator>(measurementModel);
	if (validator)
		addValidator(validator);
}

void CondensationTracker::reset() {
	measurementModel->reset();
}

optional<Rect> CondensationTracker::initialize(const Mat& imageData, const Rect& positionData) {
	if (minSize > imageData.cols || minSize > imageData.rows)
		throw runtime_error("AdaptiveCondensationTracker: the minimum size must not be greater than the width and height of the image");
	if (maxSize > imageData.cols || maxSize > imageData.rows)
		maxSize = std::min(imageData.cols, imageData.rows);
	image->setData(imageData);
	samples.clear();
	Sample::setAspectRatio(positionData.width, positionData.height);
	state = make_shared<Sample>(
			positionData.x + positionData.width / 2, positionData.y + positionData.height / 2, positionData.width);
	transitionModel->init(imageData);
	measurementModel->initialize(image, *state);
	if (measurementModel->isUsable()) {
		for (size_t i = 0; i < count; ++i)
			samples.push_back(state);
	}
	if (measurementModel->isUsable())
		return optional<Rect>(state->getBounds());
	return optional<Rect>();
}

optional<Rect> CondensationTracker::process(const Mat& imageData) {
	if (!measurementModel->isUsable())
		throw runtime_error("AdaptiveCondensationTracker: Is not usable (was not initialized or was resetted)");
	image->setData(imageData);
	samples.swap(oldSamples);
	samples.clear();
	// sampling
	resamplingAlgorithm->resample(oldSamples, (int)((1 - randomRate) * count), samples);
	transitionModel->predict(samples, image->getData(), state);
	while (samples.size() < count) {
		shared_ptr<Sample> newSample = make_shared<Sample>();
		sampleValues(*newSample, image->getData());
		samples.push_back(newSample);
	}
	// evaluate samples and extract state
	measurementModel->evaluate(image, samples);
	state = extractor->extract(samples);
	// validate target state
	if (state) {
		for (shared_ptr<StateValidator>& validator : validators) {
			if (!validator->isValid(*state, samples, image)) {
				state.reset();
				break;
			}
		}
	}
	// update model
	if (state)
		adapted = measurementModel->adapt(image, samples, *state);
	else
		adapted = measurementModel->adapt(image, samples);
	// return position
	if (state)
		return optional<Rect>(state->getBounds());
	return optional<Rect>();
}

void CondensationTracker::sampleValues(Sample& sample, const Mat& image) {
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

bool CondensationTracker::hasAdapted() {
	return adapted;
}

shared_ptr<Sample> CondensationTracker::getState() {
	return state;
}

const vector<shared_ptr<Sample>>& CondensationTracker::getSamples() const {
	return samples;
}

void CondensationTracker::addValidator(shared_ptr<StateValidator> validator) {
	validators.push_back(validator);
}

size_t CondensationTracker::getCount() {
	return count;
}

void CondensationTracker::setCount(size_t count) {
	this->count = count;
}

double CondensationTracker::getRandomRate() {
	return randomRate;
}

void CondensationTracker::setRandomRate(double randomRate) {
	this->randomRate = std::max(0.0, std::min(1.0, randomRate));
}

} /* namespace condensation */
