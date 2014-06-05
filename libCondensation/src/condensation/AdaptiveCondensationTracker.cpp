/*
 * AdaptiveCondensationTracker.cpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#include "condensation/AdaptiveCondensationTracker.hpp"
#include "condensation/Sample.hpp"
#include "condensation/Sampler.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
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

namespace condensation {

AdaptiveCondensationTracker::AdaptiveCondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<AdaptiveMeasurementModel> measurementModel, shared_ptr<StateExtractor> extractor, int initialCount) :
				initialCount(initialCount),
				samples(),
				oldSamples(),
				state(),
				adapted(false),
				image(make_shared<VersionedImage>()),
				sampler(sampler),
				measurementModel(measurementModel),
				extractor(extractor),
				validators() {
	shared_ptr<StateValidator> validator = std::dynamic_pointer_cast<StateValidator>(measurementModel);
	if (validator)
		addValidator(validator);
}

void AdaptiveCondensationTracker::reset() {
	measurementModel->reset();
}

optional<Rect> AdaptiveCondensationTracker::initialize(const Mat& imageData, const Rect& positionData) {
	image->setData(imageData);
	samples.clear();
	Sample::setAspectRatio(positionData.width, positionData.height);
	state = make_shared<Sample>(
			positionData.x + positionData.width / 2, positionData.y + positionData.height / 2, positionData.width);
	sampler->init(imageData);
	measurementModel->initialize(image, *state);
	if (measurementModel->isUsable()) {
		for (int i = 0; i < initialCount; ++i)
			samples.push_back(state);
	}
	if (measurementModel->isUsable())
		return optional<Rect>(state->getBounds());
	return optional<Rect>();
}

optional<Rect> AdaptiveCondensationTracker::process(const Mat& imageData) {
	if (!measurementModel->isUsable())
		throw runtime_error("AdaptiveCondensationTracker: Is not usable (was not initialized or was resetted)");
	image->setData(imageData);
	samples.swap(oldSamples);
	samples.clear();
	sampler->sample(oldSamples, samples, image->getData(), state);
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

bool AdaptiveCondensationTracker::hasAdapted() {
	return adapted;
}

shared_ptr<Sample> AdaptiveCondensationTracker::getState() {
	return state;
}

const vector<shared_ptr<Sample>>& AdaptiveCondensationTracker::getSamples() const {
	return samples;
}

shared_ptr<Sampler> AdaptiveCondensationTracker::getSampler() {
	return sampler;
}

void AdaptiveCondensationTracker::setSampler(shared_ptr<Sampler> sampler) {
	this->sampler = sampler;
}

void AdaptiveCondensationTracker::addValidator(shared_ptr<StateValidator> validator) {
	validators.push_back(validator);
}

} /* namespace condensation */
