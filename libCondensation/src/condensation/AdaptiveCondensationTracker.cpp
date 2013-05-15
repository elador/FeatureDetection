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
#include "condensation/PositionExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include <stdexcept>

using std::make_shared;
using std::runtime_error;

namespace condensation {

AdaptiveCondensationTracker::AdaptiveCondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<AdaptiveMeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor, int initialCount) :
				initialCount(initialCount),
				samples(),
				oldSamples(),
				state(),
				image(make_shared<VersionedImage>()),
				sampler(sampler),
				measurementModel(measurementModel),
				extractor(extractor) {}

AdaptiveCondensationTracker::~AdaptiveCondensationTracker() {}

bool AdaptiveCondensationTracker::initialize(const Mat& imageData, const Rect& positionData) {
	image->setData(imageData);
	samples.clear();
	Sample position(positionData.x + positionData.width / 2, positionData.y + positionData.height / 2, positionData.width);
	measurementModel->adapt(image, samples, position);
	if (measurementModel->isUsable()) {
		for (int i = 0; i < initialCount; ++i)
			samples.push_back(position);
	}
	return measurementModel->isUsable();
}

optional<Rect> AdaptiveCondensationTracker::process(const Mat& imageData) {
	if (!measurementModel->isUsable())
		throw runtime_error("AdaptiveCondensationTracker: Is not usable (was not initialized or was resetted)");
	image->setData(imageData);
	oldSamples = samples;
	sampler->sample(oldSamples, image->getData(), samples);
	// evaluate samples and extract position
	measurementModel->evaluate(image, samples);
	state = extractor->extract(samples);
	// update model
	if (state)
		measurementModel->adapt(image, samples, *state);
	else
		measurementModel->adapt(image, samples);
	// return position
	if (state)
		return optional<Rect>(state->getBounds());
	return optional<Rect>();
}

} /* namespace condensation */
