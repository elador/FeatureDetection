/*
 * AdaptiveCondensationTracker.cpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#include "tracking/AdaptiveCondensationTracker.h"
#include "tracking/Rectangle.h"
#include "tracking/Sample.h"
#include "tracking/Sampler.h"
#include "tracking/MeasurementModel.h"
#include "tracking/AdaptiveMeasurementModel.h"
#include "tracking/PositionExtractor.h"

namespace tracking {

AdaptiveCondensationTracker::AdaptiveCondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<MeasurementModel> initialMeasurementModel, shared_ptr<AdaptiveMeasurementModel> measurementModel,
		shared_ptr<PositionExtractor> extractor) :
				samples(),
				oldSamples(),
				oldPosition(),
				offset(3),
				useAdaptiveModel(true),
				usedAdaptiveModel(false),
				sampler(sampler),
				initialMeasurementModel(initialMeasurementModel),
				measurementModel(measurementModel),
				extractor(extractor) {
	offset.push_back(0);
	offset.push_back(0);
	offset.push_back(0);
}

AdaptiveCondensationTracker::~AdaptiveCondensationTracker() {}

optional<Rectangle> AdaptiveCondensationTracker::process(const Mat& image) {
	oldSamples = samples;
	sampler->sample(oldSamples, offset, image, samples);
	// evaluate samples and extract position
	if (useAdaptiveModel && measurementModel->isUsable()) {
		measurementModel->evaluate(image, samples);
		usedAdaptiveModel = true;
	} else {
		initialMeasurementModel->evaluate(image, samples);
		usedAdaptiveModel = false;
	}
	optional<Sample> position = extractor->extract(samples);
	// update offset
	if (oldPosition && position) {
		offset[0] = position->getX() - oldPosition->getX();
		offset[1] = position->getY() - oldPosition->getY();
		offset[2] = position->getSize() - oldPosition->getSize();
	} else {
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
	}
	oldPosition = position;
	// update model
	if (useAdaptiveModel) {
		if (position)
			measurementModel->adapt(image, samples, *position);
		else
			measurementModel->adapt(image, samples);
	}
	// return position
	if (position)
		return optional<Rectangle>(position->getBounds());
	return optional<Rectangle>();
}

void AdaptiveCondensationTracker::setUseAdaptiveModel(bool useAdaptive) {
	useAdaptiveModel = useAdaptive;
	if (!useAdaptive)
		measurementModel->reset();
}

} /* namespace tracking */
