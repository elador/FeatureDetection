/*
 * PartiallyAdaptiveCondensationTracker.cpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#include "condensation/PartiallyAdaptiveCondensationTracker.hpp"
#include "condensation/Sample.hpp"
#include "condensation/Sampler.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
#include "condensation/PositionExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"

using std::make_shared;

namespace condensation {

PartiallyAdaptiveCondensationTracker::PartiallyAdaptiveCondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<MeasurementModel> initialMeasurementModel, shared_ptr<AdaptiveMeasurementModel> measurementModel,
		shared_ptr<PositionExtractor> extractor) :
				samples(),
				oldSamples(),
				oldPosition(),
				offset(3),
				useAdaptiveModel(true),
				usedAdaptiveModel(false),
				image(make_shared<VersionedImage>()),
				sampler(sampler),
				initialMeasurementModel(initialMeasurementModel),
				measurementModel(measurementModel),
				extractor(extractor) {
	offset.push_back(0);
	offset.push_back(0);
	offset.push_back(0);
}

PartiallyAdaptiveCondensationTracker::~PartiallyAdaptiveCondensationTracker() {}

optional<Rect> PartiallyAdaptiveCondensationTracker::process(const Mat& imageData) {
	image->setData(imageData);
	oldSamples = samples;
	sampler->sample(oldSamples, offset, image->getData(), samples);
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
		return optional<Rect>(position->getBounds());
	return optional<Rect>();
}

void PartiallyAdaptiveCondensationTracker::setUseAdaptiveModel(bool useAdaptive) {
	useAdaptiveModel = useAdaptive;
	if (!useAdaptive)
		measurementModel->reset();
}

} /* namespace condensation */
