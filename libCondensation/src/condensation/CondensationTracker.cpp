/*
 * CondensationTracker.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "condensation/Sample.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/Sampler.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/PositionExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"

using std::make_shared;

namespace condensation {

CondensationTracker::CondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<MeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor) :
				samples(),
				oldSamples(),
				oldPosition(),
				offset(3),
				image(make_shared<VersionedImage>()),
				sampler(sampler),
				measurementModel(measurementModel),
				extractor(extractor) {
	offset.push_back(0);
	offset.push_back(0);
	offset.push_back(0);
}

CondensationTracker::~CondensationTracker() {}

optional<Rect> CondensationTracker::process(const Mat& imageData) {
	image->setData(imageData);
	oldSamples = samples;
	sampler->sample(oldSamples, offset, image->getData(), samples);
	// evaluate samples and extract position
	measurementModel->evaluate(image, samples);
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
	// return position
	if (position)
		return optional<Rect>(position->getBounds());
	return optional<Rect>();
}

} /* namespace condensation */
