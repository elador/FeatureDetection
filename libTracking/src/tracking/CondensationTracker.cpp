/*
 * CondensationTracker.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "tracking/CondensationTracker.h"
#include "tracking/Sampler.h"
#include "tracking/MeasurementModel.h"
#include "tracking/PositionExtractor.h"
#include "FdImage.h"

namespace tracking {

CondensationTracker::CondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<MeasurementModel> measurementModel, shared_ptr<PositionExtractor> extractor) :
				samples(),
				oldSamples(),
				oldPosition(),
				offset(3),
				sampler(sampler),
				measurementModel(measurementModel),
				extractor(extractor) {
	offset.push_back(0);
	offset.push_back(0);
	offset.push_back(0);
}

CondensationTracker::~CondensationTracker() {}

boost::optional<Rectangle> CondensationTracker::process(FdImage* image) {
	oldSamples = samples;
	sampler->sample(oldSamples, offset, image, samples);
	// evaluate samples and extract position
	measurementModel->evaluate(image, samples);
	boost::optional<Sample> position = extractor->extract(samples);
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
		return boost::optional<Rectangle>(position->getBounds());
	return boost::optional<Rectangle>();
}

} /* namespace tracking */
