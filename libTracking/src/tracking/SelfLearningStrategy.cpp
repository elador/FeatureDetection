/*
 * SelfLearningStrategy.cpp
 *
 *  Created on: 21.09.2012
 *      Author: poschmann
 */

#include "tracking/SelfLearningStrategy.h"
#include "tracking/LearningMeasurementModel.h"

namespace tracking {

SelfLearningStrategy::SelfLearningStrategy() {}

SelfLearningStrategy::~SelfLearningStrategy() {}

void SelfLearningStrategy::update(LearningMeasurementModel& model, FdImage* image, const std::vector<Sample>& samples) {
	model.update();
}

void SelfLearningStrategy::update(LearningMeasurementModel& model, FdImage* image,
		const std::vector<Sample>& samples, const Sample& position) {
	model.update();
}

} /* namespace tracking */
