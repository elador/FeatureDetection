/*
 * SelfLearningStrategy.h
 *
 *  Created on: 21.09.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGSTRATEGY_H_
#define SELFLEARNINGSTRATEGY_H_

#include "tracking/LearningStrategy.h"

namespace tracking {

/**
 * Learning strategy that leaves the selection of training samples to the measurement model.
 */
class SelfLearningStrategy : public LearningStrategy {
public:

	/**
	 * Constructs a new self-learning strategy.
	 */
	explicit SelfLearningStrategy();

	~SelfLearningStrategy();

	void update(LearningMeasurementModel& model, FdImage* image, const std::vector<Sample>& samples);

	void update(LearningMeasurementModel& model, FdImage* image,
			const std::vector<Sample>& samples, const Sample& position);
};

} /* namespace tracking */
#endif /* SELFLEARNINGSTRATEGY_H_ */
