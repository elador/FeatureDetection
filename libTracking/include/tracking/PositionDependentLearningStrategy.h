/*
 * PositionDependentLearningStrategy.h
 *
 *  Created on: 21.09.2012
 *      Author: poschmann
 */

#ifndef POSITIONDEPENDENTLEARNINGSTRATEGY_H_
#define POSITIONDEPENDENTLEARNINGSTRATEGY_H_

#include "tracking/LearningStrategy.h"

namespace tracking {

/**
 * Learning strategy that creates samples around the object position as training samples.
 */
class PositionDependentLearningStrategy : public LearningStrategy {
public:

	/**
	 * Creates a new position dependent learning strategy.
	 */
	explicit PositionDependentLearningStrategy();

	~PositionDependentLearningStrategy();

	void update(LearningMeasurementModel& model, FdImage* image, const std::vector<Sample>& samples);

	void update(LearningMeasurementModel& model, FdImage* image,
			const std::vector<Sample>& samples, const Sample& position);
};

} /* namespace tracking */
#endif /* POSITIONDEPENDENTLEARNINGSTRATEGY_H_ */
