/*
 * LearningMeasurementModel.h
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef LEARNINGMEASUREMENTMODEL_H_
#define LEARNINGMEASUREMENTMODEL_H_

#include "tracking/MeasurementModel.h"

namespace tracking {

/**
 * Measurement model that is able to learn.
 */
class LearningMeasurementModel : public MeasurementModel {
public:

	virtual ~LearningMeasurementModel() {}

	virtual void evaluate(FdImage* image, std::vector<Sample>& samples) = 0;

	/**
	 * Determines whether the updated dynamic model is used for evaluation (as opposed to the
	 * initial default model it may fall back to).
	 *
	 * @return True if the updated dynamic model is used for evaluation, false otherwise.
	 */
	virtual bool isUsingDynamicModel() = 0;

	/**
	 * Resets the model to its original non-updated state.
	 */
	virtual void reset() = 0;

	/**
	 * Updates the measurement model without adding new samples.
	 */
	virtual void update() = 0;

	/**
	 * Updates the measurement model with new samples.
	 *
	 * @param[in] image The image.
	 * @param[in] positiveSamples The new positive samples.
	 * @param[in] negativeSamples The new negative samples.
	 */
	virtual void update(FdImage* image, std::vector<Sample>& positiveSamples, std::vector<Sample>& negativeSamples) = 0;
};

} /* namespace tracking */
#endif /* LEARNINGMEASUREMENTMODEL_H_ */
