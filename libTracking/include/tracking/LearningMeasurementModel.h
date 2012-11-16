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

	virtual void evaluate(cv::Mat& image, std::vector<Sample>& samples) = 0;

	/**
	 * Determines whether the updated dynamic model was used for the previous evaluation
	 * (as opposed to the initial default model it may fall back to).
	 *
	 * @return True if the updated dynamic model was used for evaluation, false otherwise.
	 */
	virtual bool wasUsingDynamicModel() = 0;

	/**
	 * Determines whether the updated dynamic model will be used for the next evaluation
	 * (as opposed to the initial default model it may fall back to).
	 *
	 * @return True if the updated dynamic model will be used for evaluation, false otherwise.
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
	virtual void update(cv::Mat& image, std::vector<Sample>& positiveSamples, std::vector<Sample>& negativeSamples) = 0;
};

} /* namespace tracking */
#endif /* LEARNINGMEASUREMENTMODEL_H_ */
