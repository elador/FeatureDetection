/*
 * LearningMeasurementModel.h
 *
 *  Created on: 30.07.2012
 *      Author: poschmann
 */

#ifndef LEARNINGMEASUREMENTMODEL_H_
#define LEARNINGMEASUREMENTMODEL_H_

#include "tracking/MeasurementModel.h"

namespace tracking {

/**
 * Measurement model for samples that may update itself when provided with labeled samples.
 */
class LearningMeasurementModel : public MeasurementModel {
public:
	virtual ~LearningMeasurementModel() {}

	virtual void evaluate(FdImage* image, std::vector<Sample>& samples) = 0;

	/**
	 * Adds a sample with a positive label to the training data.
	 *
	 * @param[in] sample The positive sample.
	 */
	virtual void addPositive(const Sample& sample) = 0;

	/**
	 * Adds a sample with a negative label to the training data.
	 *
	 * @param[in] sample The negative sample.
	 */
	virtual void addNegative(const Sample& sample) = 0;
};

} /* namespace tracking */
#endif /* LEARNINGMEASUREMENTMODEL_H_ */
