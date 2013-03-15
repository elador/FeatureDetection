/*
 * TransitionModel.h
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef TRANSITIONMODEL_H_
#define TRANSITIONMODEL_H_

#include <vector>

using std::vector;

namespace condensation {

class Sample;

/**
 * Transition model that predicts the new positions of samples.
 */
class TransitionModel {
public:

	virtual ~TransitionModel() {}

	/**
	 * Moves the sample according to the prediction and adds some noise.
	 *
	 * @param[in,out] sample The sample.
	 * @param[in] offset The movement of the tracked object's center of the previous time step.
	 */
	virtual void predict(Sample& sample, const vector<double>& offset) = 0;
};

} /* namespace condensation */
#endif /* TRANSITIONMODEL_H_ */
