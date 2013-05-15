/*
 * TransitionModel.hpp
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef TRANSITIONMODEL_HPP_
#define TRANSITIONMODEL_HPP_

#include <vector>

using std::vector;

namespace condensation {

class Sample;

/**
 * Transition model that predicts the new state of samples.
 */
class TransitionModel {
public:

	virtual ~TransitionModel() {}

	/**
	 * Predicts the new state of the sample and adds some noise.
	 *
	 * @param[in,out] sample The sample.
	 */
	virtual void predict(Sample& sample) = 0;
};

} /* namespace condensation */
#endif /* TRANSITIONMODEL_HPP_ */
