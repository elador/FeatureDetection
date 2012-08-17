/*
 * ExactSigmoidParameterComputation.h
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#ifndef EXACTSIGMOIDPARAMETERCOMPUTATION_H_
#define EXACTSIGMOIDPARAMETERCOMPUTATION_H_

#include "tracking/SigmoidParameterComputation.h"

namespace tracking {

/**
 * Computes the exact parameters of the sigmoid function, so that the function crosses two
 * certain points. The first point is the mean SVM output of the positive samples with a
 * value of 0.99, the second point is in between the mean SVM outputs of the positive and
 * negative samples with a value of 0.01.
 */
class ExactSigmoidParameterComputation : public SigmoidParameterComputation {
public:

	/**
	 * Constructs a new extact sigmoid parameter computation.
	 */
	explicit ExactSigmoidParameterComputation();

	~ExactSigmoidParameterComputation();

	std::pair<double, double> computeSigmoidParameters(const ChangableDetectorSvm& svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount);
};

} /* namespace tracking */
#endif /* EXACTSIGMOIDPARAMETERCOMPUTATION_H_ */
