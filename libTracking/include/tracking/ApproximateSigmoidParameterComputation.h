/*
 * ApproximateSigmoidParameterComputation.h
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#ifndef APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_
#define APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_

#include "tracking/SigmoidParameterComputation.h"

namespace tracking {

/**
 * Computes the approximate parameters of the sigmoid function, so that the function
 * approximately crosses two certain points. The first point is the mean SVM output of
 * the positive samples with a value of 0.99, the second point is in between the mean
 * SVM outputs of the positive and negative samples with a value of 0.01.
 */
class ApproximateSigmoidParameterComputation : public SigmoidParameterComputation {
public:

	/**
	 * Constructs a new approximate sigmoid parameter computation.
	 */
	explicit ApproximateSigmoidParameterComputation();
	virtual ~ApproximateSigmoidParameterComputation();

	std::pair<double, double> computeSigmoidParameters(ChangableDetectorSvm* svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount);
};

} /* namespace tracking */
#endif /* APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_ */
