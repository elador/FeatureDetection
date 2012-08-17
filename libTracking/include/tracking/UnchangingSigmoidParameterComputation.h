/*
 * UnchangingSigmoidParameterComputation.h
 *
 *  Created on: 16.08.2012
 *      Author: poschmann
 */

#ifndef UNCHANGINGSIGMOIDPARAMETERCOMPUTATION_H_
#define UNCHANGINGSIGMOIDPARAMETERCOMPUTATION_H_

#include "tracking/SigmoidParameterComputation.h"

namespace tracking {

/**
 * Does not compute new sigmoid parameters, but takes the existing parameters from the SVM.
 */
class UnchangingSigmoidParameterComputation : public SigmoidParameterComputation {
public:

	/**
	 * Constructs a new unchanging sigmoid parameter computation.
	 */
	UnchangingSigmoidParameterComputation();

	~UnchangingSigmoidParameterComputation();

	std::pair<double, double> computeSigmoidParameters(const ChangableDetectorSvm& svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount);
};

} /* namespace tracking */
#endif /* UNCHANGINGSIGMOIDPARAMETERCOMPUTATION_H_ */
