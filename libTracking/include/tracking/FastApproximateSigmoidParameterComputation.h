/*
 * FastApproximateSigmoidParameterComputation.h
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#ifndef FASTAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_
#define FASTAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_

#include "tracking/ApproximateSigmoidParameterComputation.h"

namespace tracking {

/**
 * Approximates the sigmoid function parameters by considering the positive output only,
 * the probability and mean output of the negative samples will not be considered. Instead,
 * the parameter B of the sigmoid of the SVM will remain unchanged, only parameter A will
 * be computed.
 */
class FastApproximateSigmoidParameterComputation : public ApproximateSigmoidParameterComputation {
public:

	/**
	 * Constructs a new fast approximate sigmoid parameter computation.
	 *
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	explicit FastApproximateSigmoidParameterComputation(double highProb = 0.99, double lowProb = 0.01);

	~FastApproximateSigmoidParameterComputation();

	std::pair<double, double> computeSigmoidParameters(
			const ChangableDetectorSvm& svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount);
};

} /* namespace tracking */
#endif /* FASTAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_ */
