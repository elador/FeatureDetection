/*
 * ApproximateSigmoidParameterComputation.h
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#ifndef APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_
#define APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_

#include "classification/SigmoidParameterComputation.h"

namespace classification {

/**
 * Approximates the two parameters of a sigmoid function for probabilistic SVM output.
 * Two probabilities are given that are associated to the mean outputs of positive and
 * negative samples. With these two points, a sigmoid function can be computed.
 */
class ApproximateSigmoidParameterComputation : public SigmoidParameterComputation {
public:

	/**
	 * Constructs a new approximate sigmoid parameter computation.
	 *
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	explicit ApproximateSigmoidParameterComputation(double highProb = 0.99, double lowProb = 0.01);

	virtual ~ApproximateSigmoidParameterComputation();

	virtual std::pair<double, double> computeSigmoidParameters(const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount);

protected:

	double highProb; ///< The probability of the mean output of positive samples.
	double lowProb;  ///< The probability of the mean output of negative samples.
};

} /* namespace classification */
#endif /* APPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_ */
