/*
 * FixedApproximateSigmoidParameterComputation.h
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#ifndef FIXEDAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_
#define FIXEDAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_

#include "classification/ApproximateSigmoidParameterComputation.h"

namespace classification {

/**
 * Approximate sigmoid parameter computation that determines the parameter only once
 * on construction. May be used when the mean SVM outputs of the positive and negative
 * samples rarely change.
 */
class FixedApproximateSigmoidParameterComputation : public ApproximateSigmoidParameterComputation {
public:

	/**
	 * Constructs a new fixed approximate sigmoid parameter computation.
	 *
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 * @param[in] The estimated mean SVM output of the positive samples.
	 * @param[in] The estimated mean SVM output of the negative samples.
	 */
	explicit FixedApproximateSigmoidParameterComputation(double highProb = 0.99, double lowProb = 0.01,
			double meanPosOutput = 1.01, double meanNegOutput = -1.01);

	~FixedApproximateSigmoidParameterComputation();

	std::pair<double, double> computeSigmoidParameters(const struct svm_model *model,
				struct svm_node **positiveSamples, unsigned int positiveCount,
				struct svm_node **negativeSamples, unsigned int negativeCount);

private:

	double paramA; ///< Parameter A of the sigmoid function.
	double paramB; ///< Parameter B of the sigmoid function.
};

} /* namespace classification */
#endif /* FIXEDAPPROXIMATESIGMOIDPARAMETERCOMPUTATION_H_ */
