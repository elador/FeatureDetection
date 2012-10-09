/*
 * FastApproximateSigmoidParameterComputation.cpp
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#include "tracking/FastApproximateSigmoidParameterComputation.h"
#include "tracking/ChangableDetectorSvm.h"
#include <cmath>

namespace tracking {

FastApproximateSigmoidParameterComputation::FastApproximateSigmoidParameterComputation(double highProb, double lowProb):
		ApproximateSigmoidParameterComputation(highProb, lowProb) {}

FastApproximateSigmoidParameterComputation::~FastApproximateSigmoidParameterComputation() {}

std::pair<double, double> FastApproximateSigmoidParameterComputation::computeSigmoidParameters(
		const ChangableDetectorSvm& svm, const struct svm_model *model, struct svm_node **positiveSamples,
		unsigned int positiveCount, struct svm_node **negativeSamples, unsigned int negativeCount) {
	double meanPositiveOutput = computeMeanSvmOutput(model, positiveSamples, positiveCount);
	double paramA = -meanPositiveOutput;
	double paramB = svm.getProbParamB();
	while (1 / (1 + exp(paramA * meanPositiveOutput + paramB)) < highProb)
		paramA -= meanPositiveOutput;
	return std::make_pair(paramA, paramB);
}

} /* namespace tracking */
