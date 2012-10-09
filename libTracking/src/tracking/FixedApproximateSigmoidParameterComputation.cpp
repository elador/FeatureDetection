/*
 * FixedApproximateSigmoidParameterComputation.cpp
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#include "tracking/FixedApproximateSigmoidParameterComputation.h"
#include <cmath>

namespace tracking {

FixedApproximateSigmoidParameterComputation::FixedApproximateSigmoidParameterComputation(
		double highProb, double lowProb, double meanPosOutput, double meanNegOutput) :
		ApproximateSigmoidParameterComputation(highProb, lowProb) {
	paramA = (log((1 - lowProb) / lowProb) - log((1 - highProb) / highProb)) / (meanNegOutput - meanPosOutput);
	paramB = log((1 - highProb) / highProb) - paramA * meanPosOutput;
}

FixedApproximateSigmoidParameterComputation::~FixedApproximateSigmoidParameterComputation() {}

std::pair<double, double> FixedApproximateSigmoidParameterComputation::computeSigmoidParameters(
			const ChangableDetectorSvm& svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount) {
	return std::make_pair(paramA, paramB);
}

} /* namespace tracking */
