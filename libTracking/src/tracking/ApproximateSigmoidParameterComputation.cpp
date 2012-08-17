/*
 * ApproximateSigmoidParameterComputation.cpp
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#include "tracking/ApproximateSigmoidParameterComputation.h"
#include "tracking/ChangableDetectorSvm.h"
#include "svm.h"
#include <cmath>

namespace tracking {

ApproximateSigmoidParameterComputation::ApproximateSigmoidParameterComputation() {}

ApproximateSigmoidParameterComputation::~ApproximateSigmoidParameterComputation() {}

std::pair<double, double> ApproximateSigmoidParameterComputation::computeSigmoidParameters(
		const ChangableDetectorSvm& svm, const struct svm_model *model, struct svm_node **positiveSamples,
		unsigned int positiveCount, struct svm_node **negativeSamples, unsigned int negativeCount) {
	double meanPositiveOutput = computeMeanSvmOutput(model, positiveSamples, positiveCount);
	double paramA = -meanPositiveOutput;
	double paramB = svm.getProbParamB();
	double max = 0.99;
	while (1 / (1 + exp(paramA * meanPositiveOutput + paramB)) < max)
		paramA -= meanPositiveOutput;
	return std::make_pair(paramA, paramB);
}

} /* namespace tracking */
