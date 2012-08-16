/*
 * ExactSigmoidParameterComputation.cpp
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#include "tracking/ExactSigmoidParameterComputation.h"
#include "tracking/ChangableDetectorSvm.h"
#include "svm.h"
#include <cmath>

namespace tracking {

ExactSigmoidParameterComputation::ExactSigmoidParameterComputation() {}

ExactSigmoidParameterComputation::~ExactSigmoidParameterComputation() {}

std::pair<double, double> ExactSigmoidParameterComputation::computeSigmoidParameters(ChangableDetectorSvm* svm,
		const struct svm_model *model, struct svm_node **positiveSamples, unsigned int positiveCount,
		struct svm_node **negativeSamples, unsigned int negativeCount) {
	double meanPositiveOutput = computeMeanSvmOutput(model, positiveSamples, positiveCount);
	double meanNegativeOutput = computeMeanSvmOutput(model, negativeSamples, negativeCount);
	double max = 0.99;
	double min = 0.01;
	double paramA = (log((1 - min) / min) - log((1 - max) / max)) / (meanNegativeOutput - meanPositiveOutput);
	double paramB = log((1 - max) / max) - paramA * meanPositiveOutput;
	return std::make_pair(paramA, paramB);
}

} /* namespace tracking */
