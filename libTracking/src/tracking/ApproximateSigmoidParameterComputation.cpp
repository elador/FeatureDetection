/*
 * ApproximateSigmoidParameterComputation.cpp
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#include "tracking/ApproximateSigmoidParameterComputation.h"
#include "tracking/ChangableDetectorSvm.h"
#include <cmath>

namespace tracking {

ApproximateSigmoidParameterComputation::ApproximateSigmoidParameterComputation(double highProb, double lowProb) :
		highProb(highProb), lowProb(lowProb) {}

ApproximateSigmoidParameterComputation::~ApproximateSigmoidParameterComputation() {}

std::pair<double, double> ApproximateSigmoidParameterComputation::computeSigmoidParameters(
		const ChangableDetectorSvm& svm, const struct svm_model *model, struct svm_node **positiveSamples,
		unsigned int positiveCount, struct svm_node **negativeSamples, unsigned int negativeCount) {
	double meanPosOutput = computeMeanSvmOutput(model, positiveSamples, positiveCount);
	double meanNegOutput = computeMeanSvmOutput(model, negativeSamples, negativeCount);
	double paramA = (log((1 - lowProb) / lowProb) - log((1 - highProb) / highProb)) / (meanNegOutput - meanPosOutput);
	double paramB = log((1 - highProb) / highProb) - paramA * meanPosOutput;
	return std::make_pair(paramA, paramB);
}

} /* namespace tracking */
