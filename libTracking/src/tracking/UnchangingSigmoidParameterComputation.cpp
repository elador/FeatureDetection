/*
 * UnchangingSigmoidParameterComputation.cpp
 *
 *  Created on: 16.08.2012
 *      Author: poschmann
 */

#include "tracking/UnchangingSigmoidParameterComputation.h"
#include "tracking/ChangableDetectorSvm.h"

namespace tracking {

UnchangingSigmoidParameterComputation::UnchangingSigmoidParameterComputation() {}

UnchangingSigmoidParameterComputation::~UnchangingSigmoidParameterComputation() {}

std::pair<double, double> UnchangingSigmoidParameterComputation::computeSigmoidParameters(
		const ChangableDetectorSvm& svm, const struct svm_model *model, struct svm_node **positiveSamples,
		unsigned int positiveCount, struct svm_node **negativeSamples, unsigned int negativeCount) {
	return std::make_pair(svm.getProbParamA(), svm.getProbParamB());
}

} /* namespace tracking */
