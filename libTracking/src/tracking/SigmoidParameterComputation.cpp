/*
 * SigmoidParameterComputation.cpp
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#include "tracking/SigmoidParameterComputation.h"
#include "svm.h"

namespace tracking {

double SigmoidParameterComputation::computeMeanSvmOutput(const struct svm_model *model,
		struct svm_node **xs, unsigned int count) {
	double outputSum = 0;
	for (unsigned int i = 0; i < count; ++i)
		outputSum += computeSvmOutput(model, xs[i]);
	return outputSum / count;
}

double SigmoidParameterComputation::computeSvmOutput(const struct svm_model *model, const struct svm_node *x) {
	double* dec_values = new double[1];
	svm_predict_values(model, x, dec_values);
	return dec_values[0];
}

} /* namespace tracking */
