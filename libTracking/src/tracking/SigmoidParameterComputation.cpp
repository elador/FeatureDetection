/*
 * SigmoidParameterComputation.cpp
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#include "tracking/SigmoidParameterComputation.h"
#include "svm.h"
#include <cmath>

namespace tracking {

double SigmoidParameterComputation::computeMeanSvmOutput(const struct svm_model *model,
		struct svm_node **xs, unsigned int count) {
	double outputSum = 0;
	for (unsigned int i = 0; i < count; ++i)
		outputSum += computeSvmOutput(model, xs[i]);
	return outputSum / count;
}

double SigmoidParameterComputation::computeSvmOutput(const struct svm_model *model, const struct svm_node *x) {
	double output = -model->rho[0];
	for (int i = 0; i < model->l; ++i) {
		const struct svm_node *xit = x;
		const struct svm_node *svit = model->SV[i];
		double squaredEuclideanDistance = 0;
		while (xit->index != -1 || svit->index != -1) {
			if (svit->index == xit->index) {
				double diff = xit->value - svit->value;
				squaredEuclideanDistance += diff * diff;
				++xit;
				++svit;
			} else if (svit->index < xit->index || xit->index == -1) { // input vector value of that dimension is 0
				squaredEuclideanDistance += svit->value * svit->value;
				++svit;
			} else { // xit->index < svit->index || svit->index == -1 // support vector value of that dimension is 0
				squaredEuclideanDistance += xit->value * xit->value;
				++xit;
			}
		}
		output += model->sv_coef[0][i] * exp(-model->param.gamma * squaredEuclideanDistance);
	}
	return output;
}

} /* namespace tracking */
