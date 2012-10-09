/*
 * SigmoidParameterComputation.h
 *
 *  Created on: 14.08.2012
 *      Author: poschmann
 */

#ifndef SIGMOIDPARAMETERCOMPUTATION_H_
#define SIGMOIDPARAMETERCOMPUTATION_H_

#include <utility>

struct svm_node;
struct svm_model;

namespace tracking {

class ChangableDetectorSvm;

/**
 * Computation of the two parameters of the sigmoid function for probabilistic SVM output. The
 * equation of the sigmoid function f is f(x) = 1 / (1 + exp(A * x + B)), where A and B are the
 * parameters that will be computed by objects of this type.
 */
class SigmoidParameterComputation {
public:

	virtual ~SigmoidParameterComputation() {}

	/**
	 * Computes the two parameters of the sigmoid function for probabilistic SVM output.
	 * The equation of the sigmoid function f is f(x) = 1 / (1 + exp(A * x + B)) and A
	 * and B are the parameters.
	 *
	 * @param[in] svm The SVM that should be trained.
	 * @param model The libSVM model of a dual-class RBF kernel SVM.
	 * @param positiveSamples The positive samples used for the training of the SVM.
	 * @param positiveCount The amount of positive samples.
	 * @param negativeSamples The negative samples used for the training of the SVM.
	 * @param negativeCount The amount of negative samples.
	 * @return A pair containing the parameters A and B.
	 */
	virtual std::pair<double, double> computeSigmoidParameters(
			const ChangableDetectorSvm& svm, const struct svm_model *model,
			struct svm_node **positiveSamples, unsigned int positiveCount,
			struct svm_node **negativeSamples, unsigned int negativeCount) = 0;

protected:

	/**
	 * Computes the mean output of a SVM given some vectors.
	 *
	 * @param[in] model The libSVM model of a dual-class RBF kernel SVM.
	 * @param[in] xs The input vectors.
	 * @param[in] count The amount of vectors.
	 * @return The mean SVM output value.
	 */
	double computeMeanSvmOutput(const struct svm_model *model, struct svm_node **xs, unsigned int count);

	/**
	 * Computes the output of a SVM given an input vector.
	 *
	 * @param[in] model The libSVM model of a dual-class RBF kernel SVM.
	 * @param[in] x The input vector.
	 * @return The SVM output value.
	 */
	double computeSvmOutput(const struct svm_model *model, const struct svm_node *x);
};

} /* namespace tracking */
#endif /* SIGMOIDPARAMETERCOMPUTATION_H_ */
