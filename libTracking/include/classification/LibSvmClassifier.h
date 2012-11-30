/*
 * LibSvmClassifier.h
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#ifndef LIBSVMCLASSIFIER_H_
#define LIBSVMCLASSIFIER_H_

#include "classification/Classifier.h"
#include "svm.h"
#include <vector>

using std::pair;
using std::vector;

namespace classification {

class EagerFeatureVector;

/**
 * Classifier that is based on a support vector machine implemented in libSVM.
 */
class LibSvmClassifier : public Classifier {
public:

	/**
	 * Constructs a new libSVM based classifier.
	 */
	explicit LibSvmClassifier();

	virtual ~LibSvmClassifier();

	pair<bool, double> classify(const FeatureVector& featureVector) const;

	/**
	 * Changes the libSVM model and the parameters of the probabilistic output function.
	 * This classifier will be responsible for the deletion of the model.
	 *
	 * @param[in] dimensions The amount of dimensions of the feature space.
	 * @param[in] model The new libSVM model.
	 * @param[in] probParamA Parameter A of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
	 * @param[in] probParamB Parameter B of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
	 */
	void setModel(int dimensions, svm_model *model, double probParamA, double probParamB);

private:

	/**
	 * Frees the memory of the model.
	 */
	void deleteModel();

	/**
	 * Computes the kernel value.
	 *
	 * @param[in] x The input vector.
	 * @param[in] sv The support vector.
	 * @param[in] param The SVM parameters.
	 * @return The kernel value.
	 */
	double kernel(const FeatureVector& x, const FeatureVector& sv, const svm_parameter& param) const;

	svm_model *model; ///< The libSVM model.
	vector<EagerFeatureVector> supportVectors; ///< The support vectors.
	double probParamA; ///< Parameter A of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
	double probParamB; ///< Parameter B of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
};

} /* namespace tracking */
#endif /* LIBSVMCLASSIFIER_H_ */
