/*
 * LibSvmClassifier.h
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#ifndef LIBSVMCLASSIFIER_H_
#define LIBSVMCLASSIFIER_H_

#include "classification/TrainableClassifier.h"
#include "classification/Training.h"
#include "svm.h"
#include <memory>
#include <vector>

using std::shared_ptr;
using std::vector;

namespace classification {

/**
 * Classifier that is based on a support vector machine implemented in libSVM.
 */
class LibSvmClassifier : public TrainableClassifier {
public:

	/**
	 * Constructs a new libSVM based classifier.
	 *
	 * @param[in] training The training algorithm for this classifier.
	 */
	explicit LibSvmClassifier(shared_ptr<Training<LibSvmClassifier> > training);

	virtual ~LibSvmClassifier();

	pair<bool, double> classify(const FeatureVector& featureVector) const;

	bool retrain(const vector<shared_ptr<FeatureVector> >& positiveExamples,
			const vector<shared_ptr<FeatureVector> >& negativeExamples);

	void reset();

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

	/**
	 * @return The libSVM model.
	 */
	svm_model *getModel() {
		return model;
	}

private:

	/**
	 * Frees the memory of the model.
	 */
	void deleteModel();

	/**
	 * Computes the kernel value of two vectors.
	 *
	 * @param[in] x A vector.
	 * @param[in] y Another vector.
	 * @param[in] param The SVM parameters.
	 * @return The kernel value.
	 */
	double kernel(const FeatureVector& x, const FeatureVector& y, const svm_parameter& param) const;

	shared_ptr<Training<LibSvmClassifier> > training; ///< The training algorithm for this classifier.
	svm_model *model;                                 ///< The libSVM model.
	vector<FeatureVector> supportVectors;             ///< The support vectors.
	double probParamA; ///< Parameter A of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
	double probParamB; ///< Parameter B of the probabilistic output equation p(x) = 1 / (1 + exp(A * x + B)).
};

} /* namespace classification */
#endif /* LIBSVMCLASSIFIER_H_ */
