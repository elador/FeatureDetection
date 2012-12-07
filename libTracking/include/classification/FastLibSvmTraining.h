/*
 * FastLibSvmTraining.h
 *
 *  Created on: 18.09.2012
 *      Author: poschmann
 */

#ifndef FASTLIBSVMTRAINING_H_
#define FASTLIBSVMTRAINING_H_

#include "classification/LibSvmTraining.h"
#include "classification/LibSvmParameterBuilder.h"
#include "classification/RbfLibSvmParameterBuilder.h"
#include "classification/SigmoidParameterComputation.h"
#include "classification/FixedApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <vector>
#include <utility>

using boost::shared_ptr;
using boost::make_shared;
using std::vector;

namespace classification {

/**
 * LibSVM training that limits the amount of training data to ensure a fast training. It will use the support
 * vectors of the previously trained SVM and the new samples for training. If the amount of support vectors
 * is bigger than a pre-defined threshold, some of the support vectors will be removed from the training
 * data of the next learning cycle, so at most a certain amount of support vectors will be used for the next
 * training. The support vectors that will be removed are chosen to be the ones with the biggest distance to
 * the separating hyperplane.
 */
class FastLibSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new fast libSVM training.
	 *
	 * @param[in] minPosCount The minimum amount of positive training examples necessary for training (default ist 10).
	 * @param[in] minNegCount The minimum amount of negative training examples necessary for training (default ist 10).
	 * @param[in] maxCount The maximum amount of support vectors used for training (default is 50).
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FastLibSvmTraining(unsigned int minPosCount = 10, unsigned int minNegCount = 10, unsigned int maxCount = 50,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FixedApproximateSigmoidParameterComputation>());
	~FastLibSvmTraining();

	bool retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& positiveExamples,
			const vector<shared_ptr<FeatureVector> >& negativeExamples);

	void reset(LibSvmClassifier& svm);

private:

	/**
	 * @return True if the training is reasonable, false otherwise.
	 */
	bool isTrainingReasonable() const;

	/**
	 * Adds new training examples.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 */
	void addExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
			const vector<shared_ptr<FeatureVector> >& newNegativeExamples);

	/**
	 * Adds new training examples to the existing ones.
	 *
	 * @param[in] examples The existing training examples.
	 * @param[in] newExamples The new training examples.
	 */
	void addExamples(vector<struct svm_node *>& examples, const vector<shared_ptr<FeatureVector> >& newExamples);

	/**
	 * Trains a libSVM classifier with the positive and negative training examples.
	 *
	 * @param[in] svm The libSVM classifier that should be trained.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(LibSvmClassifier& svm);

	/**
	 * Removes all training examples that are not support vectors.
	 *
	 * @param[in] model The libSVM model.
	 * @return The support vectors that should be deleted when the model is destroyed.
	 */
	vector<struct svm_node *> retainSupportVectors(struct svm_model *model);

	/**
	 * Creates a list of training examples that are support vectors and destroys the other ones.
	 *
	 * @param[in] examples The training examples.
	 * @param[in] model The libSVM model.
	 * @param[in] count The estimated amount of support vectors (used to initialize the list).
	 * @return The training examples that are support vectors.
	 */
	vector<struct svm_node *> extractSupportVectors(
			vector<struct svm_node *>& examples, struct svm_model *model, unsigned int count = 1);

	/**
	 * Determines whether a vector is a support vector.
	 *
	 * @param[in] vector The vector.
	 * @param[in] model The libSVM model.
	 * @return True if the vector is a support vector, false otherwise.
	 */
	bool isSupportVector(struct svm_node *vector, struct svm_model *model);

	/**
	 * Determines whether two vectors are equal.
	 *
	 * @param[in] v1 The first vector.
	 * @param[in] v2 The second vector.
	 * @return True if the vectors are equal, false otherwise.
	 */
	bool areVectorsEqual(struct svm_node *v1, struct svm_node *v2);

	/**
	 * Determines the coefficients (alphas) to the given support vectors.
	 *
	 * @param[in] supportVectors The support vectors.
	 * @param[in] model The libSVM model.
	 * @return The coefficients of the support vectors.
	 */
	vector<double> getCoefficients(vector<struct svm_node *>& supportVectors, struct svm_model *model);

	/**
	 * Computes the distance to the separating hyperplane of each example.
	 *
	 * @param[in] examples The examples.
	 * @param[in] model The libSVM model.
	 * @return The distances.
	 */
	vector<double> computeHyperplaneDistances(vector<struct svm_node *>& examples, struct svm_model *model);

	/**
	 * Determines the smallest element.
	 *
	 * @param[in] values The values.
	 * @return A pair containing the index and value of the min element.
	 */
	std::pair<unsigned int, double> getMin(vector<double> values);

	/**
	 * Determines the biggest element.
	 *
	 * @param[in] values The values.
	 * @return A pair containing the index and value of the max element.
	 */
	std::pair<unsigned int, double> getMax(vector<double> values);

	unsigned int minPosCount; ///< The minimum amount of positive training examples necessary for training.
	unsigned int minNegCount; ///< The minimum amount of negative training examples necessary for training.
	unsigned int maxCount;    ///< The maximum amount of support vectors used for training.
	unsigned int dimensions;  ///< The amount of dimensions of the feature vectors.
	vector<struct svm_node *> positiveExamples; ///< The positive training examples.
	vector<struct svm_node *> negativeExamples; ///< The negative training examples.
};

} /* namespace classification */
#endif /* FASTLIBSVMTRAINING_H_ */
