/*
 * LibSvmTraining.h
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#ifndef LIBSVMTRAINING_H_
#define LIBSVMTRAINING_H_

#include "classification/Training.h"
#include "classification/LibSvmClassifier.h"
#include "classification/LibSvmParameterBuilder.h"
#include "classification/RbfLibSvmParameterBuilder.h"
#include "classification/SigmoidParameterComputation.h"
#include "classification/FixedApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include <memory>
#include <vector>

struct svm_parameter;

using std::shared_ptr;
using std::make_shared;

namespace classification {

/**
 * Algorithm for training LibSvmClassifier.
 */
class LibSvmTraining : public Training<LibSvmClassifier> {
public:

	/**
	 * Constructs a new libSVM training.
	 *
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit LibSvmTraining(shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation = make_shared<FixedApproximateSigmoidParameterComputation>());

	virtual ~LibSvmTraining();

	virtual bool retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
			const vector<shared_ptr<FeatureVector> >& newNegativeExamples) = 0;

	virtual void reset(LibSvmClassifier& svm) = 0;

	/**
	 * Reads the static negative training examples from a file.
	 *
	 * @param[in] negativesFilename The name of the file containing the static negative training examples.
	 * @param[in] maxNegatives The amount of static negative training examples to use.
	 */
	void readStaticNegatives(const std::string negativesFilename, int maxNegatives);

protected:

	/**
	 * @return The amount of static negative training examples.
	 */
	unsigned int getStaticNegativeCount() const {
		return staticNegativeExamples.size();
	}

	/**
	 * Deletes the training examples.
	 *
	 * @param[in] examples The training examples that should be deleted.
	 */
	void freeExamples(vector<struct svm_node *>& examples);

	/**
	 * Creates a new libSVM node from the given feature vector data.
	 *
	 * @param[in] vector The feature vector.
	 * @return The newly created libSVM node.
	 */
	struct svm_node *createNode(const FeatureVector& vector);

	/**
	 * Computes the output of a SVM given an input vector.
	 *
	 * @param[in] model The libSVM model of a dual-class SVM.
	 * @param[in] x The input vector.
	 * @return The SVM output value.
	 */
	double computeSvmOutput(const struct svm_model *model, const struct svm_node *x);

	/**
	 * Trains a libSVM classifier.
	 *
	 * @param[in] svm The libSVM classifier that should be trained.
	 * @param[in] dimensions The dimension of the feature space.
	 * @param[in] positiveExamples The positive training examples.
	 * @param[in] negativeExamples The negative training examples.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(LibSvmClassifier& svm, unsigned int dimensions,
			vector<struct svm_node *>& positiveExamples, vector<struct svm_node *>& negativeExamples);

	/**
	 * Creates the libSVM parameters. In order to free their memory, svm_destroy_param has to be called.
	 *
	 * @param[in] positiveCount The amount of positive training examples.
	 * @param[in] negativeCount The amount of negative training examples.
	 * @return The libSVM parameters.
	 */
	struct svm_parameter *createParameters(unsigned int positiveCount, unsigned int negativeCount);

	/**
	 * Creates the libSVM problem.
	 *
	 * @param[in] positiveExamples The positive training examples.
	 * @param[in] negativeExamples The negative training examples.
	 * @param[in] staticNegativeExamples The static negative training examples.
	 * @return The libSVM problem.
	 */
	struct svm_problem *createProblem(vector<struct svm_node *>& positiveExamples,
			vector<struct svm_node *>& negativeExamples, vector<struct svm_node *>& staticNegativeExamples);

	/**
	 * Changes the parameters of an SVM given a libSVM model.
	 *
	 * @param[in] svm The SVM whose parameters should be changed.
	 * @param[in] dimensions The amount of dimensions of the feature space.
	 * @param[in] model The libSVM model.
	 * @param[in] problem The libSVM problem.
	 * @param[in] positiveCount The amount of positive examples used for the training of the SVM.
	 * @param[in] negativeCount The amount of negative examples used for the training of the SVM.
	 */
	void changeSvmParameters(LibSvmClassifier& svm, unsigned int dimensions, struct svm_model *model,
			struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount);

private:

	vector<struct svm_node *> staticNegativeExamples; ///< The static negative training examples.
	shared_ptr<LibSvmParameterBuilder> parameterBuilder; ///< The libSVM parameter builder.
	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation; ///< The computation of the sigmoid parameters.
};

} /* namespace classification */
#endif /* LIBSVMTRAINING_H_ */
