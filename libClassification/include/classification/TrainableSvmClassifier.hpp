/*
 * TrainableSvmClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLESVMCLASSIFIER_HPP_
#define TRAINABLESVMCLASSIFIER_HPP_

#include "classification/TrainableBinaryClassifier.hpp"
#include "classification/LibSvmUtils.hpp"
#include <memory>
#include <unordered_map>

using std::shared_ptr;
using std::unique_ptr;
using std::string;
using std::unordered_map;

namespace classification {

class SvmClassifier;
class Kernel;

/**
 * SVM classifier that can be re-trained. Uses libSVM for training.
 */
class TrainableSvmClassifier : public TrainableBinaryClassifier {
public:

	/**
	 * Constructs a new trainable SVM classifier that wraps around the actual SVM classifier.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] constraintsViolationCosts The costs C of constraints violation.
	 */
	explicit TrainableSvmClassifier(shared_ptr<SvmClassifier> svm, double constraintsViolationCosts = 1);

	/**
	 * Constructs a new trainable SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] constraintsViolationCosts The costs C of constraints violation.
	 */
	explicit TrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts = 1);

	virtual ~TrainableSvmClassifier();

	/**
	 * @return The actual SVM.
	 */
	shared_ptr<SvmClassifier> getSvm() {
		return svm;
	}

	/**
	 * @return The actual SVM.
	 */
	const shared_ptr<SvmClassifier> getSvm() const {
		return svm;
	}

	/**
	 * Determines whether this classifier was trained successfully and may be used.
	 *
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	bool isUsable() const {
		return usable;
	}

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return True if the feature vector was positively classified, false otherwise.
	 */
	bool classify(const Mat& featureVector) const;

	/**
	 * Re-trains this SVM incrementally, adding new training examples. May not change the SVM
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @return True if this SVM was trained successfully and may be used, false otherwise.
	 */
	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	/**
	 * Resets this SVM. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 */
	void reset();

	/**
	 * Loads static negative training examples from a file.
	 *
	 * @param[in] negativesFilename The name of the file containing the static negative training examples.
	 * @param[in] maxNegatives The amount of static negative training examples to use.
	 * @param[in] scale The factor for scaling the data after loading.
	 */
	void loadStaticNegatives(const string& negativesFilename, int maxNegatives, double scale = 1);

	/**
	 * Computes the mean SVM outputs of the positive and negative training examples.
	 *
	 * @return A pair containing the positive and negative mean SVM outputs.
	 */
	pair<double, double> computeMeanSvmOutputs();

protected:

	/**
	 * Computes the SVM output given a libSVM node.
	 *
	 * @param[in] x The libSVM node.
	 * @return The SVM output value.
	 */
	double computeSvmOutput(const struct svm_node *x) const;

	/**
	 * Removes all training examples.
	 */
	virtual void clearExamples() = 0;

	/**
	 * Adds new training examples. Might remove training examples in case of a budget.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 */
	virtual void addExamples(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) = 0;

	/**
	 * @return The amount of positive examples used for training.
	 */
	virtual unsigned int getPositiveCount() const = 0;

	/**
	 * @return The amount of positive examples used for training.
	 */
	virtual unsigned int getNegativeCount() const = 0;

	/**
	 * Determines whether re-training is reasonable.
	 *
	 * @return True if re-training is reasonable, false otherwise.
	 */
	virtual bool isRetrainingReasonable() const = 0;

	/**
	 * Adds the training examples to the libSVM problem.
	 *
	 * @param[in] problem The libSVM problem.
	 * @return The index after the last inserted training example.
	 */
	virtual unsigned int fillProblem(struct svm_problem *problem) const = 0;

private:

	/**
	 * Creates the libSVM parameters.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] constraintsViolationCosts The costs C of constraints violation.
	 */
	void createParameters(shared_ptr<Kernel> kernel, double constraintsViolationCosts);

	/**
	 * Changes the parameters of this SVM by training using libSVM.
	 */
	void train();

	/**
	 * Creates the libSVM problem containing the training data.
	 *
	 * @return The libSVM problem.
	 */
	unique_ptr<struct svm_problem, ProblemDeleter> createProblem();

	/**
	 * Updates the parameters of the actual SVM using the trained a libSVM model.
	 */
	void updateSvmParameters();

protected:

	LibSvmUtils utils; ///< Utils for using libSVM.
	shared_ptr<SvmClassifier> svm; ///< The actual SVM.

private:

	double constraintsViolationCosts; ///< The costs C of constraints violation.
	bool usable; ///< Flag that indicates whether this classifier is usable.
	vector<unique_ptr<struct svm_node[], NodeDeleter>> staticNegativeExamples; ///< The static negative training examples.
	unique_ptr<struct svm_parameter, ParameterDeleter> param; ///< The libSVM parameters.
	unique_ptr<struct svm_problem, ProblemDeleter> problem;   ///< The libSVM problem.
	unique_ptr<struct svm_model, ModelDeleter> model;         ///< The libSVM model.
};

} /* namespace classification */
#endif /* TRAINABLESVMCLASSIFIER_HPP_ */
