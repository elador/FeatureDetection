/*
 * TrainableOneClassSvmClassifier.hpp
 *
 *  Created on: 11.09.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEONECLASSSVMCLASSIFIER_HPP_
#define TRAINABLEONECLASSSVMCLASSIFIER_HPP_

#include "classification/TrainableBinaryClassifier.hpp"
#include "classification/LibSvmUtils.hpp"
#include <memory>

using std::shared_ptr;
using std::unique_ptr;

namespace classification {

class SvmClassifier;
class Kernel;

/**
 * One-class SVM based on libSVM that has a limited amount of training examples.
 */
class TrainableOneClassSvmClassifier : public TrainableBinaryClassifier {
public:

	/**
	 * Constructs a new trainable one-class SVM classifier that wraps around the actual SVM classifier.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] nu The parameter nu for tuning over-fitting vs. generalization.
	 * @param[in] minExamples The minimum amount of training examples needed for training.
	 * @param[in] maxExamples The maximum amount of stored training examples.
	 */
	TrainableOneClassSvmClassifier(shared_ptr<SvmClassifier> svm, double nu, int minExamples, int maxExamples);

	/**
	 * Constructs a new trainable one-class SVM classifer.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] nu The parameter nu for tuning over-fitting vs. generalization.
	 * @param[in] minExamples The minimum amount of training examples needed for training.
	 * @param[in] maxExamples The maximum amount of stored training examples.
	 */
	TrainableOneClassSvmClassifier(const shared_ptr<Kernel> kernel, double nu, int minExamples, int maxExamples);

	~TrainableOneClassSvmClassifier();

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
	 * Determines the classification result given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return True if feature vectors of the given distance would be classified positively, false otherwise.
	 */
	bool classify(double hyperplaneDistance) const;

	/**
	 * Computes the distance of a feature vector to the decision hyperplane. This is the real distance without
	 * any influence by the offset for configuring the operating point of the SVM.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return The distance of the feature vector to the decision hyperplane.
	 */
	double computeHyperplaneDistance(const Mat& featureVector) const;

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

protected:

	/**
	 * Adds new training examples. May replace existing examples.
	 *
	 * @param[in] newExamples The new training examples.
	 */
	void addExamples(const vector<Mat>& newExamples);

	/**
	 * Determines whether re-training is reasonable.
	 *
	 * @return True if re-training is reasonable, false otherwise.
	 */
	bool isRetrainingReasonable() const;

	/**
	 * Adds the training examples to the libSVM problem.
	 *
	 * @param[in] problem The libSVM problem.
	 */
	void fillProblem(struct svm_problem *problem) const;

private:

	/**
	 * Creates the libSVM parameters.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] nu The parameter nu for tuning over-fitting vs. generalization.
	 */
	void createParameters(shared_ptr<Kernel> kernel, double nu);

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

	LibSvmUtils utils; ///< Utils for using libSVM.
	shared_ptr<SvmClassifier> svm; ///< The actual SVM.
	bool usable; ///< Flag that indicates whether this classifier is usable.
	unique_ptr<struct svm_parameter, ParameterDeleter> param; ///< The libSVM parameters.
	unique_ptr<struct svm_problem, ProblemDeleter> problem;   ///< The libSVM problem.
	unique_ptr<struct svm_model, ModelDeleter> model;         ///< The libSVM model.
	vector<unique_ptr<struct svm_node[], NodeDeleter>> examples; ///< The training examples.
	unsigned int minExamples; ///< The minimum amount of training examples needed for training.
};

} /* namespace classification */
#endif /* TRAINABLEONECLASSSVMCLASSIFIER_HPP_ */
