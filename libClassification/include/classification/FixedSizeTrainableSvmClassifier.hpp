/*
 * FixedSizeTrainableSvmClassifier.hpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#ifndef FIXEDSIZETRAINABLESVMCLASSIFIER_HPP_
#define FIXEDSIZETRAINABLESVMCLASSIFIER_HPP_

#include "classification/TrainableSvmClassifier.hpp"

namespace classification {

/**
 * Trainable SVM classifier that has a limited amount of positive and negative training examples.
 */
class FixedSizeTrainableSvmClassifier : public TrainableSvmClassifier {
public:

	/**
	 * Constructs a new fixed size trainable SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] constraintsViolationCosts The costs C of constraints violation.
	 * @param[in] positiveExamples The amount of positive examples that is stored for training.
	 * @param[in] negativeExamples The amount of negative examples that is stored for training.
	 * @param[in] minPositiveExamples The minimum amount of positive training examples needed for training.
	 */
	explicit FixedSizeTrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts = 1,
			unsigned int positiveExamples = 10, unsigned int negativeExamples = 100, unsigned int minPositiveExamples = 1);

	~FixedSizeTrainableSvmClassifier();

protected:

	void clearExamples();

	void addExamples(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	unsigned int getPositiveCount() const;

	unsigned int getNegativeCount() const;

	bool isRetrainingReasonable() const;

	unsigned int fillProblem(struct svm_problem *problem) const;

private:

	/**
	 * Adds new positive training examples. May replace existing examples.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 */
	void addPositiveExamples(const vector<Mat>& newPositiveExamples);

	/**
	 * Adds new negative training examples. May replace existing examples.
	 *
	 * @param[in] newNegativeExamples The new negative training examples.
	 */
	void addNegativeExamples(const vector<Mat>& newNegativeExamples);

	vector<unique_ptr<struct svm_node[]>> positiveExamples; ///< The positive training examples.
	vector<unique_ptr<struct svm_node[]>> negativeExamples; ///< The negative training examples.
	unsigned int negativeInsertPosition; ///< The insertion index of new negative examples.
	unsigned int minPositiveExamples; ///< The minimum amount of positive training examples needed for training.
};

} /* namespace classification */
#endif /* FIXEDSIZETRAINABLESVMCLASSIFIER_HPP_ */
