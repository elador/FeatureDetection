/*
 * FrameBasedTrainableSvmClassifier.hpp
 *
 *  Created on: 06.03.2013
 *      Author: poschmann
 */

#ifndef FRAMEBASEDTRAINABLESVMCLASSIFIER_HPP_
#define FRAMEBASEDTRAINABLESVMCLASSIFIER_HPP_

#include "classification/TrainableSvmClassifier.hpp"

namespace classification {

/**
 * Trainable SVM classifier that uses the latest examples for training.
 */
class FrameBasedTrainableSvmClassifier : public TrainableSvmClassifier {
public:

	/**
	 * Constructs a new frame based trainable SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] constraintsViolationCosts The costs C of constraints violation.
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive training examples per frame for the training to be reasonable.
	 */
	FrameBasedTrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts = 1,
			int frameLength = 5, float minAvgSamples = 1);

	~FrameBasedTrainableSvmClassifier();

	/**
	 * @return The required amount of positive training examples for the training to be reasonable.
	 */
	unsigned int getRequiredPositiveCount() const;

protected:

	void clearExamples();

	void addExamples(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	unsigned int getPositiveCount() const;

	unsigned int getNegativeCount() const;

	bool isRetrainingReasonable() const;

	unsigned int fillProblem(struct svm_problem *problem) const;

private:

	/**
	 * Replaces training examples by new ones.
	 *
	 * @param[in] examples The vector of training examples whose content should be replaced with new ones.
	 * @param[in] newExamples The new training examples.
	 */
	void replaceExamples(vector<unique_ptr<struct svm_node[], NodeDeleter>>& examples, const vector<Mat>& newExamples);

	int frameLength;     ///< The length of the memory in frames.
	float minAvgSamples; ///< The minimum average positive training examples per frame for the training to be reasonable.
	vector<vector<unique_ptr<struct svm_node[], NodeDeleter>>> positiveExamples; ///< The positive training examples of the last frames.
	vector<vector<unique_ptr<struct svm_node[], NodeDeleter>>> negativeExamples; ///< The negative training examples of the last frames.
	int oldestEntry;                                     ///< The index of the oldest example entry.
};

} /* namespace classification */
#endif /* FRAMEBASEDTRAINABLESVMCLASSIFIER_HPP_ */
