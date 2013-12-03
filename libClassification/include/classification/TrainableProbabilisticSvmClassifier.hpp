/*
 * TrainableProbabilisticSvmClassifier.hpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_
#define TRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_

#include "classification/TrainableProbabilisticClassifier.hpp"
#include <memory>

namespace classification {

class SvmClassifier;
class TrainableSvmClassifier;
class ProbabilisticSvmClassifier;

/**
 * Probabilistic SVM classifier that can be re-trained. Computes the new parameters of a logistic function by
 * using the mean output of the positive and negative training examples and associated probabilities.
 */
class TrainableProbabilisticSvmClassifier : public TrainableProbabilisticClassifier {
public:

	/**
	 * Constructs a new trainable probabilistic SVM classifier.
	 *
	 * @param[in] trainableSvm The trainable SVM classifier.
	 * @param[in] positiveCount The maximum amount of stored positive test examples.
	 * @param[in] negativeCount The maximum amount of stored negative test examples.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	TrainableProbabilisticSvmClassifier(std::shared_ptr<TrainableSvmClassifier> trainableSvm,
			int positiveCount, int negativeCount, double highProb = 0.99, double lowProb = 0.01);

	virtual ~TrainableProbabilisticSvmClassifier();

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const;

	bool isUsable() const;

	bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples);

	bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples,
			const std::vector<cv::Mat>& newPositiveTestExamples, const std::vector<cv::Mat>& newNegativeTestExamples);

	void reset();

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	std::shared_ptr<ProbabilisticSvmClassifier> getProbabilisticSvm() {
		return probabilisticSvm;
	}

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	const std::shared_ptr<ProbabilisticSvmClassifier> getProbabilisticSvm() const {
		return probabilisticSvm;
	}

	/**
	 * Enables threshold adjustment of the SVM, so at each re-training the threshold will be set to a value that classifies
	 * all vectors as positive whose probability is above the given target probability.
	 *
	 * @param[in] targetProbability Target probability of the SVM threshold.
	 */
	void setAdjustThreshold(double targetProbability) {
		this->adjustThreshold = true;
		this->targetProbability = targetProbability;
	}

protected:

	/**
	 * Adds new test examples to a given vector. May replace existing examples.
	 *
	 * @param[in,out] examples The existing test examples.
	 * @param[in] newExamples The new test examples.
	 * @param[in,out] insertPosition The insertion index of new examples.
	 */
	void addTestExamples(std::vector<cv::Mat>& examples, const std::vector<cv::Mat>& newExamples, size_t& insertPosition);

	/**
	 * Computes the parameters of the logistic function given the trained SVM classifier.
	 *
	 * @param[in] svm The trained SVM classifier.
	 * @return A pair containing the parameters a and b of the logistic function p(x) = 1 / (1 + exp(a + b * x)).
	 */
	virtual std::pair<double, double> computeLogisticParameters(std::shared_ptr<SvmClassifier> svm) const;

	/**
	 * Computes the mean SVM hyperplane distance of the test examples.
	 *
	 * @param[in] svm The SVM classifier.
	 * @param[in] examples The test examples.
	 * @return The mean hyperplane distance of the test examples.
	 */
	double computeMeanOutput(std::shared_ptr<SvmClassifier> svm, const std::vector<cv::Mat>& examples) const;

	/**
	 * Computes the parameters of the logistic function given the mean SVM outputs of the positive and negative training data.
	 *
	 * @param[in] meanPosOutput The mean SVM output of the positive data.
	 * @param[in] meanNegOutput The mean SVM output of the negative data.
	 * @return A pair containing the parameters a and b of the logistic function p(x) = 1 / (1 + exp(a + b * x)).
	 */
	std::pair<double, double> computeLogisticParameters(double meanPosOutput, double meanNegOutput) const;

private:

	std::shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm; ///< The actual probabilistic SVM classifier.
	std::shared_ptr<TrainableSvmClassifier> trainableSvm;         ///< The trainable SVM classifier.
	std::vector<cv::Mat> positiveTestExamples; ///< The positive test examples.
	std::vector<cv::Mat> negativeTestExamples; ///< The negative test examples.
	size_t positiveInsertPosition; ///< The insertion index of new positive examples.
	size_t negativeInsertPosition; ///< The insertion index of new negative examples.
	double highProb; ///< The probability of the mean output of positive samples.
	double lowProb;  ///< The probability of the mean output of negative samples.
	bool adjustThreshold;     ///< Flag that indicates whether the SVM threshold should be adjusted to a certain probability.
	double targetProbability; ///< Target probability of the SVM threshold in case of adjustment.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_ */
