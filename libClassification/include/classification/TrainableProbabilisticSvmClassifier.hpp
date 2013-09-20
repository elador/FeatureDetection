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

using std::shared_ptr;

namespace classification {

class SvmClassifier;
class ProbabilisticSvmClassifier;
class TrainableClassifier;
class TrainableSvmClassifier;
class TrainableOneClassSvmClassifier;

/**
 * Probabilistic SVM classifier that can be re-trained. Computes the new parameters of a logistic function by
 * using the mean output of the positive and negative training examples and associated probabilities.
 */
class TrainableProbabilisticSvmClassifier : public TrainableProbabilisticClassifier {
public:

	/**
	 * Constructs a new trainable probabilistic SVM classifier based on an ordinary SVM.
	 *
	 * @param[in] trainableSvm The trainable SVM classifier.
	 * @param[in] positiveCount The maximum amount of stored positive test examples.
	 * @param[in] negativeCount The maximum amount of stored negative test examples.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	TrainableProbabilisticSvmClassifier(shared_ptr<TrainableSvmClassifier> trainableSvm,
			int positiveCount, int negativeCount, double highProb = 0.99, double lowProb = 0.01);

	/**
	 * Constructs a new trainable probabilistic SVM classifier based on a one-class SVM.
	 *
	 * @param[in] trainableSvm The trainable one-class SVM classifier.
	 * @param[in] positiveCount The maximum amount of stored positive test examples.
	 * @param[in] negativeCount The maximum amount of stored negative test examples.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	TrainableProbabilisticSvmClassifier(shared_ptr<TrainableOneClassSvmClassifier> trainableSvm,
			int positiveCount, int negativeCount, double highProb = 0.99, double lowProb = 0.01);

	/**
	 * Constructs a new trainable probabilistic SVM classifier.
	 *
	 * @param[in] trainableSvm The trainable SVM classifier.
	 * @param[in] probabilisticSvm The actual probabilistic SVM classifier.
	 * @param[in] positiveCount The maximum amount of stored positive test examples.
	 * @param[in] negativeCount The maximum amount of stored negative test examples.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	TrainableProbabilisticSvmClassifier(
			shared_ptr<TrainableClassifier> trainableSvm, shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm,
			int positiveCount, int negativeCount, double highProb = 0.99, double lowProb = 0.01);

	virtual ~TrainableProbabilisticSvmClassifier();

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	shared_ptr<ProbabilisticSvmClassifier> getProbabilisticSvm() {
		return probabilisticSvm;
	}

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	const shared_ptr<ProbabilisticSvmClassifier> getProbabilisticSvm() const {
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

	/**
	 * Determines whether this classifier was trained successfully and may be used.
	 *
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	bool isUsable() const;

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	pair<bool, double> classify(const Mat& featureVector) const;

	/**
	 * Re-trains this classifier incrementally, adding new training examples. The training examples will also
	 * be used as test examples to determine the parameters of the logistic function for probabilistic output.
	 * May not change the classifier if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training and test examples.
	 * @param[in] newNegativeExamples The new negative training and test examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	/**
	 * Re-trains this classifier incrementally, adding new training examples. The test examples will be used to
	 * determine the parameters of the logistic function for probabilistic output. May not change the classifier
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @param[in] newPositiveTestExamples The new positive test examples.
	 * @param[in] newNegativeTestExamples The new negative test examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples,
			const vector<Mat>& newPositiveTestExamples, const vector<Mat>& newNegativeTestExamples);

	/**
	 * Resets this classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 */
	void reset();

protected:

	/**
	 * Adds new test examples to a given vector. May replace existing examples.
	 *
	 * @param[in,out] examples The existing test examples.
	 * @param[in] newExamples The new test examples.
	 * @param[in,out] insertPosition The insertion index of new examples.
	 */
	void addTestExamples(vector<Mat>& examples, const vector<Mat>& newExamples, unsigned int& insertPosition);

	/**
	 * Computes the parameters of the logistic function given the trained SVM classifier.
	 *
	 * @param[in] svm The trained SVM classifier.
	 * @return A pair containing the parameters a and b of the logistic function p(x) = 1 / (1 + exp(a + b * x)).
	 */
	virtual pair<double, double> computeLogisticParameters(shared_ptr<SvmClassifier> svm) const;

	/**
	 * Computes the mean SVM hyperplane distance of the test examples.
	 *
	 * @param[in] svm The SVM classifier.
	 * @param[in] examples The test examples.
	 * @return The mean hyperplane distance of the test examples.
	 */
	double computeMeanOutput(shared_ptr<SvmClassifier> svm, const vector<Mat>& examples) const;

	/**
	 * Computes the parameters of the logistic function given the mean SVM outputs of the positive and negative training data.
	 *
	 * @param[in] meanPosOutput The mean SVM output of the positive data.
	 * @param[in] meanNegOutput The mean SVM output of the negative data.
	 * @return A pair containing the parameters a and b of the logistic function p(x) = 1 / (1 + exp(a + b * x)).
	 */
	pair<double, double> computeLogisticParameters(double meanPosOutput, double meanNegOutput) const;

private:

	shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm; ///< The actual probabilistic SVM classifier.
	shared_ptr<TrainableClassifier> trainableSvm;            ///< The trainable SVM classifier.
	vector<Mat> positiveTestExamples; ///< The positive test examples.
	vector<Mat> negativeTestExamples; ///< The negative test examples.
	unsigned int positiveInsertPosition; ///< The insertion index of new positive examples.
	unsigned int negativeInsertPosition; ///< The insertion index of new negative examples.
	double highProb; ///< The probability of the mean output of positive samples.
	double lowProb;  ///< The probability of the mean output of negative samples.
	bool adjustThreshold;     ///< Flag that indicates whether the SVM threshold should be adjusted to a certain probability.
	double targetProbability; ///< Target probability of the SVM threshold in case of adjustment.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_ */
