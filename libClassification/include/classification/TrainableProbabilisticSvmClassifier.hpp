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

class ProbabilisticSvmClassifier;
class TrainableSvmClassifier;

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
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 */
	TrainableProbabilisticSvmClassifier(shared_ptr<TrainableSvmClassifier> trainableSvm,
			double highProb = 0.99, double lowProb = 0.01);

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
	 * Re-trains this classifier incrementally, adding new training examples. May not change the classifier
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	/**
	 * Resets this classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 */
	void reset();

protected:

	/**
	 * Computes the parameters of the logistic function given the trained SVM classifier.
	 *
	 * @param[in] trainableSvm The trained SVM classifier.
	 * @return A pair containing the parameters a and b of the logistic function p(x) = 1 / (1 + exp(a + b * x)).
	 */
	virtual pair<double, double> computeLogisticParameters(shared_ptr<TrainableSvmClassifier> trainableSvm) const;

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
	shared_ptr<TrainableSvmClassifier> trainableSvm;         ///< The trainable SVM classifier.
	double highProb; ///< The probability of the mean output of positive samples.
	double lowProb;  ///< The probability of the mean output of negative samples.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_ */
