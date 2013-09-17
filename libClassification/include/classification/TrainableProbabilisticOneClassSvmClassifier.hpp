/*
 * TrainableProbabilisticOneClassSvmClassifier.hpp
 *
 *  Created on: 13.09.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEPROBABILISTICONECLASSSVMCLASSIFIER_HPP_
#define TRAINABLEPROBABILISTICONECLASSSVMCLASSIFIER_HPP_

#include "classification/TrainableProbabilisticClassifier.hpp"
#include <memory>

using std::shared_ptr;

namespace classification {

class ProbabilisticSvmClassifier;
class TrainableOneClassSvmClassifier;

/**
 * Probabilistic one-class SVM classifier that can be re-trained. Computes the new parameters of a logistic function
 * on construction assuming fixed mean positive and negative SVM outputs.
 */
class TrainableProbabilisticOneClassSvmClassifier : public TrainableProbabilisticClassifier {
public:

	/**
	 * Constructs a new trainable probabilistic one-class SVM classifier.
	 *
	 * @param[in] trainableSvm The trainable one-class SVM classifier.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 * @param[in] The estimated mean SVM output of the positive samples.
	 * @param[in] The estimated mean SVM output of the negative samples.
	 */
	explicit TrainableProbabilisticOneClassSvmClassifier(shared_ptr<TrainableOneClassSvmClassifier> trainableSvm,
			double highProb = 0.95, double lowProb = 0.05, double meanPosOutput = 1.01, double meanNegOutput = -1.01);

	~TrainableProbabilisticOneClassSvmClassifier();

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

private:

	shared_ptr<ProbabilisticSvmClassifier> probabilisticSvm; ///< The actual probabilistic SVM classifier.
	shared_ptr<TrainableOneClassSvmClassifier> trainableSvm; ///< The trainable SVM classifier.
	double highProb; ///< The probability of the mean output of positive samples.
	double lowProb;  ///< The probability of the mean output of negative samples.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICONECLASSSVMCLASSIFIER_HPP_ */
