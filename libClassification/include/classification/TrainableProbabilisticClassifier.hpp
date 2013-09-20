/*
 * TrainableProbabilisticClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEPROBABILISTICCLASSIFIER_HPP_
#define TRAINABLEPROBABILISTICCLASSIFIER_HPP_

#include "classification/TrainableClassifier.hpp"
#include "classification/ProbabilisticClassifier.hpp"

namespace classification {

/**
 * Probabilistic classifier that may be re-trained using new examples. Re-training is an incremental procedure
 * that adds new examples and refines the classifier. Nevertheless, it may be possible that previous
 * training examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableProbabilisticClassifier : public TrainableClassifier, public ProbabilisticClassifier {
public:

	virtual ~TrainableProbabilisticClassifier() {}

	/**
	 * Determines whether this classifier was trained successfully and may be used.
	 *
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool isUsable() const = 0;

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	virtual pair<bool, double> classify(const Mat& featureVector) const = 0;

	/**
	 * Re-trains this classifier incrementally, adding new training examples. The training examples will also
	 * be used as test examples to determine the parameters of the logistic function for probabilistic output.
	 * May not change the classifier if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training and test examples.
	 * @param[in] newNegativeExamples The new negative training and test examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) = 0;

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
	virtual bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples,
			const vector<Mat>& newPositiveTestExamples, const vector<Mat>& newNegativeTestExamples) = 0;

	/**
	 * Resets this classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 */
	virtual void reset() = 0;
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICCLASSIFIER_HPP_ */
