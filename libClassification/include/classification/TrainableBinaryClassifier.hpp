/*
 * TrainableBinaryClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEBINARYCLASSIFIER_HPP_
#define TRAINABLEBINARYCLASSIFIER_HPP_

#include "classification/TrainableClassifier.hpp"
#include "classification/BinaryClassifier.hpp"

namespace classification {

/**
 * Binary classifier that may be re-trained using new examples. Re-training is an incremental procedure
 * that adds new examples and refines the classifier. Nevertheless, it may be possible that previous
 * training examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableBinaryClassifier : public TrainableClassifier, public BinaryClassifier {
public:

	virtual ~TrainableBinaryClassifier() {}

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return True if the feature vector was positively classified, false otherwise.
	 */
	virtual bool classify(const Mat& featureVector) const = 0;

	/**
	 * Re-trains this classifier incrementally, adding new training examples. May not change the classifier
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) = 0;

	/**
	 * Resets this classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 */
	virtual void reset() = 0;
};

} /* namespace classification */
#endif /* TRAINABLEBINARYCLASSIFIER_HPP_ */
