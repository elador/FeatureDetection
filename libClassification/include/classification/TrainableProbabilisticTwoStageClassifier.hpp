/*
 * TrainableProbabilisticTwoStageClassifier.h
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef PROBABILISTICTRAINABLETWOSTAGECLASSIFIER_HPP_
#define PROBABILISTICTRAINABLETWOSTAGECLASSIFIER_HPP_

#include "classification/TrainableProbabilisticClassifier.hpp"
#include "classification/ProbabilisticTwoStageClassifier.hpp"

namespace classification {

/**
 * Probabilistic two-stage classifier whose second classifier can be re-trained using new examples.
 */
class TrainableProbabilisticTwoStageClassifier : public ProbabilisticTwoStageClassifier, public TrainableProbabilisticClassifier {
public:

	/**
	 * Constructs a new trainable probabilistic two-stage classifier.
	 *
	 * @param[in] first The first classifier.
	 * @param[in] second The second classifier (that will be re-trained).
	 */
	explicit TrainableProbabilisticTwoStageClassifier(
			shared_ptr<ProbabilisticClassifier> first, shared_ptr<TrainableProbabilisticClassifier> second) :
					ProbabilisticTwoStageClassifier(first, second), trainable(second) {}

	~TrainableProbabilisticTwoStageClassifier() {}

	pair<bool, double> classify(const Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::classify(featureVector);
	}

	bool retrain(const vector<Mat>& positiveExamples, const vector<Mat>& negativeExamples) {
		return trainable->retrain(positiveExamples, negativeExamples);
	}

	void reset() {
		return trainable->reset();
	}

private:

	shared_ptr<TrainableProbabilisticClassifier> trainable; ///< The classifier that will be trained.
};

} /* namespace classification */
#endif /* PROBABILISTICTRAINABLETWOSTAGECLASSIFIER_HPP_ */
