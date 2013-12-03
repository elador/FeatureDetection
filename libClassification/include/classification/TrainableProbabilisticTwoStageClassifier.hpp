/*
 * TrainableProbabilisticTwoStageClassifier.hpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef TRAINABLEPROBABILISTICTWOSTAGECLASSIFIER_HPP_
#define TRAINABLEPROBABILISTICTWOSTAGECLASSIFIER_HPP_

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
	TrainableProbabilisticTwoStageClassifier(
			shared_ptr<ProbabilisticClassifier> first, shared_ptr<TrainableProbabilisticClassifier> second) :
					ProbabilisticTwoStageClassifier(first, second), trainable(second) {}

	~TrainableProbabilisticTwoStageClassifier() {}

	bool classify(const Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::classify(featureVector);
	}

	pair<bool, double> getConfidence(const Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::getConfidence(featureVector);
	}

	pair<bool, double> getProbability(const Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::getProbability(featureVector);
	}

	bool isUsable() const {
		return trainable->isUsable();
	}

	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
		return trainable->retrain(newPositiveExamples, newNegativeExamples);
	}

	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples,
			const vector<Mat>& newPositiveTestExamples, const vector<Mat>& newNegativeTestExamples) {
		return trainable->retrain(newPositiveExamples, newNegativeExamples, newPositiveTestExamples, newNegativeTestExamples);
	}

	void reset() {
		return trainable->reset();
	}

private:

	shared_ptr<TrainableProbabilisticClassifier> trainable; ///< The classifier that will be trained.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICTWOSTAGECLASSIFIER_HPP_ */
