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
			std::shared_ptr<ProbabilisticClassifier> first, std::shared_ptr<TrainableProbabilisticClassifier> second) :
					ProbabilisticTwoStageClassifier(first, second), trainable(second) {}

	~TrainableProbabilisticTwoStageClassifier() {}

	bool classify(const cv::Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::classify(featureVector);
	}

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::getConfidence(featureVector);
	}

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const {
		return ProbabilisticTwoStageClassifier::getProbability(featureVector);
	}

	bool isUsable() const {
		return trainable->isUsable();
	}

	bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples) {
		return trainable->retrain(newPositiveExamples, newNegativeExamples);
	}

	bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples,
			const std::vector<cv::Mat>& newPositiveTestExamples, const std::vector<cv::Mat>& newNegativeTestExamples) {
		return trainable->retrain(newPositiveExamples, newNegativeExamples, newPositiveTestExamples, newNegativeTestExamples);
	}

	void reset() {
		return trainable->reset();
	}

private:

	std::shared_ptr<TrainableProbabilisticClassifier> trainable; ///< The classifier that will be trained.
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICTWOSTAGECLASSIFIER_HPP_ */
