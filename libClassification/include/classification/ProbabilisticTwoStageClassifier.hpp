/*
 * ProbabilisticTwoStageClassifier.hpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef PROBABILISTICTWOSTAGECLASSIFIER_HPP_
#define PROBABILISTICTWOSTAGECLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include <memory>

namespace classification {

/**
 * Probabilistic classifier that consists of two stages. The first stage acts as a guard for the second
 * one, so feature vectors have to get past the first classifier to get classified by the second one. If
 * they do not get past the first one, they will be regarded as negative.
 */
class ProbabilisticTwoStageClassifier : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new probabilistic two-stage classifier.
	 *
	 * @param[in] first The first classifier.
	 * @param[in] second The second classifier.
	 */
	explicit ProbabilisticTwoStageClassifier(
			std::shared_ptr<ProbabilisticClassifier> first, std::shared_ptr<ProbabilisticClassifier> second);

	virtual ~ProbabilisticTwoStageClassifier();

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const;

private:

	std::shared_ptr<ProbabilisticClassifier> first;  ///< The first classifier.
	std::shared_ptr<ProbabilisticClassifier> second; ///< The second classifier.
};

} /* namespace classification */
#endif /* PROBABILISTICTWOSTAGECLASSIFIER_HPP_ */
