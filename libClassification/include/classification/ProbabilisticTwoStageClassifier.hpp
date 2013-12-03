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

using std::shared_ptr;

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
	explicit ProbabilisticTwoStageClassifier(shared_ptr<ProbabilisticClassifier> first, shared_ptr<ProbabilisticClassifier> second);

	virtual ~ProbabilisticTwoStageClassifier();

	bool classify(const Mat& featureVector) const;

	pair<bool, double> getConfidence(const Mat& featureVector) const;

	pair<bool, double> getProbability(const Mat& featureVector) const;

private:

	shared_ptr<ProbabilisticClassifier> first;  ///< The first classifier.
	shared_ptr<ProbabilisticClassifier> second; ///< The second classifier.
};

} /* namespace classification */
#endif /* PROBABILISTICTWOSTAGECLASSIFIER_HPP_ */
