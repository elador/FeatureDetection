/*
 * ProbabilisticClassifier.hpp
 *
 *  Created on: 15.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef PROBABILISTICCLASSIFIER_HPP_
#define PROBABILISTICCLASSIFIER_HPP_

#include "opencv2/core/core.hpp"
#include <utility>

using cv::Mat;
using std::pair;

namespace classification {

/**
 * Classifier that determines the probability of a feature vector being of a certain class.
 */
class ProbabilisticClassifier {
public:

	virtual ~ProbabilisticClassifier() {}

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	virtual pair<bool, double> classify(const Mat& featureVector) const = 0;
};

} /* namespace classification */
#endif /* PROBABILISTICCLASSIFIER_HPP_ */
