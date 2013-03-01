/*
 * BinaryClassifier.hpp
 *
 *  Created on: 15.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef BINARYCLASSIFIER_HPP_
#define BINARYCLASSIFIER_HPP_

#include "opencv2/core/core.hpp"
#include <utility>

using cv::Mat;
using std::pair;

namespace classification {

/**
 * Classifier that determines whether a feature vector is of a certain class or not.
 */
class BinaryClassifier {
public:

	virtual ~BinaryClassifier() {}

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return True if the feature vector was positively classified, false otherwise.
	 */
	virtual bool classify(const Mat& featureVector) const = 0;
};

} /* namespace classification */
#endif /* BINARYCLASSIFIER_HPP_ */
