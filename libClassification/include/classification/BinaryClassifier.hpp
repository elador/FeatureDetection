/*
 * BinaryClassifier.hpp
 *
 *  Created on: 15.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef BINARYCLASSIFIER_H_
#define BINARYCLASSIFIER_H_

#include "opencv2/core/core.hpp"

using cv::Mat;

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
	 * @return A binary flag, true if positively classified.) and a probability for being positive.
	 */
	virtual bool classify(const Mat& featureVector) const = 0;
};

} /* namespace classification */
#endif /* BINARYCLASSIFIER_H_ */
