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
	 * @return A binary flag, true if positively classified.) and a probability for being positive.
	 * TODO: We should really only return a bool. I left this for backward compatibility for now.
	 *			- Do we actually need the hyperplane distance from the SVM somewhere? Peter?
	 *			- The ProbabilisticVectorMachine needs it for sure. I think we shouldn't just add a public 
	 *			  function getHyperplaneDist() or something like that because it would brake the interface
	 *			  with other BinaryClassifier's. Maybe make VectorMachineClassifier a friend of
	 *			  ProbabilisticVectorMachineClassifier and let it use its private/protected function? Peter?
	 */
	virtual pair<bool, double> classify(const Mat& featureVector) const = 0;
};

} /* namespace classification */
#endif /* BINARYCLASSIFIER_HPP_ */
