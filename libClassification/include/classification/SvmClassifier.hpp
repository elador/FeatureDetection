/*
 * SvmClassifier.hpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann & huber
 */
#pragma once

#ifndef SVMCLASSIFIER_HPP_
#define SVMCLASSIFIER_HPP_

#include "classification/VectorMachineClassifier.hpp"
#include "opencv2/core/core.hpp"

using cv::Mat;

namespace classification {

/**
 * Classifier based on a Support Vector Machine.
 */
class SvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new SVM classifier.
	 *
	 * @param[in] svm The SVM.
	 */
	explicit SvmClassifier();

	~SvmClassifier();

	pair<bool, double> classify(const Mat& featureVector) const;

};

} /* namespace classification */
#endif /* SVMCLASSIFIER_HPP_ */
