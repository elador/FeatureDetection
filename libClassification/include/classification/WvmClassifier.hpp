/*
 * WvmClassifier.hpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann & huber
 */
#pragma once

#ifndef WVMCLASSIFIER_HPP_
#define WVMCLASSIFIER_HPP_

#include "classification/VectorMachineClassifier.hpp"
#include "opencv2/core/core.hpp"

using cv::Mat;

namespace classification {

/**
 * Classifier based on a Wavelet Reduced Vector Machine.
 */
class WvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new WVM classifier.
	 *
	 * @param[in] wvm The WVM.
	 */
	explicit WvmClassifier();

	~WvmClassifier();

	pair<bool, double> classify(const Mat& featureVector) const;

};

} /* namespace classification */
#endif /* WVMCLASSIFIER_HPP_ */
