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

	void load(const std::string classifierFilename, const std::string thresholdsFilename, float limitReliability); // TODO: Re-work this. Should also pass a Kernel.

private:
	int numSV;
	unsigned char** support;	// support[i] hold support vector i
	float* alpha;				// alpha[i] hold the weight of support vector i

	float limitReliability;	// if fout>=limitReliability(threshold_fullsvm), then its a face. (MR default: -1.2)

};

} /* namespace classification */
#endif /* SVMCLASSIFIER_HPP_ */
