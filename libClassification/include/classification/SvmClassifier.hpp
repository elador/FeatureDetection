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
#include <string>

using cv::Mat;
using std::string;

namespace classification {

/**
 * Classifier based on a Support Vector Machine.
 */
class SvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new SVM classifier.
	 */
	SvmClassifier();

	~SvmClassifier();

	bool classify(const Mat& featureVector) const;

	/**
	 * Determines the classification result given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return True if feature vectors of the given distance would be classified positively, false otherwise.
	 */
	bool classify(double hyperplaneDistance) const;

	/**
	 * Computes the distance of a feature vector to the decision hyperplane. This is the real distance without
	 * any influence by the offset for configuring the operating point of the SVM.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return The distance of the feature vector to the decision hyperplane.
	 */
	double computeHyperplaneDistance(const Mat& featureVector) const;

	static shared_ptr<SvmClassifier> load(const string& classifierFilename, const string& thresholdsFilename); // TODO: Re-work this. Should also pass a Kernel.

private:
	int numSV;
	unsigned char** support;	///< support[i] hold support vector i
	float* alpha;				///< alpha[i] hold the weight of support vector i

};

} /* namespace classification */
#endif /* SVMCLASSIFIER_HPP_ */
