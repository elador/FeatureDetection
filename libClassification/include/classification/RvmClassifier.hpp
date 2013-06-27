/*
 * RvmClassifier.hpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RVMCLASSIFIER_HPP_
#define RVMCLASSIFIER_HPP_

#include "classification/VectorMachineClassifier.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <vector>

using cv::Mat;
using std::string;
using std::vector;

namespace classification {

/**
 * Classifier based on a Support Vector Machine.
 */
class RvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 */
	explicit RvmClassifier(shared_ptr<Kernel> kernel);

	~RvmClassifier();

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

	/**
	 * Changes the parameters of this SVM.
	 *
	 * @param[in] supportVectors The support vectors.
	 * @param[in] coefficients The coefficients of the support vectors.
	 * @param[in] bias The bias.
	 */
	void setSvmParameters(vector<Mat> supportVectors, vector<float> coefficients, double bias);

	/**
	 * Creates a new SVM classifier from the parameters given in some Matlab file.
	 *
	 * @param[in] classifierFilename The name of the file containing the SVM parameters.
	 * @return The newly created SVM classifier.
	 */
	//TODO static shared_ptr<RvmClassifier> loadMatlab(const string& classifierFilename);

	/**
	 * Creates a new SVM classifier from the parameters given in some text file.
	 *
	 * @param[in] classifierFilename The name of the file containing the SVM parameters.
	 * @return The newly created SVM classifier.
	 */
	static shared_ptr<RvmClassifier> loadText(const string& classifierFilename);

private:

	vector<Mat> supportVectors; ///< The support vectors.
	vector<float> coefficients; ///< The coefficients of the support vectors.
};

} /* namespace classification */
#endif /* RVMCLASSIFIER_HPP_ */
