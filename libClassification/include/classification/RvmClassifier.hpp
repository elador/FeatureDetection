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
#include "boost/property_tree/ptree.hpp"
#include <string>
#include <vector>

using boost::property_tree::ptree;
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
	 * Creates a new RVM classifier from the parameters given in some Matlab file. TODO update doc
	 *
	 * @param[in] classifierFilename The name of the file containing the RVM vectors and weights.
	 * @return The newly created RVM classifier.
	 */
	static shared_ptr<RvmClassifier> loadMatlab(const string& classifierFilename, const string& thresholdsFilename);

	/**
	 * Creates a new RVM classifier from the parameters given in the ptree sub-tree. Loads
	 * the vectors and thresholds from the Matlab file.
	 *
	 * @param[in] subtree The subtree containing the config information for this classifier.
	 * @return The newly created probabilistic RVM classifier.
	 */
	static shared_ptr<RvmClassifier> loadConfig(const ptree& subtree);

private:

	vector<Mat> supportVectors; ///< The support vectors (or here, RSVs).
	//vector<vector<float>> coefficients; ///< The coefficients of the support vectors. Each step in the cascade has its own set of coefficients.
	vector<float> coefficients;
};

} /* namespace classification */
#endif /* RVMCLASSIFIER_HPP_ */
