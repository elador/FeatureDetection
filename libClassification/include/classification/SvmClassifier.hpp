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
#include <vector>

namespace classification {

/**
 * Classifier based on a Support Vector Machine.
 */
class SvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 */
	explicit SvmClassifier(std::shared_ptr<Kernel> kernel);

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	/**
	 * Determines the classification result given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return True if feature vectors of the given distance would be classified positively, false otherwise.
	 */
	bool classify(double hyperplaneDistance) const;

	/**
	 * Computes the classification confidence given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return A pair containing the binary classification result and the confidence of the classification.
	 */
	std::pair<bool, double> getConfidence(double hyperplaneDistance) const;

	/**
	 * Computes the distance of a feature vector to the decision hyperplane. This is the real distance without
	 * any influence by the offset for configuring the operating point of the SVM.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return The distance of the feature vector to the decision hyperplane.
	 */
	double computeHyperplaneDistance(const cv::Mat& featureVector) const;

	/**
	 * Changes the parameters of this SVM.
	 *
	 * @param[in] supportVectors The support vectors.
	 * @param[in] coefficients The coefficients of the support vectors.
	 * @param[in] bias The bias.
	 */
	void setSvmParameters(std::vector<cv::Mat> supportVectors, std::vector<float> coefficients, double bias);

	/**
	 * @return The support vectors.
	 */
	const std::vector<cv::Mat>& getSupportVectors() const {
		return supportVectors;
	}

	/**
	 * @return The coefficients of the support vectors.
	 */
	const std::vector<float>& getCoefficients() const {
		return coefficients;
	}

private:

	std::vector<cv::Mat> supportVectors; ///< The support vectors.
	std::vector<float> coefficients; ///< The coefficients of the support vectors.
};

} /* namespace classification */
#endif /* SVMCLASSIFIER_HPP_ */
