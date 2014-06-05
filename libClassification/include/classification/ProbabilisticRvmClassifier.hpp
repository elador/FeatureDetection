/*
 * ProbabilisticRvmClassifier.hpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PROBABILISTICRVMCLASSIFIER_HPP_
#define PROBABILISTICRVMCLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include "boost/property_tree/ptree.hpp"
#include <memory>
#include <utility>

namespace classification {

class RvmClassifier;

/**
 * RVM classifier that produces pseudo-probabilistic output. The hyperplane distance of a feature vector will be transformed
 * into a probability using a logistic function p(x) = 1 / (1 + exp(a + b * x)) with x being the hyperplane distance and a and
 * b being parameters.
 */
class ProbabilisticRvmClassifier : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new empty probabilistic RVM classifier.
	 */
	ProbabilisticRvmClassifier();

	/**
	 * Constructs a new probabilistic RVM classifier.
	 *
	 * @param[in] svm The actual RVM.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticRvmClassifier(std::shared_ptr<RvmClassifier> rvm, double logisticA = 0.00556, double logisticB = -2.95);

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const;

	/**
	 * Computes the probability for being positive given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] levelAndDistance The index of the last used filter and distance of that filter level.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	std::pair<bool, double> getProbability(std::pair<int, double> levelAndDistance) const;

	std::pair<bool, double> classifyCached(const cv::Mat& featureVector);

	/**
	 * Changes the logistic parameters of this probabilistic SVM.
	 *
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticParameters(double logisticA, double logisticB);

	/**
	 * Loads the logistic function's parameters from the Matlab file and returns them.
	 *
	 * @param[in] logisticFilename The name of the Matlab-file containing the logistic function's parameters.
	 * @return A pair (a, b) containing the logistic parameters.
	 */
	static std::pair<double, double> loadSigmoidParamsFromMatlab(const std::string& logisticFilename);

	/**
	 * Creates a new probabilistic RVM classifier from the parameters given in the ptree sub-tree. Loads the logistic function's
	 * parameters, then passes the loading to the underlying RVM which loads the vectors and thresholds
	 * from the Matlab file.
	 *
	 * @param[in] subtree The subtree containing the config information for this classifier.
	 * @return The newly created probabilistic RVM classifier.
	 */
	static std::shared_ptr<ProbabilisticRvmClassifier> load(const boost::property_tree::ptree& subtree);

	/**
	 * @return The actual RVM.
	 */
	std::shared_ptr<RvmClassifier> getRvm() {
		return rvm;
	}

	/**
	 * @return The actual RVM.
	 */
	const std::shared_ptr<RvmClassifier> getRvm() const {
		return rvm;
	}

private:

	std::shared_ptr<RvmClassifier> rvm; ///< The actual RVM.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).

};

} /* namespace classification */
#endif /* PROBABILISTICRVMCLASSIFIER_HPP_ */

