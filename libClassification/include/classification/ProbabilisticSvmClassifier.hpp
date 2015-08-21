/*
 * ProbabilisticSvmClassifier.hpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PROBABILISTICSVMCLASSIFIER_HPP_
#define PROBABILISTICSVMCLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include "boost/property_tree/ptree.hpp"
#include <memory>

namespace classification {

class SvmClassifier;
class Kernel;

/**
 * SVM classifier that produces pseudo-probabilistic output. The hyperplane distance of a feature vector will be transformed
 * into a probability using a logistic function p(x) = 1 / (1 + exp(a + b * x)) with x being the hyperplane distance and a and
 * b being parameters.
 */
class ProbabilisticSvmClassifier : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new probabilistic SVM classifier that creates the underlying SVM using the given kernel.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticSvmClassifier(std::shared_ptr<Kernel> kernel, double logisticA = 0.00556, double logisticB = -2.95);

	/**
	 * Constructs a new probabilistic SVM classifier that is based on an already constructed SVM.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticSvmClassifier(std::shared_ptr<SvmClassifier> svm, double logisticA = 0.00556, double logisticB = -2.95);

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	std::pair<bool, double> getProbability(const cv::Mat& featureVector) const;

	/**
	 * Computes the probability for being positive given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @return A pair containing the binary classification result and a probability between zero and one for being positive.
	 */
	std::pair<bool, double> getProbability(double hyperplaneDistance) const;

	/**
	 * Changes the logistic parameters of this probabilistic SVM.
	 *
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticParameters(double logisticA, double logisticB);

	/**
	 * Creates a new probabilistic WVM classifier from the parameters given in some Matlab file. Loads the logistic function's
	 * parameters from the matlab file, then passes the loading to the underlying WVM which loads the vectors and thresholds
	 * from the matlab file. TODO update doc
	 *
	 * @param[in] classifierFilename The name of the file containing the WVM parameters.
	 * @param[in] thresholdsFilename The name of the file containing the thresholds of the filter levels and the logistic function's parameters.
	 * @return The newly created probabilistic WVM classifier.
	 */
	static std::pair<double, double> loadSigmoidParamsFromMatlab(const std::string& thresholdsFilename);

	/**
	 * Creates a new probabilistic SVM classifier from the parameters given in some Matlab file. Loads the logistic function's
	 * parameters from the matlab file, then passes the loading to the underlying SVM which loads the vectors and thresholds
	 * from the matlab file.
	 *
	 * @param[in] classifierFilename The name of the file containing the SVM parameters.
	 * @param[in] logisticFilename The name of the file containing the logistic function's parameters.
	 * @return The newly created probabilistic SVM classifier. TODO: This could be renamed just to "load(...)". But NOTE: The classifier will then be loaded with
	 * default settings, and any deviation from that (e.g. adjusting the thresholds) must be done manually.
	 */
	static std::shared_ptr<ProbabilisticSvmClassifier> loadFromMatlab(const std::string& classifierFilename, const std::string& logisticFilename);

	/**
	 * Creates a new probabilistic SVM classifier from the parameters given in the ptree sub-tree. Loads the logistic function's
	 * parameters, then passes the loading to the underlying SVM which loads the vectors and thresholds
	 * from the matlab file.
	 *
	 * @param[in] subtree The subtree containing the config information for this classifier.
	 * @return The newly created probabilistic WVM classifier.
	 */
	static std::shared_ptr<ProbabilisticSvmClassifier> load(const boost::property_tree::ptree& subtree);

	/**
	 * Creates a new probabilistic SVM from parameters (kernel, bias, coefficients, support vectors, logistic)
	 * given in a text file.
	 *
	 * @param[in] file The file input stream to load the parameters from.
	 * @return The newly created probabilistic SVM classifier.
	 */
	static std::shared_ptr<ProbabilisticSvmClassifier> load(std::ifstream& file);

	/**
	 * Stores the logistic and SVM parameters (kernel, bias, coefficients, support vectors) into a text file.
	 *
	 * @param[in] file The file output stream to store the parameters into.
	 */
	void store(std::ofstream& file);

	/**
	 * @return The actual SVM.
	 */
	std::shared_ptr<SvmClassifier> getSvm() {
		return svm;
	}

	/**
	 * @return The actual SVM.
	 */
	const std::shared_ptr<SvmClassifier> getSvm() const {
		return svm;
	}

private:

	std::shared_ptr<SvmClassifier> svm; ///< The actual SVM.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
};

} /* namespace classification */
#endif /* PROBABILISTICSVMCLASSIFIER_HPP_ */

