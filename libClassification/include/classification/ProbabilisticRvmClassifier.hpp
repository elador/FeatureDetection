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

using boost::property_tree::ptree;
using std::shared_ptr;
using std::string;
using std::pair;

namespace classification {

class RvmClassifier;

/**
 * SVM classifier that produces pseudo-probabilistic output. The hyperplane distance of a feature vector will be transformed
 * into a probability using a logistic function p(x) = 1 / (1 + exp(a + b * x)) with x being the hyperplane distance and a and
 * b being parameters.
 */
class ProbabilisticRvmClassifier : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new empty probabilistic SVM classifier.
	 */
	ProbabilisticRvmClassifier();

	/**
	 * Constructs a new probabilistic SVM classifier.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticRvmClassifier(shared_ptr<RvmClassifier> svm, double logisticA = 0.00556, double logisticB = -2.95);

	~ProbabilisticRvmClassifier();

	/**
	 * @return The actual SVM.
	 */
	shared_ptr<RvmClassifier> getSvm() {
		return svm;
	}

	/**
	 * @return The actual SVM.
	 */
	const shared_ptr<RvmClassifier> getSvm() const {
		return svm;
	}

	pair<bool, double> classify(const Mat& featureVector) const;

	/**
	 * Changes the logistic parameters of this probabilistic SVM.
	 *
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	void setLogisticParameters(double logisticA, double logisticB);

	/**
	 * Creates a new probabilistic RVM classifier from the parameters given in some Matlab file. Loads the logistic function's
	 * parameters from the Matlab file, then passes the loading to the underlying SVM which loads the vectors and thresholds
	 * from the Matlab file.
	 *
	 * @param[in] classifierFilename The name of the file containing the SVM parameters.
	 * @param[in] logisticFilename The name of the file containing the logistic function's parameters.
	 * @return The newly created probabilistic RVM classifier. TODO update doc
	 */
	static pair<double, double> loadSigmoidParamsFromMatlab(const string& logisticFilename);

	/**
	 * Creates a new probabilistic RVM classifier from the parameters given in the ptree sub-tree. Loads the logistic function's
	 * parameters, then passes the loading to the underlying SVM which loads the vectors and thresholds
	 * from the Matlab file.
	 *
	 * @param[in] subtree The subtree containing the config information for this classifier.
	 * @return The newly created probabilistic RVM classifier.
	 */
	static shared_ptr<ProbabilisticRvmClassifier> loadConfig(const ptree& subtree);

private:

	shared_ptr<RvmClassifier> svm; ///< The actual SVM.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
};

} /* namespace classification */
#endif /* PROBABILISTICRVMCLASSIFIER_HPP_ */

