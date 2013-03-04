/*
 * ProbabilisticWvmClassifier.hpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PROBABILISTICWVMCLASSIFIER_HPP_
#define PROBABILISTICWVMCLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include <memory>

using std::shared_ptr;
using std::string;

namespace classification {

class WvmClassifier;

/**
 * WVM classifier that produces pseudo-probabilistic output. The hyperplane distance of a feature vector will be transformed
 * into a probability using a logistic function p(x) = 1 / (1 + exp(a + b * x)) with x being the hyperplane distance and a and
 * b being parameters.
 * TODO: different computation when feature vector did not reach the final stage -> mention it in this documentation (and change computation)
 * TODO: It's a shame that we have to implement this separately for the SVM and the WVM. But the load() methods differ, one has to load 'posterior_svm' and the other one 'posterior_wrvm'.
 */
class ProbabilisticWvmClassifier : public ProbabilisticClassifier {
public:

	/**
	 * Constructs a new probabilistic WVM classifier.
	 *
	 * @param[in] wvm The actual WVM.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	explicit ProbabilisticWvmClassifier(shared_ptr<WvmClassifier> wvm, double logisticA = 0.00556, double logisticB = -2.95);

	~ProbabilisticWvmClassifier();

	pair<bool, double> classify(const Mat& featureVector) const;

	/**
	 * Creates a new probabilistic WVM classifier from the parameters given in some Matlab file. Loads the logistic function's
	 * parameters from the matlab file, then passes the loading to the underlying WVM which loads the vectors and thresholds
	 * from the matlab file.
	 *
	 * @param[in] classifierFilename TODO.
	 * @param[in] thresholdsFilename TODO.
	 */
	static shared_ptr<ProbabilisticWvmClassifier> loadMatlab(const string& classifierFilename, const string& thresholdsFilename); // TODO: Re-work this. Should also pass a Kernel.

private:

	shared_ptr<WvmClassifier> wvm; ///< The actual SVM.
	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
};

} /* namespace classification */
#endif /* PROBABILISTICWVMCLASSIFIER_HPP_ */

