/*
 * ProbabilisticVectorMachineClassifier.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PROBABILISTICVECTORMACHINECLASSIFIER_HPP_
#define PROBABILISTICVECTORMACHINECLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include <memory>
#include <string>

using std::shared_ptr;
using std::string;

namespace classification {

class VectorMachineClassifier;

/**
 * A classifier that uses a VectorMachineClassifier together with a sigmoid function to produce pseudo-probabilistic output.
 * The probability is calculated as 1.0f / (1.0f + exp(posterior_svm[0]*hyperplaneDist + posterior_svm[1])).
 */
class ProbabilisticVectorMachineClassifier : public ProbabilisticClassifier
{
public:
	//ProbabilisticVectorMachineClassifier(shared_ptr<VectorMachineClassifier> classifier);	// TODO Hmm what do we do with this...?
	//~ProbabilisticVectorMachineClassifier(void);

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector as a rectangular image-patch.
	 * @return A probability between 0 and 1 for being in the positive class.
	 */
	//pair<bool, double> classify(const Mat& featureVector) const;

	/**
	 * Loads the sigmoid parameters from the matlab file, then passes the loading to the underlying classifier which loads the vectors and thresholds from the matlab file.
	 *
	 * @param[in] classifierFilename TODO.
	 * @param[in] thresholdsFilename TODO.
	 */
	//void load(const string classifierFilename, const string thresholdsFilename); // TODO: Re-work this. Should also pass a Kernel.

protected:
	//shared_ptr<VectorMachineClassifier> classifier;
	//float sigmoidParameters[2];	// probabilistic svm output: p(ffp|t) = 1 / (1 + exp(p[0]*t +p[1]))

};

} /* namespace classification */
#endif /* PROBABILISTICVECTORMACHINECLASSIFIER_HPP_ */

