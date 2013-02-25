/*
 * ProbabilisticWvmClassifier.hpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef PROBABILISTICWVMCLASSIFIER_HPP_
#define PROBABILISTICWVMCLASSIFIER_HPP_

#include "classification/ProbabilisticVectorMachineClassifier.hpp"

namespace classification {

class VectorMachineClassifier;

/**
 * A classifier that uses a VectorMachineClassifier together with a sigmoid function to produce pseudo-probabilistic output.
 * The probability is calculated as 1.0f / (1.0f + exp(posterior_svm[0]*hyperplaneDist + posterior_svm[1])).
 * TODO: It's a shame that we have to implement this separately for the SVM and the WVM. But the load() methods differ, one has to load 'posterior_svm' and the other one 'posterior_wrvm'.
 */
class ProbabilisticWvmClassifier : public ProbabilisticVectorMachineClassifier
{
public:
	//ProbabilisticWvmClassifier(shared_ptr<WvmClassifier> classifier); 	// TODO See also ProbabilisticVectorMachineClassifier.hpp. How do we solve this?
	~ProbabilisticWvmClassifier(void);

	/**
	 * Loads the sigmoid parameters from the matlab file, then passes the loading to the underlying classifier which loads the vectors and thresholds from the matlab file.
	 *
	 * @param[in] classifierFilename TODO.
	 * @param[in] thresholdsFilename TODO.
	 */
	void load(const string classifierFilename, const string thresholdsFilename); // TODO: Re-work this. Should also pass a Kernel.

};

} /* namespace classification */
#endif /* PROBABILISTICWVMCLASSIFIER_HPP_ */

