/*
 * ResamplingAlgorithm.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef RESAMPLINGALGORITHM_H_
#define RESAMPLINGALGORITHM_H_

#include <vector>

using std::vector;

namespace condensation {

class Sample;

/**
 * Resampling algorithm.
 */
class ResamplingAlgorithm {
public:

	virtual ~ResamplingAlgorithm() {}

	/**
	 * Resamples the given samples.
	 *
	 * @param[in] samples The vector of samples that should be resampled.
	 * @param[in] count The amount of resulting samples.
	 * @param[in,out] newSamples The vector to insert the new samples into.
	 */
	virtual void resample(const vector<Sample>& samples, unsigned int count, vector<Sample>& newSamples) = 0;
};

} /* namespace condensation */
#endif /* RESAMPLINGALGORITHM_H_ */
