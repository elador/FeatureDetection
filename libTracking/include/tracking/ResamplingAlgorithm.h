/*
 * ResamplingAlgorithm.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef RESAMPLINGALGORITHM_H_
#define RESAMPLINGALGORITHM_H_

#include <vector>

namespace tracking {

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
	virtual void resample(const std::vector<Sample>& samples, unsigned int count, std::vector<Sample>& newSamples) = 0;
};

} /* namespace tracking */
#endif /* RESAMPLINGALGORITHM_H_ */
