/*
 * ResamplingAlgorithm.hpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef RESAMPLINGALGORITHM_HPP_
#define RESAMPLINGALGORITHM_HPP_

#include <vector>
#include <memory>

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
	virtual void resample(const std::vector<std::shared_ptr<Sample>>& samples,
			size_t count, std::vector<std::shared_ptr<Sample>>& newSamples) = 0;
};

} /* namespace condensation */
#endif /* RESAMPLINGALGORITHM_HPP_ */
