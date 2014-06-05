/*
 * LowVarianceSampling.hpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef LOWVARIANCESAMPLING_HPP_
#define LOWVARIANCESAMPLING_HPP_

#include "condensation/ResamplingAlgorithm.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_01.hpp"

namespace condensation {

/**
 * Low variance sampling algorithm.
 */
class LowVarianceSampling : public ResamplingAlgorithm {
public:

	/**
	 * Constructs a new low variance sampling algorithm.
	 */
	LowVarianceSampling();

	void resample(const std::vector<std::shared_ptr<Sample>>& samples,
			size_t count, std::vector<std::shared_ptr<Sample>>& newSamples);

private:

	/**
	 * Computes the sum of the sample weights.
	 *
	 * @param[in] samples The samples.
	 * @return The sum of the sample weights.
	 */
	double computeWeightSum(const std::vector<std::shared_ptr<Sample>>& samples);

	boost::mt19937 generator;         ///< Random number generator.
	boost::uniform_01<> distribution; ///< Uniform real distribution.
};

} /* namespace condensation */
#endif /* LOWVARIANCESAMPLING_HPP_ */
