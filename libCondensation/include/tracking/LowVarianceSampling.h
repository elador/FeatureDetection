/*
 * LowVarianceSampling.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef LOWVARIANCESAMPLING_H_
#define LOWVARIANCESAMPLING_H_

#include "tracking/ResamplingAlgorithm.h"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_01.hpp"

namespace tracking {

/**
 * Low variance sampling algorithm.
 */
class LowVarianceSampling : public ResamplingAlgorithm {
public:

	/**
	 * Constructs a new low variance sampling algorithm.
	 */
	LowVarianceSampling();

	~LowVarianceSampling();

	void resample(const vector<Sample>& samples, unsigned int count, vector<Sample>& newSamples);

private:

	/**
	 * Computes the sum of the sample weights.
	 *
	 * @param[in] samples The samples.
	 * @return The sum of the sample weights.
	 */
	double computeWeightSum(const vector<Sample>& samples);

	boost::mt19937 generator;         ///< Random number generator.
	boost::uniform_01<> distribution; ///< Uniform real distribution.
};

} /* namespace tracking */
#endif /* LOWVARIANCESAMPLING_H_ */
