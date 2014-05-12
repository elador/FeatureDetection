/*
 * WeightedMeanStateExtractor.hpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef WEIGHTEDMEANSTATEEXTRACTOR_HPP_
#define WEIGHTEDMEANSTATEEXTRACTOR_HPP_

#include "condensation/StateExtractor.hpp"

namespace condensation {

/**
 * State extractor that determines the state by computing the weighted mean of the samples.
 */
class WeightedMeanStateExtractor : public StateExtractor {
public:

	/**
	 * Constructs a new weighted mean state extractor.
	 */
	WeightedMeanStateExtractor();

	std::shared_ptr<Sample> extract(const std::vector<std::shared_ptr<Sample>>& samples);
};

} /* namespace condensation */
#endif /* WEIGHTEDMEANSTATEEXTRACTOR_HPP_ */
