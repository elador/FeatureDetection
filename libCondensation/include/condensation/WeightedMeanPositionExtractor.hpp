/*
 * WeightedMeanPositionExtractor.hpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef WEIGHTEDMEANPOSITIONEXTRACTOR_HPP_
#define WEIGHTEDMEANPOSITIONEXTRACTOR_HPP_

#include "condensation/PositionExtractor.hpp"

namespace condensation {

/**
 * Position extractor that determines the position by computing the weighted mean of the samples.
 */
class WeightedMeanPositionExtractor : public PositionExtractor {
public:

	/**
	 * Constructs a new weighted mean position extractor.
	 */
	WeightedMeanPositionExtractor();

	~WeightedMeanPositionExtractor();

	shared_ptr<Sample> extract(const vector<shared_ptr<Sample>>& samples);
};

} /* namespace condensation */
#endif /* WEIGHTEDMEANPOSITIONEXTRACTOR_HPP_ */
