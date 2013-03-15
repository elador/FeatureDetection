/*
 * WeightedMeanPositionExtractor.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef WEIGHTEDMEANPOSITIONEXTRACTOR_H_
#define WEIGHTEDMEANPOSITIONEXTRACTOR_H_

#include "condensation/PositionExtractor.h"

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

	optional<Sample> extract(const vector<Sample>& samples);
};

} /* namespace condensation */
#endif /* WEIGHTEDMEANPOSITIONEXTRACTOR_H_ */
