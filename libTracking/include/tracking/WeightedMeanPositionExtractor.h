/*
 * WeightedMeanPositionExtractor.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef WEIGHTEDMEANPOSITIONEXTRACTOR_H_
#define WEIGHTEDMEANPOSITIONEXTRACTOR_H_

#include "tracking/PositionExtractor.h"

namespace tracking {

/**
 * Position extractor that determines the position by computing the weighted mean of the samples.
 */
class WeightedMeanPositionExtractor : public PositionExtractor {
public:

	/**
	 * Constructs a new weighted mean position extractor.
	 */
	explicit WeightedMeanPositionExtractor();

	~WeightedMeanPositionExtractor();

	boost::optional<Sample> extract(const std::vector<Sample>& samples);
};

} /* namespace tracking */
#endif /* WEIGHTEDMEANPOSITIONEXTRACTOR_H_ */
