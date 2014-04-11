/*
 * MaxWeightPositionExtractor.hpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef MAXWEIGHTPOSITIONEXTRACTOR_HPP_
#define MAXWEIGHTPOSITIONEXTRACTOR_HPP_

#include "condensation/PositionExtractor.hpp"

namespace condensation {

/**
 * Position extractor that uses the position of the sample with the highest weight.
 */
class MaxWeightPositionExtractor : public PositionExtractor {
public:

	/**
	 * Constructs a new max weight position extractor.
	 */
	MaxWeightPositionExtractor();

	~MaxWeightPositionExtractor();

	shared_ptr<Sample> extract(const vector<shared_ptr<Sample>>& samples);
};

} /* namespace condensation */
#endif /* MAXWEIGHTPOSITIONEXTRACTOR_HPP_ */
