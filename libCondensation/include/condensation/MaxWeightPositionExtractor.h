/*
 * MaxWeightPositionExtractor.h
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef MAXWEIGHTPOSITIONEXTRACTOR_H_
#define MAXWEIGHTPOSITIONEXTRACTOR_H_

#include "condensation/PositionExtractor.h"

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

	optional<Sample> extract(const vector<Sample>& samples);
};

} /* namespace condensation */
#endif /* MAXWEIGHTPOSITIONEXTRACTOR_H_ */
