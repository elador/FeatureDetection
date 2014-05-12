/*
 * MaxWeightStateExtractor.hpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef MAXWEIGHTSTATEEXTRACTOR_HPP_
#define MAXWEIGHTSTATEEXTRACTOR_HPP_

#include "condensation/StateExtractor.hpp"

namespace condensation {

/**
 * State extractor that uses the state of the sample with the highest weight.
 */
class MaxWeightStateExtractor : public StateExtractor {
public:

	/**
	 * Constructs a new max weight state extractor.
	 */
	MaxWeightStateExtractor();

	std::shared_ptr<Sample> extract(const std::vector<std::shared_ptr<Sample>>& samples);
};

} /* namespace condensation */
#endif /* MAXWEIGHTSTATEEXTRACTOR_HPP_ */
