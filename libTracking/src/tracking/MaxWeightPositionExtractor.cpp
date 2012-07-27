/*
 * MaxWeightPositionExtractor.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "tracking/MaxWeightPositionExtractor.h"
#include "tracking/Sample.h"

namespace tracking {

MaxWeightPositionExtractor::MaxWeightPositionExtractor() {}

MaxWeightPositionExtractor::~MaxWeightPositionExtractor() {}

boost::optional<Sample> MaxWeightPositionExtractor::extract(const std::vector<Sample>& samples) {
	Sample best;
	double maxWeight = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->getWeight() > maxWeight) {
			maxWeight = sit->getWeight();
			best = *sit;
		}
	}
	if (maxWeight > 0 && best.isObject())
		return boost::optional<Sample>(best);
	return boost::optional<Sample>();
}

} /* namespace tracking */
