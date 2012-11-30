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

optional<Sample> MaxWeightPositionExtractor::extract(const vector<Sample>& samples) {
	Sample best;
	double maxWeight = 0;
	for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->getWeight() > maxWeight) {
			maxWeight = sit->getWeight();
			best = *sit;
		}
	}
	if (maxWeight > 0 && best.isObject())
		return optional<Sample>(best);
	return optional<Sample>();
}

} /* namespace tracking */
