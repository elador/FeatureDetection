/*
 * MaxWeightPositionExtractor.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "condensation/MaxWeightPositionExtractor.h"
#include "condensation/Sample.h"

namespace condensation {

MaxWeightPositionExtractor::MaxWeightPositionExtractor() {}

MaxWeightPositionExtractor::~MaxWeightPositionExtractor() {}

optional<Sample> MaxWeightPositionExtractor::extract(const vector<Sample>& samples) {
	Sample best;
	double maxWeight = 0;
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
		if (sample->getWeight() > maxWeight) {
			maxWeight = sample->getWeight();
			best = *sample;
		}
	}
	if (maxWeight > 0 && best.isObject())
		return optional<Sample>(best);
	return optional<Sample>();
}

} /* namespace condensation */
