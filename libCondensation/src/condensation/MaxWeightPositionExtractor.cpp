/*
 * MaxWeightPositionExtractor.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "condensation/MaxWeightPositionExtractor.hpp"
#include "condensation/Sample.hpp"

namespace condensation {

MaxWeightPositionExtractor::MaxWeightPositionExtractor() {}

MaxWeightPositionExtractor::~MaxWeightPositionExtractor() {}

shared_ptr<Sample> MaxWeightPositionExtractor::extract(const vector<shared_ptr<Sample>>& samples) {
	shared_ptr<Sample> best;
	double maxWeight = 0;
	for (shared_ptr<Sample> sample : samples) {
		if (sample->getWeight() > maxWeight) {
			maxWeight = sample->getWeight();
			best = sample;
		}
	}
	if (maxWeight > 0 && best->isObject())
		return best;
	return shared_ptr<Sample>();
}

} /* namespace condensation */
