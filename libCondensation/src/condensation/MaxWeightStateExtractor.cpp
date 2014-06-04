/*
 * MaxWeightStateExtractor.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "condensation/MaxWeightStateExtractor.hpp"
#include "condensation/Sample.hpp"

using std::vector;
using std::shared_ptr;

namespace condensation {

MaxWeightStateExtractor::MaxWeightStateExtractor() {}

shared_ptr<Sample> MaxWeightStateExtractor::extract(const vector<shared_ptr<Sample>>& samples) {
	shared_ptr<Sample> best;
	double maxWeight = 0;
	for (shared_ptr<Sample> sample : samples) {
		if (sample->getWeight() > maxWeight) {
			maxWeight = sample->getWeight();
			best = sample;
		}
	}
	if (maxWeight > 0 && best->isTarget())
		return best;
	return shared_ptr<Sample>();
}

} /* namespace condensation */
