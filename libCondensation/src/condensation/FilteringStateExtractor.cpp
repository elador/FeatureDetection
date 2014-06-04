/*
 * FilteringStateExtractor.cpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/Sample.hpp"

using std::vector;
using std::shared_ptr;

namespace condensation {

FilteringStateExtractor::FilteringStateExtractor(shared_ptr<StateExtractor> extractor) : extractor(extractor) {}

shared_ptr<Sample> FilteringStateExtractor::extract(const vector<shared_ptr<Sample>>& samples) {
	vector<shared_ptr<Sample>> objects;
	for (const shared_ptr<Sample>& sample : samples) {
		if (sample->isTarget())
			objects.push_back(sample);
	}
	return extractor->extract(objects);
}

} /* namespace condensation */
