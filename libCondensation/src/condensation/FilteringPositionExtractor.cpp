/*
 * FilteringPositionExtractor.cpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#include "condensation/FilteringPositionExtractor.hpp"
#include "condensation/Sample.hpp"

namespace condensation {

FilteringPositionExtractor::FilteringPositionExtractor(shared_ptr<PositionExtractor> extractor) : extractor(extractor) {}

FilteringPositionExtractor::~FilteringPositionExtractor() {}

shared_ptr<Sample> FilteringPositionExtractor::extract(const vector<shared_ptr<Sample>>& samples) {
	vector<shared_ptr<Sample>> objects;
	for (const shared_ptr<Sample>& sample : samples) {
		if (sample->isObject())
			objects.push_back(sample);
	}
	return extractor->extract(objects);
}

} /* namespace condensation */
