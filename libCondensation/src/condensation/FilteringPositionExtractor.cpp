/*
 * FilteringPositionExtractor.cpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#include "condensation/FilteringPositionExtractor.h"
#include "condensation/Sample.h"

namespace condensation {

FilteringPositionExtractor::FilteringPositionExtractor(shared_ptr<PositionExtractor> extractor) : extractor(extractor) {}

FilteringPositionExtractor::~FilteringPositionExtractor() {}

optional<Sample> FilteringPositionExtractor::extract(const vector<Sample>& samples) {
	vector<Sample> objects;
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
		if (sample->isObject())
			objects.push_back((*sample));
	}
	return extractor->extract(objects);
}

} /* namespace condensation */
