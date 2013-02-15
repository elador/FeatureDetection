/*
 * FilteringPositionExtractor.cpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#include "tracking/FilteringPositionExtractor.h"
#include "tracking/Sample.h"

namespace tracking {

FilteringPositionExtractor::FilteringPositionExtractor(shared_ptr<PositionExtractor> extractor) : extractor(extractor) {}

FilteringPositionExtractor::~FilteringPositionExtractor() {}

optional<Sample> FilteringPositionExtractor::extract(const vector<Sample>& samples) {
	vector<Sample> objects;
	for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->isObject())
			objects.push_back((*sit));
	}
	return extractor->extract(objects);
}

} /* namespace tracking */
