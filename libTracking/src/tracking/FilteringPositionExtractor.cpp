/*
 * FilteringPositionExtractor.cpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#include "tracking/FilteringPositionExtractor.h"
#include "tracking/Sample.h"

namespace tracking {

FilteringPositionExtractor::FilteringPositionExtractor(PositionExtractor* extractor) : extractor(extractor) {}

FilteringPositionExtractor::~FilteringPositionExtractor() {
	delete extractor;
}

boost::optional<Sample> FilteringPositionExtractor::extract(const std::vector<Sample>& samples) {
	std::vector<Sample> objects;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->isObject())
			objects.push_back((*sit));
	}
	return extractor->extract(objects);
}

} /* namespace tracking */
