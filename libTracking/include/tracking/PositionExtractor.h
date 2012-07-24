/*
 * PositionExtractor.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef POSITIONEXTRACTOR_H_
#define POSITIONEXTRACTOR_H_

#include "boost/optional.hpp"
#include <vector>

namespace tracking {

class Sample;

/**
 * Extractor of a bounding box around the estimated object position.
 */
class PositionExtractor {
public:

	virtual ~PositionExtractor() {}

	/**
	 * Estimates the most probable object position from a set of samples.
	 *
	 * @param[in] samples The samples.
	 * @return The the most probable object position if there is one.
	 */
	virtual boost::optional<Sample> extract(const std::vector<Sample>& samples) = 0;
};

} /* namespace tracking */
#endif /* POSITIONEXTRACTOR_H_ */
