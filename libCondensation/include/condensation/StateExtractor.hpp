/*
 * StateExtractor.hpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef STATEEXTRACTOR_HPP_
#define STATEEXTRACTOR_HPP_

#include "condensation/Sample.hpp"
#include <vector>

namespace condensation {

/**
 * Extractor of the estimated target state given weighted samples of a particle filter.
 */
class StateExtractor {
public:

	virtual ~StateExtractor() {}

	/**
	 * Estimates the most probable target state from a set of samples.
	 *
	 * @param[in] samples The samples.
	 * @return The the most probable target state.
	 */
	virtual std::shared_ptr<Sample> extract(const std::vector<std::shared_ptr<Sample>>& samples) = 0;
};

} /* namespace condensation */
#endif /* STATEEXTRACTOR_HPP_ */
