/*
 * FilteringStateExtractor.hpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#ifndef FILTERINGSTATEEXTRACTOR_HPP_
#define FILTERINGSTATEEXTRACTOR_HPP_

#include "condensation/StateExtractor.hpp"
#include <memory>

namespace condensation {

/**
 * State extractor that filters the samples leaving only those that represent the target and
 * delegates to another extractor.
 */
class FilteringStateExtractor : public StateExtractor {
public:

	/**
	 * Constructs a new filtering state extractor.
	 *
	 * @param[in] extractor The wrapped state extractor.
	 */
	explicit FilteringStateExtractor(std::shared_ptr<StateExtractor> extractor);

	std::shared_ptr<Sample> extract(const std::vector<std::shared_ptr<Sample>>& samples);

private:

	std::shared_ptr<StateExtractor> extractor; ///< The wrapped state extractor.
};

} /* namespace condensation */
#endif /* FILTERINGSTATEEXTRACTOR_HPP_ */
