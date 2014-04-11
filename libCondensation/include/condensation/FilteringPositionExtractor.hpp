/*
 * FilteringPositionExtractor.hpp
 *
 *  Created on: 27.07.2012
 *      Author: poschmann
 */

#ifndef FILTERINGPOSITIONEXTRACTOR_HPP_
#define FILTERINGPOSITIONEXTRACTOR_HPP_

#include "condensation/PositionExtractor.hpp"
#include <memory>

using std::shared_ptr;

namespace condensation {

/**
 * Position extractor that filters the samples leaving only those that represent the searched object
 * and delegates to another extractor.
 */
class FilteringPositionExtractor : public PositionExtractor {
public:

	/**
	 * Constructs a new filtering position extractor.
	 *
	 * @param[in] extractor The wrapped position extractor.
	 */
	explicit FilteringPositionExtractor(shared_ptr<PositionExtractor> extractor);

	~FilteringPositionExtractor();

	shared_ptr<Sample> extract(const vector<shared_ptr<Sample>>& samples);

private:
	shared_ptr<PositionExtractor> extractor; ///< The wrapped position extractor.
};

} /* namespace condensation */
#endif /* FILTERINGPOSITIONEXTRACTOR_HPP_ */
