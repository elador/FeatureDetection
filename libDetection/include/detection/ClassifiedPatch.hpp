/*
 * ClassifiedPatch.hpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#ifndef CLASSIFIEDPATCH_HPP_
#define CLASSIFIEDPATCH_HPP_

#include "imageprocessing/Patch.hpp"
#include <memory>

using imageprocessing::Patch;
using std::shared_ptr;
using std::pair;

namespace imageprocessing {

/**
 * Patch with a classification result.
 */
class ClassifiedPatch {
public:

	/**
	 * Constructs a new classified patch.
	 *
	 * @param[in] patch The actual patch.
	 * @param[in] positive Flag that indicates whether the patch was classified as positive.
	 * @param[in] probability Probability of the patch being positive (optional).
	 */
	ClassifiedPatch(shared_ptr<Patch> patch, bool positive, double probability = (positive ? 1 : 0)) :
			patch(patch), positive(positive), probability(probability) {}

	/**
	 * Constructs a new classified patch.
	 *
	 * @param[in] patch The actual patch.
	 * @param[in] result The probabilistic classification result.
	 */
	ClassifiedPatch(shared_ptr<Patch> patch, pair<bool, double> result) :
			patch(patch), positive(result.first), probability(result.second) {}

	~ClassifiedPatch();

	/**
	 * @return The actual patch.
	 */
	shared_ptr<Patch> getPatch() {
		return patch;
	}

	/**
	 * @return The actual patch.
	 */
	const shared_ptr<Patch> getPatch() const {
		return patch;
	}

	/**
	 * @return True if the patch was classified as positive, false otherwise.
	 */
	bool isPositive() const {
		return positive;
	}

	/**
	 * @return Probability of the patch being positive.
	 */
	double getProbability() const {
		return probability;
	}

private:

	shared_ptr<Patch> patch; ///< The actual patch.
	bool positive;           ///< Flag that indicates whether the patch was classified as positive.
	double probability;      ///< Probability of the patch being positive.
};

} /* namespace imageprocessing */
#endif /* CLASSIFIEDPATCH_HPP_ */
