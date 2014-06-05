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

namespace detection {

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
	ClassifiedPatch(std::shared_ptr<imageprocessing::Patch> patch, bool positive, double probability = 0.5) :
			patch(patch), positive(positive), probability(probability) {}

	/**
	 * Constructs a new classified patch.
	 *
	 * @param[in] patch The actual patch.
	 * @param[in] result The probabilistic classification result.
	 */
	ClassifiedPatch(std::shared_ptr<imageprocessing::Patch> patch, std::pair<bool, double> result) :
			patch(patch), positive(result.first), probability(result.second) {}

	/**
	 * @return The actual patch.
	 */
	std::shared_ptr<imageprocessing::Patch> getPatch() {
		return patch;
	}

	/**
	 * @return The actual patch.
	 */
	const std::shared_ptr<imageprocessing::Patch> getPatch() const {
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

	/**
	 * Determines whether this classified patch is less than another one using the probability. This patch is
	 * less than the other patch if the probability of this one is less than the probability of the other patch.
	 *
	 * @param[in] other The other patch.
	 * @return True if this patch comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator<(const ClassifiedPatch& other) const {
		return probability < other.probability;
	}

	/**
	 * Determines whether this classified patch is bigger than another one using the probability. This patch is
	 * considered bigger than the other patch if the probability of this one is bigger than the probability of
	 * the other patch.
	 *
	 * @param[in] other The other patch.
	 * @return True if this patch comes after the other in a strict weak ordering, false otherwise.
	 */
	bool operator>(const ClassifiedPatch& other) const {
		return probability > other.probability;
	}

private:

	std::shared_ptr<imageprocessing::Patch> patch; ///< The actual patch.
	bool positive; ///< Flag that indicates whether the patch was classified as positive.
	double probability; ///< Probability of the patch being positive.
};

} /* namespace detection */
#endif /* CLASSIFIEDPATCH_HPP_ */
