/*
 * ClassificationBasedStateValidator.hpp
 *
 *  Created on: 10.04.2014
 *      Author: poschmann
 */

#ifndef CLASSIFICATIONBASEDSTATEVALIDATOR_HPP_
#define CLASSIFICATIONBASEDSTATEVALIDATOR_HPP_

#include "condensation/StateValidator.hpp"

namespace imageprocessing {
class FeatureExtractor;
}

namespace classification {
class BinaryClassifier;
}

namespace condensation {

/**
 * State validator that extracts and classifies patches around the target state. At least one of those patches
 * must be positively classified in order to validate the target state.
 */
class ClassificationBasedStateValidator : public StateValidator {
public:

	/**
	 * Constructs a new classification based state validator.
	 *
	 * @param[in] extractor Feature extractor.
	 * @param[in] classifier Feature classifier.
	 * @param[in] sizes Sizes around the target size that are searched for a valid patch.
	 * @param[in] displacements Displacements of the target position that are searched for a valid patch.
	 */
	ClassificationBasedStateValidator(
			std::shared_ptr<imageprocessing::FeatureExtractor> extractor, std::shared_ptr<classification::BinaryClassifier> classifier,
			std::vector<double> sizes = { 1 }, std::vector<double> displacements = { 0 });

	bool isValid(const Sample& target, const std::vector<std::shared_ptr<Sample>>& samples,
			std::shared_ptr<imageprocessing::VersionedImage> image);

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> extractor; ///< Feature extractor.
	std::shared_ptr<classification::BinaryClassifier> classifier; ///< Feature classifier.
	std::vector<double> sizes; ///< Sizes around the target size that are searched for a valid patch.
	std::vector<double> displacements; ///< Displacements of the target position that are searched for a valid patch.
};

} /* namespace condensation */
#endif /* CLASSIFICATIONBASEDSTATEVALIDATOR_HPP_ */
