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
	 */
	ClassificationBasedStateValidator(
			std::shared_ptr<imageprocessing::FeatureExtractor> extractor, std::shared_ptr<classification::BinaryClassifier> classifier);

	bool isValid(const Sample& target, const std::vector<std::shared_ptr<Sample>>& samples,
			std::shared_ptr<imageprocessing::VersionedImage> image);

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> extractor; ///< Feature extractor.
	std::shared_ptr<classification::BinaryClassifier> classifier; ///< Feature classifier.
};

} /* namespace condensation */
#endif /* CLASSIFICATIONBASEDSTATEVALIDATOR_HPP_ */
