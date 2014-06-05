/*
 * SingleClassifierModel.hpp
 *
 *  Created on: 28.03.2013
 *      Author: poschmann
 */

#ifndef SINGLECLASSIFIERMODEL_HPP_
#define SINGLECLASSIFIERMODEL_HPP_

#include "condensation/MeasurementModel.hpp"
#include <unordered_map>
#include <utility>

namespace imageprocessing {
class FeatureExtractor;
class Patch;
}

namespace classification {
class ProbabilisticClassifier;
}

namespace condensation {

/**
 * Simple measurement model that evaluates each sample by classifying its extracted feature vector.
 */
class SingleClassifierModel : public MeasurementModel {
public:

	/**
	 * Constructs a new single classifier measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier.
	 */
	SingleClassifierModel(std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::ProbabilisticClassifier> classifier);

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	void evaluate(Sample& sample) const;

	using MeasurementModel::evaluate;

private:

	/**
	 * Classifies a patch using its data.
	 *
	 * @param[in] patch The patch.
	 * @return The classification result.
	 */
	std::pair<bool, double> classify(std::shared_ptr<imageprocessing::Patch> patch) const;

	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< The feature extractor.
	std::shared_ptr<classification::ProbabilisticClassifier> classifier; ///< The classifier.
	mutable std::unordered_map<std::shared_ptr<imageprocessing::Patch>, std::pair<bool, double>> cache; ///< The classification result cache.
};

} /* namespace condensation */
#endif /* SINGLECLASSIFIERMODEL_HPP_ */
