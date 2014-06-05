/*
 * FilteringClassifierModel.hpp
 *
 *  Created on: 24.07.2013
 *      Author: poschmann
 */

#ifndef FILTERINGCLASSIFIERMODEL_HPP_
#define FILTERINGCLASSIFIERMODEL_HPP_

#include "condensation/MeasurementModel.hpp"
#include <unordered_map>

namespace imageprocessing {
class FeatureExtractor;
class Patch;
}

namespace classification {
class BinaryClassifier;
class ProbabilisticClassifier;
}

namespace condensation {

/**
 * Measurement model that uses a classifier as a filter before or after passing the samples to another measurement model.
 */
class FilteringClassifierModel : public MeasurementModel {
public:

	/**
	 * The filtering behavior. There are two possible ways: The filter runs beforehand and the samples that do not pass the
	 * filter will get a weight of zero and are not evaluated further or the filter runs afterwards on all samples that are
	 * considered to be possible object locations and samples that do not pass the filter will be set to not be a valid object
	 * location, but remain their weight.
	 */
	enum class Behavior {
		RESET_WEIGHT, ///< Samples that do not pass the filter will have their weight reset to zero (and will not be evaluated further).
		KEEP_WEIGHT   ///< Samples will keep their weight, the filter will only revert the object property if the sample does not pass.
	};

	/**
	 * Constructs a new filtering classifier model that is based on a single classifier model.
	 *
	 * @param[in] filterFeatureExtractor The feature extractor used for the filter.
	 * @param[in] filter The filtering classifier.
	 * @param[in] featureExtractor The feature extractor used for the actual evaluation.
	 * @param[in] classifier The classifier that evaluates the samples.
	 * @param[in] behavior The filtering behavior.
	 */
	FilteringClassifierModel(
			std::shared_ptr<imageprocessing::FeatureExtractor> filterFeatureExtractor,
			std::shared_ptr<classification::BinaryClassifier> filter,
			std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::ProbabilisticClassifier> classifier,
			Behavior behavior);

	/**
	 * Constructs a new filtering classifier model.
	 *
	 * @param[in] filterFeatureExtractor The feature extractor used for the filter.
	 * @param[in] filter The filtering classifier.
	 * @param[in] measurementModel The measurement model used for evaluating the samples.
	 * @param[in] behavior The filtering behavior.
	 */
	FilteringClassifierModel(std::shared_ptr<imageprocessing::FeatureExtractor> filterFeatureExtractor,
			std::shared_ptr<classification::BinaryClassifier> filter, std::shared_ptr<MeasurementModel> measurementModel, Behavior behavior);

	~FilteringClassifierModel();

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	void evaluate(Sample& sample) const;

	void evaluate(std::shared_ptr<imageprocessing::VersionedImage> image, std::vector<std::shared_ptr<Sample>>& samples);

private:

	/**
	 * Determines whether a sample passes the filter.
	 *
	 * @param[in] sample The sample.
	 * @return True if the sample passed the filter, false otherwise.
	 */
	bool passesFilter(const Sample& sample) const;

	/**
	 * Determines whether a patch passes the filter.
	 *
	 * @param[in] patch The patch.
	 * @return True if the patch passed the filter, false otherwise.
	 */
	bool passesFilter(const std::shared_ptr<imageprocessing::Patch> patch) const;

	Behavior behavior; ///< The filtering behavior.
	std::shared_ptr<MeasurementModel> measurementModel; ///< The measurement model.
	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< The feature extractor used for the filter.
	std::shared_ptr<classification::BinaryClassifier> filter; ///< The filtering classifier.
	mutable std::unordered_map<std::shared_ptr<imageprocessing::Patch>, bool> cache; ///< The filter result cache.
};

} /* namespace condensation */
#endif /* FILTERINGCLASSIFIERMODEL_HPP_ */
