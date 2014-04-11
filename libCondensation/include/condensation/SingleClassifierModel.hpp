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

using std::unordered_map;
using std::pair;

namespace imageprocessing {
class FeatureExtractor;
class Patch;
}
using imageprocessing::FeatureExtractor;
using imageprocessing::Patch;

namespace classification {
class ProbabilisticClassifier;
}
using classification::ProbabilisticClassifier;

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
	SingleClassifierModel(shared_ptr<FeatureExtractor> featureExtractor, shared_ptr<ProbabilisticClassifier> classifier);

	~SingleClassifierModel();

	void update(shared_ptr<VersionedImage> image);

	void evaluate(Sample& sample) const;

	using MeasurementModel::evaluate;

private:

	/**
	 * Classifies a patch using its data.
	 *
	 * @param[in] patch The patch.
	 * @return The classification result.
	 */
	pair<bool, double> classify(shared_ptr<Patch> patch) const;

	shared_ptr<FeatureExtractor> featureExtractor;  ///< The feature extractor.
	shared_ptr<ProbabilisticClassifier> classifier; ///< The classifier.
	mutable unordered_map<shared_ptr<Patch>, pair<bool, double>> cache; ///< The classification result cache.
};

} /* namespace condensation */
#endif /* SINGLECLASSIFIERMODEL_HPP_ */
