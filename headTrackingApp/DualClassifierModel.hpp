/*
 * SingleClassifierModel.hpp
 *
 *  Created on: 28.03.2013
 *      Author: poschmann
 */

#ifndef DUALCLASSIFIERMODEL_HPP_
#define DUALCLASSIFIERMODEL_HPP_

#include "condensation/MeasurementModel.hpp"

namespace imageprocessing {
class FeatureExtractor;
}
using imageprocessing::FeatureExtractor;

namespace classification {
class ProbabilisticClassifier;
}
using classification::ProbabilisticClassifier;

namespace condensation {

/**
 * Simple measurement model that evaluates each sample by classifying its extracted feature vector.
 */
class DualClassifierModel : public MeasurementModel {
public:

	/**
	 * Constructs a new single classifier measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier.
	 */
	DualClassifierModel(shared_ptr<FeatureExtractor> featureExtractor, shared_ptr<ProbabilisticClassifier> classifier,
			shared_ptr<FeatureExtractor> filterFeatureExtractor, shared_ptr<ProbabilisticClassifier> filter);

	virtual ~DualClassifierModel();

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples);

protected:

	shared_ptr<FeatureExtractor> featureExtractor;  ///< The feature extractor.
	shared_ptr<ProbabilisticClassifier> classifier; ///< The classifier.
	shared_ptr<FeatureExtractor> filterFeatureExtractor;  ///< The feature extractor.
	shared_ptr<ProbabilisticClassifier> filter; ///< The classifier.
};

} /* namespace condensation */
#endif /* DUALCLASSIFIERMODEL_HPP_ */
