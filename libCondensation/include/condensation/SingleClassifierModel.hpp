/*
 * SingleClassifierModel.hpp
 *
 *  Created on: 28.03.2013
 *      Author: poschmann
 */

#ifndef SINGLECLASSIFIERMODEL_HPP_
#define SINGLECLASSIFIERMODEL_HPP_

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
class SingleClassifierModel : public MeasurementModel {
public:

	/**
	 * Constructs a new single classifier measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier.
	 */
	SingleClassifierModel(shared_ptr<FeatureExtractor> featureExtractor, shared_ptr<ProbabilisticClassifier> classifier);

	virtual ~SingleClassifierModel();

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples);

protected:

	shared_ptr<FeatureExtractor> featureExtractor;  ///< The feature extractor.
	shared_ptr<ProbabilisticClassifier> classifier; ///< The classifier.
};

} /* namespace condensation */
#endif /* SINGLECLASSIFIERMODEL_HPP_ */
