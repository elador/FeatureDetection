/*
 * PositionDependentMeasurementModel.h
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef POSITIONDEPENDENTMEASUREMENTMODEL_H_
#define POSITIONDEPENDENTMEASUREMENTMODEL_H_

#include "tracking/AdaptiveMeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>

using boost::shared_ptr;

namespace classification {

class FeatureVector;
class FeatureExtractor;
class TrainableClassifier;
}
using namespace classification;

namespace tracking {

/**
 * Measurement model that adapts the classifier using the target position for positive samples, the neighborhood
 * for negative samples and positively evaluated samples at positions other than the target as additional negative
 * samples.
 */
class PositionDependentMeasurementModel : public AdaptiveMeasurementModel {
public:

	/**
	 * Constructs a new position dependent measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor used with the dynamic SVM.
	 * @param[in] classifier The classifier that will be re-trained.
	 */
	explicit PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableClassifier> classifier);

	~PositionDependentMeasurementModel();

	void evaluate(const Mat& image, vector<Sample>& samples);

	bool isUsable() {
		return usable;
	}

	void adapt(const Mat& image, const vector<Sample>& samples, const Sample& target);

	void adapt(const Mat& image, const vector<Sample>& samples);

	void reset();

private:

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 */
	vector<shared_ptr<FeatureVector> > getFeatureVectors(vector<Sample>& samples);

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<TrainableClassifier> classifier;    ///< The classifier that will be re-trained.
	bool usable; ///< Flag that indicates whether this model may be used for evaluation.
};

} /* namespace tracking */
#endif /* POSITIONDEPENDENTMEASUREMENTMODEL_H_ */
