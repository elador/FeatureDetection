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
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
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
	 * @param[in] startFrameCount The amount of subsequent frames with detections that leads to being usable.
	 * @param[in] stopFrameCount The amount of subsequent frames without any detection that leads to not being usable again.
	 * @param[in] positiveOffsetFactor The position offset relative to the target size of still positive examples.
	 * @param[in] negativeOffsetFactor The minimum position offset relative to the target size of surely negative examples.
	 * @param[in] sampleNegativesAroundTarget Indicates whether negative examples should be sampled around the target.
	 * @param[in] sampleFalsePositives Indicates whether positive detections other than the target should be negative examples.
	 * @param[in] randomNegatives The amount of additional random negative examples sampled from the image.
	 */
	explicit PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableClassifier> classifier, int startFrameCount = 3, int stopFrameCount = 20,
			float positiveOffsetFactor = 0.05, float negativeOffsetFactor = 0.5,
			bool sampleNegativesAroundTarget = true, bool sampleFalsePositives = true, unsigned int randomNegatives = 0);

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
	 * Creates a new random sample.
	 *
	 * @param[in] image The image.
	 * @return The new sample.
	 */
	Sample createRandomSample(const Mat& image);

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 */
	vector<shared_ptr<FeatureVector> > getFeatureVectors(vector<Sample>& samples);

	boost::mt19937 generator;          ///< Random number generator.
	boost::uniform_int<> distribution; ///< Uniform integer distribution.

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<TrainableClassifier> classifier;    ///< The classifier that will be re-trained.

	bool usable;         ///< Flag that indicates whether this model may be used for evaluation.
	int frameCount;      ///< Count of frames without detections if usable, count of frames with detections otherwise.
	int startFrameCount; ///< The amount of subsequent frames with detections that leads to being usable.
	int stopFrameCount;  ///< The amount of subsequent frames without any detection that leads to not being usable again.

	float positiveOffsetFactor; ///< The position offset relative to the target size of still positive examples.
	float negativeOffsetFactor; ///< The minimum position offset relative to the target size of surely negative examples.
	bool sampleNegativesAroundTarget; ///< Indicates whether negative examples should be sampled around the target.
	bool sampleFalsePositives; ///< Indicates whether positive detections other than the target should be negative examples.
	unsigned int randomNegatives; ///< The amount of additional random negative examples sampled from the image.
};

} /* namespace tracking */
#endif /* POSITIONDEPENDENTMEASUREMENTMODEL_H_ */
