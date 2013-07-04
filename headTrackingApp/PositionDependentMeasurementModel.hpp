/*
 * PositionDependentMeasurementModel.hpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef POSITIONDEPENDENTMEASUREMENTMODEL2_HPP_
#define POSITIONDEPENDENTMEASUREMENTMODEL2_HPP_

#include "DualClassifierModel.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
#include "opencv2/core/core.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include <string>

using cv::Mat;
using std::function;

namespace classification {
class TrainableProbabilisticClassifier;
}
using classification::TrainableProbabilisticClassifier;

namespace condensation {

/**
 * Measurement model that adapts the classifier using the target position for positive samples, the neighborhood
 * for negative samples and positively evaluated samples at positions other than the target as additional negative
 * samples.
 */
class PositionDependentMeasurementModel2 : public DualClassifierModel, public AdaptiveMeasurementModel {
public:

	/**
	 * Constructs a new position dependent measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier that will be re-trained.
	 * @param[in] startFrameCount The amount of subsequent frames with detections that leads to being usable.
	 * @param[in] stopFrameCount The amount of subsequent frames without any detection that leads to not being usable again.
	 * @param[in] targetThreshold The threshold of the classification probability of the target position for the training to start.
	 * @param[in] confidenceThreshold The confidence threshold that must be undercut by examples to be used for training.
	 * @param[in] positiveOffsetFactor The position offset relative to the target size of still positive examples.
	 * @param[in] negativeOffsetFactor The minimum position offset relative to the target size of surely negative examples.
	 * @param[in] sampleNegativesAroundTarget Indicates whether negative examples should be sampled around the target.
	 * @param[in] sampleFalsePositives Indicates whether positive detections other than the target should be negative examples.
	 * @param[in] randomNegatives The amount of additional random negative examples sampled from the image.
	 * @param[in] exploitSymmetry Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
	 */
	PositionDependentMeasurementModel2(shared_ptr<FeatureExtractor> filterFeatureExtractor,
			shared_ptr<ProbabilisticClassifier> filter, shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableProbabilisticClassifier> classifier, int startFrameCount = 3, int stopFrameCount = 20,
			float targetThreshold = 0.7, float confidenceThreshold = 0.95, float positiveOffsetFactor = 0.05, float negativeOffsetFactor = 0.5,
			bool sampleNegativesAroundTarget = true, bool sampleFalsePositives = true, unsigned int randomNegatives = 0, bool exploitSymmetry = false);

	~PositionDependentMeasurementModel2();

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
		DualClassifierModel::evaluate(image, samples);
	}

	bool isUsable() {
		return usable;
	}

	void adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target);

	void adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples);

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
	 * @param[in] pred Predicate that determines whether a feature vector should be used for training.
	 * @return The extracted feature vectors.
	 */
	vector<Mat> getFeatureVectors(vector<Sample>& samples, function<bool(Mat&)> pred);

	boost::mt19937 generator;          ///< Random number generator.
	boost::uniform_int<> distribution; ///< Uniform integer distribution.

	shared_ptr<TrainableProbabilisticClassifier> classifier; ///< The classifier.

	bool usable;         ///< Flag that indicates whether this model may be used for evaluation.
	int frameCount;      ///< Count of frames without detections if usable, count of frames with detections otherwise.
	int startFrameCount; ///< The amount of subsequent frames with detections that leads to being usable.
	int stopFrameCount;  ///< The amount of subsequent frames without any detection that leads to not being usable again.

	float targetThreshold;      ///< The threshold of the classification probability of the target position for the training to start.
	float confidenceThreshold;  ///< The confidence threshold that must be fallen short of by examples to be used for training.
	float positiveOffsetFactor; ///< The position offset relative to the target size of still positive examples.
	float negativeOffsetFactor; ///< The minimum position offset relative to the target size of surely negative examples.
	bool sampleNegativesAroundTarget; ///< Indicates whether negative examples should be sampled around the target.
	bool sampleFalsePositives;        ///< Indicates whether positive detections other than the target should be negative examples.
	unsigned int randomNegatives;     ///< The amount of additional random negative examples sampled from the image.
	bool exploitSymmetry;             ///< Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
};

} /* namespace condensation */
#endif /* POSITIONDEPENDENTMEASUREMENTMODEL2_HPP_ */
