/*
 * PositionDependentMeasurementModel.hpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef POSITIONDEPENDENTMEASUREMENTMODEL_HPP_
#define POSITIONDEPENDENTMEASUREMENTMODEL_HPP_

#include "condensation/AdaptiveMeasurementModel.hpp"
#include "opencv2/core/core.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include <string>
#include <functional>

using cv::Mat;
using std::function;

namespace imageprocessing {
class FeatureExtractor;
}
using imageprocessing::FeatureExtractor;

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
class PositionDependentMeasurementModel : public AdaptiveMeasurementModel {
public:

	/**
	 * Constructs a new position dependent measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier that is used for evaluating of the particles.
	 * @param[in] startFrameCount The amount of subsequent frames with detections that leads to being usable.
	 * @param[in] stopFrameCount The amount of subsequent frames without any detection that leads to not being usable again.
	 * @param[in] targetThreshold The threshold of the classification probability of the target position for the training to start.
	 * @param[in] confidenceThreshold The confidence threshold that must be undercut by examples to be used for training.
	 * @param[in] positiveOffsetFactor The position offset relative to the target size of still positive examples.
	 * @param[in] negativeOffsetFactor The minimum position offset relative to the target size of surely negative examples.
	 * @param[in] sampleNegativesAroundTarget Amount of dimensions that may be used for sampling negatives around the target at the same time. Zero means no sampling.
	 * @param[in] sampleFalsePositives Indicates whether positive detections other than the target should be negative examples.
	 * @param[in] randomNegatives The amount of additional random negative examples sampled from the image.
	 * @param[in] exploitSymmetry Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
	 */
	PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableProbabilisticClassifier> classifier, int startFrameCount = 3, int stopFrameCount = 20,
			float targetThreshold = 0.7, float confidenceThreshold = 0.95, float positiveOffsetFactor = 0.05, float negativeOffsetFactor = 0.5,
			int sampleNegativesAroundTarget = 0, bool sampleFalsePositives = true, unsigned int randomNegatives = 0, bool exploitSymmetry = false);

	/**
	 * Constructs a new position dependent measurement model that wraps another measurement model used for evaluation.
	 *
	 * @param[in] measurementModel The model used for evaluating the particles.
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier that is used for evaluating of the particles.
	 * @param[in] startFrameCount The amount of subsequent frames with detections that leads to being usable.
	 * @param[in] stopFrameCount The amount of subsequent frames without any detection that leads to not being usable again.
	 * @param[in] targetThreshold The threshold of the classification probability of the target position for the training to start.
	 * @param[in] confidenceThreshold The confidence threshold that must be undercut by examples to be used for training.
	 * @param[in] positiveOffsetFactor The position offset relative to the target size of still positive examples.
	 * @param[in] negativeOffsetFactor The minimum position offset relative to the target size of surely negative examples.
	 * @param[in] sampleNegativesAroundTarget Amount of dimensions that may be used for sampling negatives around the target at the same time. Zero means no sampling.
	 * @param[in] sampleFalsePositives Indicates whether positive detections other than the target should be negative examples.
	 * @param[in] randomNegatives The amount of additional random negative examples sampled from the image.
	 * @param[in] exploitSymmetry Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
	 */
	PositionDependentMeasurementModel(shared_ptr<MeasurementModel> measurementModel, shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableProbabilisticClassifier> classifier, int startFrameCount = 3, int stopFrameCount = 20,
			float targetThreshold = 0.7, float confidenceThreshold = 0.95, float positiveOffsetFactor = 0.05, float negativeOffsetFactor = 0.5,
			int sampleNegativesAroundTarget = 0, bool sampleFalsePositives = true, unsigned int randomNegatives = 0, bool exploitSymmetry = false);

	~PositionDependentMeasurementModel();

	void update(shared_ptr<VersionedImage> image);

	void evaluate(Sample& sample);

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples);

	bool isUsable();

	bool adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target);

	bool adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples);

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

	shared_ptr<MeasurementModel> measurementModel;           ///< The model used for evaluating the particles.
	shared_ptr<FeatureExtractor> featureExtractor;           ///< The feature extractor.
	shared_ptr<TrainableProbabilisticClassifier> classifier; ///< The classifier that is used for evaluating of the particles.

	bool usable;         ///< Flag that indicates whether this model may be used for evaluation.
	int frameCount;      ///< Count of frames without detections if usable, count of frames with detections otherwise.
	int startFrameCount; ///< The amount of subsequent frames with detections that leads to being usable.
	int stopFrameCount;  ///< The amount of subsequent frames without any detection that leads to not being usable again.

	float targetThreshold;      ///< The threshold of the classification probability of the target position for the training to start.
	float confidenceThreshold;  ///< The confidence threshold that must be fallen short of by examples to be used for training.
	float positiveOffsetFactor; ///< The position offset relative to the target size of still positive examples.
	float negativeOffsetFactor; ///< The minimum position offset relative to the target size of surely negative examples.
	int sampleNegativesAroundTarget; ///< Amount of dimensions that may be used for sampling negatives around the target at the same time. Zero means no sampling.
	bool sampleFalsePositives;        ///< Indicates whether positive detections other than the target should be negative examples.
	unsigned int randomNegatives;     ///< The amount of additional random negative examples sampled from the image.
	bool exploitSymmetry;             ///< Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
};

} /* namespace condensation */
#endif /* POSITIONDEPENDENTMEASUREMENTMODEL_HPP_ */
