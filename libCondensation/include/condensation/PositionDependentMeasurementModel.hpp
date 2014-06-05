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

namespace imageprocessing {
class FeatureExtractor;
}

namespace classification {
class TrainableProbabilisticClassifier;
}

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
	 */
	PositionDependentMeasurementModel(
			std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::TrainableProbabilisticClassifier> classifier);

	/**
	 * Constructs a new position dependent measurement model that wraps another measurement model used for evaluation.
	 *
	 * @param[in] measurementModel The model used for evaluating the particles.
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] classifier The classifier that is used for evaluating of the particles.
	 */
	PositionDependentMeasurementModel(
			std::shared_ptr<MeasurementModel> measurementModel,
			std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::TrainableProbabilisticClassifier> classifier);

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	void evaluate(Sample& sample) const;

	void evaluate(std::shared_ptr<imageprocessing::VersionedImage> image, std::vector<std::shared_ptr<Sample>>& samples);

	bool isUsable() const;

	bool initialize(std::shared_ptr<imageprocessing::VersionedImage> image, Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples, const Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples);

	void reset();

	/**
	 * Changes the amounts of subsequent detections (misses) that lead this model to (not) being usable.
	 *
	 * @param[in] startFrameCount The amount of subsequent frames with detections that leads to being usable.
	 * @param[in] stopFrameCount The amount of subsequent frames without any detection that leads to not being usable again.
	 */
	void setFrameCounts(unsigned int startFrameCount = 3, unsigned int stopFrameCount = 0) {
		this->startFrameCount = startFrameCount;
		this->stopFrameCount = stopFrameCount;
	}

	/**
	 * Changes the training thresholds.
	 *
	 * @param[in] targetThreshold The threshold of the classification probability of the target position for the training to start.
	 * @param[in] confidenceThreshold The confidence threshold that must be undercut by examples to be used for training.
	 */
	void setThresholds(float targetThreshold = 0.7, float confidenceThreshold = 0.95) {
		this->targetThreshold = targetThreshold;
		this->confidenceThreshold = confidenceThreshold;
	}

	/**
	 * Changes the offset factors used to determine positive and negative training samples.
	 *
	 * @param[in] positiveOffsetFactor The position offset relative to the target size of still positive examples.
	 * @param[in] negativeOffsetFactor The minimum position offset relative to the target size of surely negative examples.
	 */
	void setOffsetFactors(float positiveOffsetFactor = 0.05, float negativeOffsetFactor = 0.5) {
		this->positiveOffsetFactor = positiveOffsetFactor;
		this->negativeOffsetFactor = negativeOffsetFactor;
	}

	/**
	 * Changes the sampling properties.
	 *
	 * @param[in] sampleNegativesAroundTarget Amount of combined dimensions that may be used for sampling negatives around the target.
	 * @param[in] sampleAdditionalNegatives The amount of additional negative examples sampled from the image (prefers bad classified ones).
	 * @param[in] sampleTestNegatives The amount of negative test examples randomly sampled from the image.
	 * @param[in] exploitSymmetry Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
	 */
	void setSamplingProperties(
			unsigned int sampleNegativesAroundTarget = 0, unsigned int sampleAdditionalNegatives = 10,
			unsigned int sampleTestNegatives = 10, bool exploitSymmetry = false) {
		this->sampleNegativesAroundTarget = sampleNegativesAroundTarget;
		this->sampleAdditionalNegatives = sampleAdditionalNegatives;
		this->sampleTestNegatives = sampleTestNegatives;
		this->exploitSymmetry = exploitSymmetry;
	}

private:

	/**
	 * Creates a new random sample.
	 *
	 * @param[in] image The image.
	 * @return The new sample.
	 */
	Sample createRandomSample(const cv::Mat& image);

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 * @param[in] pred Predicate that determines whether a feature vector should be used for training.
	 * @return The extracted feature vectors.
	 */
	std::vector<cv::Mat> getFeatureVectors(std::vector<Sample>& samples, std::function<bool(cv::Mat&)> pred);

	boost::mt19937 generator;          ///< Random number generator.
	boost::uniform_int<> distribution; ///< Uniform integer distribution.

	std::shared_ptr<MeasurementModel> measurementModel; ///< The model used for evaluating the particles.
	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< The feature extractor.
	std::shared_ptr<classification::TrainableProbabilisticClassifier> classifier; ///< The classifier that is used for evaluating of the particles.

	bool usable; ///< Flag that indicates whether this model may be used for evaluation.
	unsigned int frameCount;      ///< Count of frames without detections if usable, count of frames with detections otherwise.
	unsigned int startFrameCount; ///< The amount of subsequent frames with detections that leads to being usable.
	unsigned int stopFrameCount;  ///< The amount of subsequent frames without any detection that leads to not being usable again.

	float targetThreshold;      ///< The threshold of the classification probability of the target position for the training to start.
	float confidenceThreshold;  ///< The confidence threshold that must be fallen short of by examples to be used for training.
	float positiveOffsetFactor; ///< The position offset relative to the target size of still positive examples.
	float negativeOffsetFactor; ///< The minimum position offset relative to the target size of surely negative examples.
	unsigned int sampleNegativesAroundTarget; ///< Amount of dimensions that may be used for sampling negatives around the target at the same time.
	unsigned int sampleAdditionalNegatives;   ///< The amount of additional negative examples sampled from the image (prefers bad classified ones).
	unsigned int sampleTestNegatives;         ///< The amount of negative test examples randomly sampled from the image.
	bool exploitSymmetry; ///< Flag that indicates whether mirrored (y-axis) patches should be used for training, too.
};

} /* namespace condensation */
#endif /* POSITIONDEPENDENTMEASUREMENTMODEL_HPP_ */
