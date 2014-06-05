/*
 * SelfLearningMeasurementModel.hpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGMEASUREMENTMODEL_HPP_
#define SELFLEARNINGMEASUREMENTMODEL_HPP_

#include "condensation/AdaptiveMeasurementModel.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <unordered_map>
#include <utility>

namespace imageprocessing {
class Patch;
class FeatureExtractor;
}

namespace classification {
class TrainableProbabilisticClassifier;
}

namespace detection {
class ClassifiedPatch;
}

namespace condensation {

/**
 * Measurement model that adapts the classifier using self-learning. The classifier will be trained from the samples
 * with the highest and lowest probability.
 */
class SelfLearningMeasurementModel : public AdaptiveMeasurementModel {
public:

	/**
	 * Constructs a new self-learning measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor used with the dynamic SVM.
	 * @param[in] classifier The classifier that will be re-trained.
	 * @param[in] positiveThreshold The certainty threshold for patches to be used as positive samples (must be exceeded).
	 * @param[in] negativeThreshold The certainty threshold for patches to be used as negative samples (must fall below).
	 */
	SelfLearningMeasurementModel(std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::TrainableProbabilisticClassifier> classifier,
			double positiveThreshold = 0.85, double negativeThreshold = 0.05);

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	void evaluate(Sample& sample) const;

	using AdaptiveMeasurementModel::evaluate;

	bool isUsable() const {
		return usable;
	}

	bool initialize(std::shared_ptr<imageprocessing::VersionedImage> image, Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples, const Sample& target);

	bool adapt(std::shared_ptr<imageprocessing::VersionedImage> image, const std::vector<std::shared_ptr<Sample>>& samples);

	void reset();

private:

	/**
	 * Creates a list of feature vectors from the given classified patches.
	 *
	 * @param[in] pairs The extracted and classified patches.
	 * @return The feature vectors.
	 */
	std::vector<cv::Mat> getFeatureVectors(const std::vector<std::shared_ptr<detection::ClassifiedPatch>>& patches);

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 * @return The extracted feature vectors.
	 */
	std::vector<cv::Mat> getFeatureVectors(std::vector<std::shared_ptr<Sample>>& samples);

	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	std::shared_ptr<classification::TrainableProbabilisticClassifier> classifier; ///< The classifier that will be re-trained.
	bool usable; ///< Flag that indicates whether this model may be used for evaluation.
	mutable std::unordered_map<std::shared_ptr<imageprocessing::Patch>, std::pair<bool, double>> cache; ///< The classification result cache.
	double positiveThreshold; ///< The threshold for samples to be used as positive training samples (must be exceeded).
	double negativeThreshold; ///< The threshold for samples to be used as negative training samples (must fall below).
	mutable std::vector<std::shared_ptr<detection::ClassifiedPatch>> positiveTrainingExamples; ///< The positive training examples.
	mutable std::vector<std::shared_ptr<detection::ClassifiedPatch>> negativeTrainingExamples; ///< The negative training examples.
};

} /* namespace condensation */
#endif /* SELFLEARNINGMEASUREMENTMODEL_HPP_ */
