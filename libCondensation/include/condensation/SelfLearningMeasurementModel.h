/*
 * SelfLearningMeasurementModel.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGMEASUREMENTMODEL_H_
#define SELFLEARNINGMEASUREMENTMODEL_H_

#include "condensation/AdaptiveMeasurementModel.h"
#include "opencv2/core/core.hpp"
#include <string>
#include <utility>

using cv::Mat;
using std::pair;

namespace imageprocessing {
class Patch;
class FeatureExtractor;
}
using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;

namespace classification {
class TrainableProbabilisticClassifier;
}
using classification::TrainableProbabilisticClassifier;

namespace detection {
class ClassifiedPatch;
}
using detection::ClassifiedPatch;

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
	SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<TrainableProbabilisticClassifier> classifier, double positiveThreshold = 0.85, double negativeThreshold = 0.05);

	~SelfLearningMeasurementModel();

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples);

	bool isUsable() {
		return usable;
	}

	void adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target);

	void adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples);

	void reset();

private:

	/**
	 * Creates a list of feature vectors from the given classified patches.
	 *
	 * @param[in] pairs The extracted and classified patches.
	 * @return The feature vectors.
	 */
	vector<Mat> getFeatureVectors(const vector<shared_ptr<ClassifiedPatch>>& patches);

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 * @return The extracted feature vectors.
	 */
	vector<Mat> getFeatureVectors(vector<Sample>& samples);

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<TrainableProbabilisticClassifier> classifier;    ///< The classifier that will be re-trained.
	bool usable; ///< Flag that indicates whether this model may be used for evaluation.
	double positiveThreshold; ///< The threshold for samples to be used as positive training samples (must be exceeded).
	double negativeThreshold; ///< The threshold for samples to be used as negative training samples (must fall below).
	vector<shared_ptr<ClassifiedPatch>> positiveTrainingExamples; ///< The positive training examples.
	vector<shared_ptr<ClassifiedPatch>> negativeTrainingExamples; ///< The negative training examples.
};

} /* namespace condensation */
#endif /* SELFLEARNINGMEASUREMENTMODEL_H_ */
