/*
 * SelfLearningMeasurementModel.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGMEASUREMENTMODEL_H_
#define SELFLEARNINGMEASUREMENTMODEL_H_

#include "tracking/AdaptiveMeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <utility>

using boost::shared_ptr;
using std::pair;

namespace classification {

class FeatureVector;
class FeatureExtractor;
class LibSvmClassifier;
class LibSvmTraining;
}
using namespace classification;

namespace tracking {

/**
 * Measurement model that adapts the classifier using self-learning. The classifier will be trained from the samples
 * with the highest and lowest probability.
 */
class SelfLearningMeasurementModel : public AdaptiveMeasurementModel {
public:

	/**
	 * Constructs a new self-learning WVM SVM measurement model. The machines and algorithm
	 * must have been initialized.
	 *
	 * @param[in] The feature extractor used with the dynamic SVM.
	 * @param[in] classifier The classifier that will be re-trained.
	 * @param[in] training The classifier training algorithm.
	 * @param[in] positiveThreshold The certainty threshold for patches to be used as positive samples (must be exceeded).
	 * @param[in] negativeThreshold The certainty threshold for patches to be used as negative samples (must fall below).
	 */
	explicit SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<LibSvmClassifier> classifier, shared_ptr<LibSvmTraining> training,
			double positiveThreshold = 0.85, double negativeThreshold = 0.05);

	~SelfLearningMeasurementModel();

	void evaluate(const Mat& image, vector<Sample>& samples);

	bool isUsable() {
		return usable;
	}

	void adapt(const Mat& image, const vector<Sample>& samples, const Sample& target);

	void adapt(const Mat& image, const vector<Sample>& samples);

	void reset();

private:

	/**
	 * Creates a list of feature vectors from the given pairs.
	 *
	 * @param[in] pairs The feature vectors paired with their probability.
	 */
	vector<shared_ptr<FeatureVector> > getFeatureVectors(vector<pair<shared_ptr<FeatureVector>, double> >& pairs);

	/**
	 * Creates a list of feature vectors from the given samples.
	 *
	 * @param[in] samples The samples.
	 */
	vector<shared_ptr<FeatureVector> > getFeatureVectors(vector<Sample>& samples);

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<LibSvmClassifier> classifier;       ///< The classifier that will be re-trained.
	shared_ptr<LibSvmTraining> training;           ///< The classifier training algorithm.
	bool usable; ///< Flag that indicates whether this model may be used for evaluation.
	double positiveThreshold; ///< The threshold for samples to be used as positive training samples (must be exceeded).
	double negativeThreshold; ///< The threshold for samples to be used as negative training samples (must fall below).
	vector<pair<shared_ptr<FeatureVector>, double> > positiveTrainingSamples; ///< The positive training samples.
	vector<pair<shared_ptr<FeatureVector>, double> > negativeTrainingSamples; ///< The negative training samples.
};

} /* namespace tracking */
#endif /* SELFLEARNINGMEASUREMENTMODEL_H_ */
