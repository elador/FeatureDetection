/*
 * SelfLearningWvmSvmModel.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGWVMSVMMODEL_H_
#define SELFLEARNINGWVMSVMMODEL_H_

#include "tracking/PatchDuplicateFilter.h"
#include "tracking/LearningMeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <utility>

class FdImage;
class VDetectorVectorMachine;
class OverlapElimination;

using boost::shared_ptr;
using std::pair;
using std::string;

namespace classification {

class FeatureVector;
class FeatureExtractor;
class LibSvmClassifier;
class LibSvmTraining;
}
using namespace classification;

namespace tracking {

/**
 * Measurement model that trains a dynamic SVM using self-learning. At first, it will use a combination of a WVM for
 * quick elimination and a static SVM for final classification, and later it will use a dynamically trained SVM. An
 * overlap elimination further reduces the amount of samples that are classified by the static SVM to speed up the
 * evaluation. The dynamic SVM will be trained from the samples with the highest and lowest SVM certainty. Depending
 * on the quality of the training, the dynamic SVM or WVM and static SVM will be used. In the latter mode, the weight
 * of the samples will be the product of the certainties from the WVM and SVM, they will be regarded as being
 * independent (although they are not). The certainties for the SVM of samples that are not evaluated by it will be
 * chosen to be 0.5 (unknown).
 */
class SelfLearningWvmSvmModel : public LearningMeasurementModel, public PatchDuplicateFilter {
public:

	/**
	 * Constructs a new self-learning WVM SVM measurement model. The machines and algorithm
	 * must have been initialized.
	 *
	 * @param[in] The feature extractor used with the dynamic SVM.
	 * @param[in] wvm The fast WVM.
	 * @param[in] staticSvm The slower static SVM.
	 * @param[in] dynamicSvm The dynamic SVM that will be re-trained.
	 * @param[in] oe The overlap elimination algorithm.
	 * @param[in] svmTraining The SVM training algorithm.
	 * @param[in] positiveThreshold The certainty threshold for patches to be used as positive samples (must be exceeded).
	 * @param[in] negativeThreshold The certainty threshold for patches to be used as negative samples (must fall below).
	 */
	explicit SelfLearningWvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> staticSvm,
			shared_ptr<LibSvmClassifier> dynamicSvm, shared_ptr<OverlapElimination> oe,
			shared_ptr<LibSvmTraining> svmTraining, double positiveThreshold = 0.85, double negativeThreshold = 0.05);

	/**
	 * Constructs a new self-learning WVM SVM measurement model with default SVMs and overlap
	 * elimination algorithm.
	 *
	 * @param[in] configFilename The name of the Matlab config file containing the parameters.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 */
	explicit SelfLearningWvmSvmModel(string configFilename, string negativesFilename);

	~SelfLearningWvmSvmModel();

	void evaluate(Mat& image, vector<Sample>& samples);

	void reset();

	void update();

	void update(vector<Sample>& positiveSamples, vector<Sample>& negativeSamples);

	inline bool wasUsingDynamicModel() {
		return wasUsingDynamicSvm;
	}

	inline bool isUsingDynamicModel() {
		return useDynamicSvm;
	}

private:

	/**
	 * Extracts the feature vectors and adds them to the training samples.
	 *
	 * @param[in] trainingSamples The training samples.
	 * @param[in] pairs The training samples paired with their probability.
	 */
	void addTrainingSamples(vector<shared_ptr<FeatureVector> >& trainingSamples,
			vector<pair<shared_ptr<FeatureVector>, double> >& pairs);

	/**
	 * Adds additional training samples.
	 *
	 * @param[in] trainingSamples The training samples.
	 * @param[in] samples The additional samples.
	 */
	void addTrainingSamples(vector<shared_ptr<FeatureVector> >& trainingSamples, vector<Sample>& samples);

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<VDetectorVectorMachine> wvm;        ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> staticSvm;  ///< The slower static SVM.
	shared_ptr<LibSvmClassifier> dynamicSvm;       ///< The dynamic SVM that will be re-trained.
	shared_ptr<OverlapElimination> oe;             ///< The overlap elimination algorithm.
	shared_ptr<LibSvmTraining> svmTraining;        ///< The SVM training algorithm.
	bool useDynamicSvm;      ///< Flag that indicates whether the dynamic SVM should be used in the next evaluation.
	bool wasUsingDynamicSvm; ///< Flag that indicates whether the dynamic SVM was used for the previous evaluation.
	double positiveThreshold; ///< The threshold for samples to be used as positive training samples (must be exceeded).
	double negativeThreshold; ///< The threshold for samples to be used as negative training samples (must fall below).
	vector<pair<shared_ptr<FeatureVector>, double> > positiveTrainingSamples; ///< The positive training samples.
	vector<pair<shared_ptr<FeatureVector>, double> > negativeTrainingSamples; ///< The negative training samples.
};

} /* namespace tracking */
#endif /* SELFLEARNINGWVMSVMMODEL_H_ */
