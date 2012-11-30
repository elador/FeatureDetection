/*
 * HeadLearningWvmSvmModel.h
 *
 *  Created on: 13.11.2012
 *      Author: poschmann
 */

#ifndef HEADLEARNINGWVMSVMMODEL_H_
#define HEADLEARNINGWVMSVMMODEL_H_

#include "tracking/PatchDuplicateFilter.h"
#include "tracking/LearningMeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>

class FdImage;
class VDetectorVectorMachine;
class OverlapElimination;

using boost::shared_ptr;
using std::string;

namespace classification {

class FeatureVector;
class FeatureExtractor;
class LibSvmClassifier;
class LibSvmTraining;
}
using namespace classification;
using namespace tracking;

/**
 * Measurement model that trains a dynamic SVM. At first, it will use a combination of a WVM for quick elimination
 * and a static SVM for final classification, and later it will use a dynamically trained SVM. An overlap elimination
 * further reduces the amount of samples that are classified by the static SVM to speed up the evaluation. The
 * dynamic SVM will be trained from the samples given to the update method. Depending on the quality of the training,
 * the dynamic SVM or WVM and static SVM will be used. In the latter mode, the weight of the samples will be the
 * product of the certainties from the WVM and SVM, they will be regarded as being independent (although they are not).
 * The certainties for the SVM of samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class HeadLearningWvmSvmModel : public LearningMeasurementModel, public PatchDuplicateFilter {
public:

	/**
	 * Constructs a new head learning WVM SVM measurement model. The machines and algorithm
	 * must have been initialized.
	 *
	 * @param[in] The feature extractor used with the dynamic SVM.
	 * @param[in] wvm The fast WVM.
	 * @param[in] staticSvm The slower static SVM.
	 * @param[in] dynamicSvm The dynamic SVM that will be re-trained.
	 * @param[in] oe The overlap elimination algorithm.
	 * @param[in] svmTraining The SVM training algorithm.
	 */
	explicit HeadLearningWvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor, shared_ptr<VDetectorVectorMachine> wvm,
			shared_ptr<VDetectorVectorMachine> staticSvm, shared_ptr<LibSvmClassifier> dynamicSvm,
			shared_ptr<OverlapElimination> oe, shared_ptr<LibSvmTraining> svmTraining);

	/**
	 * Constructs a new head learning WVM SVM measurement model with default SVMs and overlap
	 * elimination algorithm.
	 *
	 * @param[in] configFilename The name of the Matlab config file containing the parameters.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 */
	explicit HeadLearningWvmSvmModel(string configFilename, string negativesFilename);

	~HeadLearningWvmSvmModel();

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
	 * Creates a list of training samples.
	 *
	 * @param[in] samples The samples to create the training data from.
	 */
	vector<shared_ptr<FeatureVector> > getTrainingSamples(vector<Sample>& samples);

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor used with the dynamic SVM.
	shared_ptr<VDetectorVectorMachine> wvm;        ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> staticSvm;  ///< The slower static SVM.
	shared_ptr<LibSvmClassifier> dynamicSvm;       ///< The dynamic SVM that will be re-trained.
	shared_ptr<OverlapElimination> oe;             ///< The overlap elimination algorithm.
	shared_ptr<LibSvmTraining> svmTraining;        ///< The SVM training algorithm.
	bool useDynamicSvm;      ///< Flag that indicates whether the dynamic SVM should be used in the next evaluation.
	bool wasUsingDynamicSvm; ///< Flag that indicates whether the dynamic SVM was used for the previous evaluation.
};

#endif /* HEADLEARNINGWVMSVMMODEL_H_ */
