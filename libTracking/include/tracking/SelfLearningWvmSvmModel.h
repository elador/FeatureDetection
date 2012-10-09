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

class VDetectorVectorMachine;
class OverlapElimination;

using boost::shared_ptr;

namespace tracking {

class SvmTraining;
class ChangableDetectorSvm;

/**
 * Measurement model that trains a dynamic SVM using self-learning. Additionally, it uses a WVM for quick
 * elimination and evaluates the samples that remain after that. When using the static SVM, an overlap
 * elimination further reduces the amount of samples. A dynamic SVM will be trained from the samples with
 * the highest and lowest SVM certainty. Depending on the quality of the training, the dynamic or static
 * SVM will be used. The weight of the samples will be the product of the certainties from the WVM and SVM,
 * they will be regarded as being independent (although they are not). The certainties for the SVM of
 * samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class SelfLearningWvmSvmModel : public LearningMeasurementModel, public PatchDuplicateFilter {
public:

	/**
	 * Constructs a new self-learning WVM SVM measurement model. The machines and algorithm
	 * must have been initialized.
	 *
	 * @param[in] wvm The fast WVM.
	 * @param[in] staticSvm The slower static SVM.
	 * @param[in] dynamicSvm The dynamic SVM that will be re-trained.
	 * @param[in] oe The overlap elimination algorithm.
	 * @param[in] svmTraining The SVM training algorithm.
	 * @param[in] positiveThreshold The certainty threshold for patches to be used as positive samples (must be exceeded).
	 * @param[in] negativeThreshold The certainty threshold for patches to be used as negative samples (must fall below).
	 */
	explicit SelfLearningWvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm,
			shared_ptr<VDetectorVectorMachine> staticSvm, shared_ptr<ChangableDetectorSvm> dynamicSvm,
			shared_ptr<OverlapElimination> oe, shared_ptr<SvmTraining> svmTraining,
			double positiveThreshold = 0.85, double negativeThreshold = 0.05);

	/**
	 * Constructs a new self-learning WVM SVM measurement model with default SVMs and overlap
	 * elimination algorithm.
	 *
	 * @param[in] configFilename The name of the Matlab config file containing the parameters.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 */
	explicit SelfLearningWvmSvmModel(std::string configFilename, std::string negativesFilename);

	~SelfLearningWvmSvmModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

	void reset();

	void update();

	void update(FdImage* image, std::vector<Sample>& positiveSamples, std::vector<Sample>& negativeSamples);

	inline bool isUsingDynamicModel() {
		return usingDynamicSvm;
	}

private:

	/**
	 * Creates patches of samples and adds them to an existing list of patches.
	 *
	 * @param[in] patches The existing patches.
	 * @param[in] samples The samples to create new patches from.
	 */
	void addSamples(FdImage* image, std::vector<FdPatch*>& patches, std::vector<Sample>& samples);

	shared_ptr<VDetectorVectorMachine> wvm;       ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> staticSvm; ///< The slower static SVM.
	shared_ptr<ChangableDetectorSvm> dynamicSvm;  ///< The dynamic SVM that will be re-trained.
	shared_ptr<OverlapElimination> oe;            ///< The overlap elimination algorithm.
	shared_ptr<SvmTraining> svmTraining;          ///< The SVM training algorithm.
	bool usingDynamicSvm;     ///< Flag that indicates whether the dynamic SVM is used.
	double positiveThreshold; ///< The threshold for patches to be used as positive samples (must be exceeded).
	double negativeThreshold; ///< The threshold for patches to be used as negative samples (must fall below).
	std::vector<FdPatch*> positiveTrainingPatches; ///< The positive training patches for the next update.
	std::vector<FdPatch*> negativeTrainingPatches; ///< The negative training patches for the next update.
};

} /* namespace tracking */
#endif /* SELFLEARNINGWVMSVMMODEL_H_ */
