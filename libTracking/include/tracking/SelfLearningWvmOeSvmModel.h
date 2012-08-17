/*
 * SelfLearningWvmOeSvmModel.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SELFLEARNINGWVMOESVMMODEL_H_
#define SELFLEARNINGWVMOESVMMODEL_H_

#include "tracking/MeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>

class VDetectorVectorMachine;
class OverlapElimination;
class FdPatch;

using boost::shared_ptr;

namespace tracking {

class SvmTraining;
class ChangableDetectorSvm;

/**
 * Measurement model that trains a dynamic SVM using self-learning. Additionally, it uses a WVM for quick
 * elimination and evaluates the samples that remain after an overlap elimination with a static or the
 * dynamic SVM. The weight of the samples will be the product of the certainties from the two detectors,
 * they will be regarded as being independent (although they are not). The certainties for the SVM of
 * samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class SelfLearningWvmOeSvmModel : public MeasurementModel {
public:

	/**
	 * Constructs a new self-learning WVM OE SVM measurement model. The machines and algorithm
	 * must have been initialized.
	 *
	 * @param[in] wvm The fast WVM.
	 * @param[in] staticSvm The slower static SVM.
	 * @param[in] dynamicSvm The dynamic SVM that will be re-trained.
	 * @param[in] oe The overlap elimination algorithm.
	 * @param[in] svmTraining The SVM training algorithm.
	 * @param[in] positiveThreshold The threshold for patches to be used as positive samples (must be exceeded).
	 * @param[in] negativeThreshold The threshold for patches to be used as negative samples (must fall below).
	 */
	explicit SelfLearningWvmOeSvmModel(shared_ptr<VDetectorVectorMachine> wvm,
			shared_ptr<VDetectorVectorMachine> staticSvm, shared_ptr<ChangableDetectorSvm> dynamicSvm,
			shared_ptr<OverlapElimination> oe, shared_ptr<SvmTraining> svmTraining,
			double positiveThreshold = 0.85, double negativeThreshold = 0.4);

	/**
	 * Constructs a new self-learning WVM OE SVM measurement model with default SVMs and overlap
	 * elimination algorithm.
	 *
	 * @param[in] configFilename The name of the Matlab config file containing the parameters.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 */
	explicit SelfLearningWvmOeSvmModel(std::string configFilename, std::string negativesFilename);

	~SelfLearningWvmOeSvmModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

	/**
	 * @return True if the dynamic SVM was used for the last evaluation, false otherwise.
	 */
	inline bool isUsingDynamicSvm() {
		return selfLearningActive && usingDynamicSvm;
	}

	/**
	 * @return True if self-learning is active, false otherwise.
	 */
	inline bool isSelfLearningActive() {
		return selfLearningActive;
	}

	/**
	 * @param[in] active Flag that indicates whether self-learning should be active.
	 */
	inline void setSelfLearningActive(bool active) {
		selfLearningActive = active;
	}

private:

	/**
	 * Eliminates all but the ten best patches.
	 *
	 * @param[in] patches The patches.
	 * @param[in] detectorId The identifier of the detector used for computing the certainties.
	 * @return A new vector containing the remaining patches.
	 */
	std::vector<FdPatch*> eliminate(std::vector<FdPatch*> patches, std::string detectorId);

	shared_ptr<VDetectorVectorMachine> wvm;       ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> staticSvm; ///< The slower static SVM.
	shared_ptr<ChangableDetectorSvm> dynamicSvm;  ///< The dynamic SVM that will be re-trained.
	shared_ptr<OverlapElimination> oe;            ///< The overlap elimination algorithm.
	shared_ptr<SvmTraining> svmTraining;          ///< The SVM training algorithm.
	bool usingDynamicSvm;     ///< Flag that indicates whether the dynamic SVM is used.
	double positiveThreshold; ///< The threshold for patches to be used as positive samples (must be exceeded).
	double negativeThreshold; ///< The threshold for patches to be used as negative samples (must fall below).
	bool selfLearningActive;  ///< Flag that indicates whether self-learning is active.
};

} /* namespace tracking */
#endif /* SELFLEARNINGWVMOESVMMODEL_H_ */
