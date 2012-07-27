/*
 * WvmOeSvmModel.h
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef WVMOESVMMODEL_H_
#define WVMOESVMMODEL_H_

#include "tracking/MeasurementModel.h"
#include <string>

class VDetectorVectorMachine;
class OverlapElimination;
class FdPatch;

namespace tracking {

/**
 * Measurement model that uses a WVM for quick elimination and evaluates the samples that remain after an
 * overlap elimination with a SVM. The weight of the samples will be the product of the certainties from
 * the two detectors, they will be regarded as being independent (although they are not). The certainties
 * for the SVM of samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class WvmOeSvmModel : public MeasurementModel {
public:

	/**
	 * Constructs a new WVM OE SVM measurement model. The machines and algorithm must have been initialized.
	 *
	 * @param[in] wvm The fast WVM. Will be deleted at destruction.
	 * @param[in] svm The slower SVM. Will be deleted at destruction.
	 * @param[in] oe The overlap elimination algorithm. Will be deleted at destruction.
	 */
	explicit WvmOeSvmModel(VDetectorVectorMachine* wvm, VDetectorVectorMachine* svm, OverlapElimination* oe);

	/**
	 * Constructs a new WVM OE SVM measurement model with default SVMs and overlapping elimination.
	 *
	 * @param[in] configFilename The name of the Matlab config file containing the parameters.
	 */
	explicit WvmOeSvmModel(std::string configFilename);
	virtual ~WvmOeSvmModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

private:

	/**
	 * Eliminates all but the ten best patches.
	 *
	 * @param[in] patches The patches.
	 * @param[in] detectorId The identifier of the detector used for computing the certainties.
	 * @return A new vector containing the remaining patches.
	 */
	std::vector<FdPatch*> eliminate(const std::vector<FdPatch*>& patches, std::string detectorId);

	VDetectorVectorMachine* wvm; ///< The fast WVM.
	VDetectorVectorMachine* svm; ///< The slower SVM.
	OverlapElimination* oe;      ///< The overlap elimination algorithm.
};

} /* namespace tracking */
#endif /* WVMOESVMMODEL_H_ */
