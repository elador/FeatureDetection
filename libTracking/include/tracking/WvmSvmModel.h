/*
 * WvmSvmModel.h
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef WVMSVMMODEL_H_
#define WVMSVMMODEL_H_

#include "tracking/MeasurementModel.h"
#include "boost/shared_ptr.hpp"
#include <string>

class VDetectorVectorMachine;
class OverlapElimination;
class FdPatch;

using boost::shared_ptr;

namespace tracking {

/**
 * Measurement model that uses a WVM for quick elimination and evaluates the samples that remain after an
 * overlap elimination with a SVM. The weight of the samples will be the product of the certainties from
 * the two detectors, they will be regarded as being independent (although they are not). The certainties
 * for the SVM of samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class WvmSvmModel : public MeasurementModel {
public:

	/**
	 * Constructs a new WVM SVM measurement model. The machines and algorithm must have been initialized.
	 *
	 * @param[in] wvm The fast WVM.
	 * @param[in] svm The slower SVM.
	 * @param[in] oe The overlap elimination algorithm.
	 */
	explicit WvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> svm,
			shared_ptr<OverlapElimination> oe);

	~WvmSvmModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

private:

	/**
	 * Eliminates all but the ten best patches.
	 *
	 * @param[in] patches The patches.
	 * @param[in] detectorId The identifier of the detector used for computing the certainties.
	 * @return A new vector containing the remaining patches.
	 */
	std::vector<FdPatch*> eliminate(std::vector<FdPatch*> patches, std::string detectorId);

	shared_ptr<VDetectorVectorMachine> wvm; ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> svm; ///< The slower SVM.
	shared_ptr<OverlapElimination> oe;      ///< The overlap elimination algorithm.
};

} /* namespace tracking */
#endif /* WVMSVMMODEL_H_ */
