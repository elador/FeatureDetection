/*
 * HeadWvmSvmModel.h
 *
 *  Created on: 13.11.2012
 *      Author: poschmann
 */

#ifndef HEADWVMSVMMODEL_H_
#define HEADWVMSVMMODEL_H_

#include "tracking/MeasurementModel.h"
#include "tracking/PatchDuplicateFilter.h"
#include <memory>
#include <string>

class VDetectorVectorMachine;
class OverlapElimination;

using tracking::MeasurementModel;
using tracking::PatchDuplicateFilter;
using tracking::Sample;
using std::shared_ptr;

/**
 * Measurement model that uses a WVM for quick elimination and evaluates the samples that remain after an
 * overlap elimination with a SVM. The weight of the samples will be the product of the certainties from
 * the two detectors, they will be regarded as being independent (although they are not). The certainties
 * for the SVM of samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class HeadWvmSvmModel : public MeasurementModel, public PatchDuplicateFilter {
public:

	/**
	 * Constructs a new WVM SVM measurement model. The machines and algorithm must have been initialized.
	 *
	 * @param[in] wvm The fast WVM.
	 * @param[in] svm The slower SVM.
	 * @param[in] oe The overlap elimination algorithm.
	 */
	explicit HeadWvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> svm,
			shared_ptr<OverlapElimination> oe);

	~HeadWvmSvmModel();

	void evaluate(const Mat& image, vector<Sample>& samples);

private:

	shared_ptr<VDetectorVectorMachine> wvm; ///< The fast WVM.
	shared_ptr<VDetectorVectorMachine> svm; ///< The slower SVM.
	shared_ptr<OverlapElimination> oe;      ///< The overlap elimination algorithm.
};

#endif /* HEADWVMSVMMODEL_H_ */
