/*
 * DualSvmModel.h
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#ifndef DUALSVMMODEL_H_
#define DUALSVMMODEL_H_

#include "tracking/MeasurementModel.h"

class VDetectorVectorMachine;

namespace tracking {

/**
 * Measurement model for samples with two SVMs - one acts as a pre-stage for fast elimination,
 * the second one is a slower one for the few remaining samples. Samples that are eliminated by
 * the first detector will get a weight of zero and the remaining samples will get a weight
 * according to the certainty of the second detector.
 */
class DualSvmModel : public MeasurementModel {
public:

	/**
	 * Constructs a new dual SVM measurement model.
	 *
	 * @param[in] preStage The (fast) pre-stage SVM. Will be deleted at destruction.
	 * @param[in] mainStage The (slow) main SVM. Will be deleted at destruction.
	 */
	DualSvmModel(VDetectorVectorMachine* preStage, VDetectorVectorMachine* mainStage);
	virtual ~DualSvmModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

private:
	VDetectorVectorMachine* preStage;  ///< The (fast) pre-stage SVM.
	VDetectorVectorMachine* mainStage; ///< The (slow) main SVM.
};

} /* namespace tracking */
#endif /* DUALSVMMODEL_H_ */
