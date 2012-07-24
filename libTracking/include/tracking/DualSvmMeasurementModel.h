/*
 * DualSvmMeasurementModel.h
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#ifndef DUALSVMMEASUREMENTMODEL_H_
#define DUALSVMMEASUREMENTMODEL_H_

#include "tracking/MeasurementModel.h"

class VDetectorVectorMachine;

namespace tracking {

/**
 * Measurement model for samples with two SVMs - one acts as a pre-stage for fast elimination,
 * the second one is a slower one for the few remaining samples.
 */
class DualSvmMeasurementModel : public MeasurementModel {
public:

	/**
	 * Constructs a new dual SVM measurement model.
	 *
	 * @param[in] preStage The (fast) pre-stage SVM.
	 * @param[in] mainStage The (slow) main SVM.
	 */
	DualSvmMeasurementModel(VDetectorVectorMachine* preStage, VDetectorVectorMachine* mainStage);
	virtual ~DualSvmMeasurementModel();

	void evaluate(FdImage* image, std::vector<Sample>& samples);

private:
	VDetectorVectorMachine* preStage;		///< The (fast) pre-stage SVM.
	VDetectorVectorMachine* mainStage;	///< The (slow) main SVM.
};

} /* namespace tracking */
#endif /* DUALSVMMEASUREMENTMODEL_H_ */
