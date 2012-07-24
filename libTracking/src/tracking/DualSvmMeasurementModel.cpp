/*
 * DualSvmMeasurementModel.cpp
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#include "tracking/DualSvmMeasurementModel.h"

#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "tracking/Sample.h"
#include <iostream>

namespace tracking {

DualSvmMeasurementModel::DualSvmMeasurementModel(VDetectorVectorMachine* preStage, VDetectorVectorMachine* mainStage)
		: preStage(preStage), mainStage(mainStage) {}

DualSvmMeasurementModel::~DualSvmMeasurementModel() {
	delete preStage;
	delete mainStage;
}

void DualSvmMeasurementModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
	preStage->initPyramids(image);
	mainStage->initPyramids(image);
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		double weight = 0.0;
		FdPatch* patch = preStage->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
		if (patch != 0) {
			if (preStage->detect_on_patch(patch)) {
				patch = mainStage->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
				if (patch == 0) {
					weight = patch->certainty[preStage->getIdentifier()];
				} else {
					mainStage->detect_on_patch(patch);
					weight = patch->certainty[mainStage->getIdentifier()];
				}
			} else {
				weight = patch->certainty[preStage->getIdentifier()];
			}
		}
		sample.setWeight(weight);
	}
}

} /* namespace tracking */
