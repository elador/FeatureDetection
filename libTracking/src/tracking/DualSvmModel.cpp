/*
 * DualSvmModel.cpp
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#include "tracking/DualSvmModel.h"
#include "tracking/Sample.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"

namespace tracking {

DualSvmModel::DualSvmModel(shared_ptr<VDetectorVectorMachine> preStage, shared_ptr<VDetectorVectorMachine> mainStage)
		: preStage(preStage), mainStage(mainStage) {}

DualSvmModel::~DualSvmModel() {}

void DualSvmModel::evaluate(Mat& image, vector<Sample>& samples) {
	FdImage* fdImage = new FdImage();
	fdImage->load(&image);
	preStage->initPyramids(fdImage);
	preStage->initROI(fdImage);
	mainStage->initPyramids(fdImage);
	mainStage->initROI(fdImage);
	for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		sample.setWeight(0);
		sample.setObject(false);
		FdPatch* patch = preStage->extractPatchToPyramid(fdImage, sample.getX(), sample.getY(), sample.getSize());
		if (patch != 0 && preStage->detectOnPatch(patch)) {
			patch = mainStage->extractPatchToPyramid(fdImage, sample.getX(), sample.getY(), sample.getSize());
			if (patch != 0) {
				if (mainStage->detectOnPatch(patch))
					sample.setObject(true);
				sample.setWeight(patch->certainty[mainStage->getIdentifier()]);
			}
		}
	}
	delete fdImage;
}

} /* namespace tracking */
