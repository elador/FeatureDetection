/*
 * WvmOeSvmModel.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "tracking/WvmOeSvmModel.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "tracking/Sample.h"
#include <map>

namespace tracking {

WvmOeSvmModel::WvmOeSvmModel(shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> svm,
		shared_ptr<OverlapElimination> oe) : wvm(wvm), svm(svm), oe(oe) {}

WvmOeSvmModel::~WvmOeSvmModel() {}

void WvmOeSvmModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
	wvm->initPyramids(image);
	wvm->initROI(image);
	std::vector<FdPatch*> remainingPatches;
	std::map<FdPatch*, Sample*> patch2sample;
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		sample.setObject(false);
		FdPatch* patch = wvm->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
		if (patch == 0) {
			sample.setWeight(0);
		} else {
			if (wvm->detect_on_patch(patch)) {
				remainingPatches.push_back(patch);
				patch2sample[patch] = &sample;
			}
			sample.setWeight(0.5 * patch->certainty[wvm->getIdentifier()]);
		}
	}
	if (!remainingPatches.empty()) {
		//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
		remainingPatches = eliminate(remainingPatches, wvm->getIdentifier());
		svm->initPyramids(image);
		svm->initROI(image);
		std::vector<FdPatch*> objectPatches = svm->detect_on_patchvec(remainingPatches);
		for (std::vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
			FdPatch* patch = (*pit);
			Sample* sample = patch2sample[patch];
			sample->setWeight(2 * sample->getWeight() * patch->certainty[svm->getIdentifier()]);
		}
		for (std::vector<FdPatch*>::iterator pit = objectPatches.begin(); pit < objectPatches.end(); ++pit) {
			Sample* sample = patch2sample[(*pit)];
			sample->setObject(true);
		}
	}
}

std::vector<FdPatch*> WvmOeSvmModel::eliminate(std::vector<FdPatch*> patches, std::string detectorId) {
	if (patches.size() > 10) {
		std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
		patches.resize(10);
	}
	return patches;
}

} /* namespace tracking */
