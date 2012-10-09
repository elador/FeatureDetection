/*
 * WvmSvmModel.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "tracking/WvmSvmModel.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "tracking/Sample.h"
#include <map>

namespace tracking {

WvmSvmModel::WvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> svm,
		shared_ptr<OverlapElimination> oe) : wvm(wvm), svm(svm), oe(oe) {}

WvmSvmModel::~WvmSvmModel() {}

void WvmSvmModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
	wvm->initPyramids(image);
	wvm->initROI(image);
	std::vector<FdPatch*> remainingPatches;
	std::map<FdPatch*, std::vector<Sample*> > patch2samples;
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		sample.setObject(false);
		FdPatch* patch = wvm->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
		if (patch == 0) {
			sample.setWeight(0);
		} else {
			if (wvm->detect_on_patch(patch)) {
				remainingPatches.push_back(patch);
				patch2samples[patch].push_back(&sample);
			}
			sample.setWeight(0.5 * patch->certainty[wvm->getIdentifier()]);
		}
	}
	if (!remainingPatches.empty()) {
		//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
		remainingPatches = takeDistinctBest(remainingPatches, 10, wvm->getIdentifier());
		svm->initPyramids(image);
		svm->initROI(image);
		std::vector<FdPatch*> objectPatches = svm->detect_on_patchvec(remainingPatches);
		for (std::vector<FdPatch*>::iterator pit = objectPatches.begin(); pit < objectPatches.end(); ++pit) {
			std::vector<Sample*>& patchSamples = patch2samples[(*pit)];
			for (std::vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setObject(true);
			}
		}
		for (std::vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
			FdPatch* patch = (*pit);
			double certainty = patch->certainty[svm->getIdentifier()];
			std::vector<Sample*>& patchSamples = patch2samples[patch];
			for (std::vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setWeight(2 * sample->getWeight() * certainty);
			}
			patchSamples.clear();
		}
	}
}

} /* namespace tracking */
