/*
 * SelfLearningWvmOeSvmModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "tracking/SelfLearningWvmOeSvmModel.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "tracking/ChangableDetectorSvm.h"
#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/Sample.h"
#include <map>

namespace tracking {

SelfLearningWvmOeSvmModel::SelfLearningWvmOeSvmModel(VDetectorVectorMachine* wvm, VDetectorVectorMachine* staticSvm,
		ChangableDetectorSvm* dynamicSvm, OverlapElimination* oe, SvmTraining* svmTraining,
		double positiveThreshold, double negativeThreshold) :
				wvm(wvm),
				staticSvm(staticSvm),
				dynamicSvm(dynamicSvm),
				oe(oe),
				svmTraining(svmTraining),
				usingDynamicSvm(false),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				selfLearningActive(true) {}

SelfLearningWvmOeSvmModel::SelfLearningWvmOeSvmModel(std::string configFilename, std::string negativesFilename) :
		wvm(new DetectorWVM()),
		staticSvm(new DetectorSVM()),
		dynamicSvm(new ChangableDetectorSvm()),
		oe(new OverlapElimination()),
		svmTraining(new FrameBasedSvmTraining(5, 4, negativesFilename, 200)),
		usingDynamicSvm(false),
		positiveThreshold(0.85),
		negativeThreshold(0.4),
		selfLearningActive(true) {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	dynamicSvm->load(configFilename);
	oe->load(configFilename);
}

SelfLearningWvmOeSvmModel::~SelfLearningWvmOeSvmModel() {
	delete wvm;
	delete staticSvm;
	delete dynamicSvm;
	delete oe;
}

void SelfLearningWvmOeSvmModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
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
	std::vector<FdPatch*> positiveTrainingPatches;
	std::vector<FdPatch*> negativeTrainingPatches;
	if (!remainingPatches.empty()) {
		//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
		remainingPatches = eliminate(remainingPatches, wvm->getIdentifier());
		VDetectorVectorMachine* svm;
		if (selfLearningActive && usingDynamicSvm)
			svm = dynamicSvm;
		else
			svm = staticSvm;
		svm->initPyramids(image);
		svm->initROI(image);
		std::vector<FdPatch*> objectPatches = svm->detect_on_patchvec(remainingPatches);
		for (std::vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
			FdPatch* patch = (*pit);
			Sample* sample = patch2sample[patch];
			double certainty = patch->certainty[svm->getIdentifier()];
			sample->setWeight(2 * sample->getWeight() * certainty);
			if (selfLearningActive) {
				if (certainty > positiveThreshold)
					positiveTrainingPatches.push_back(patch);
				else if (certainty < negativeThreshold)
					negativeTrainingPatches.push_back(patch);
			}
		}
		for (std::vector<FdPatch*>::iterator pit = objectPatches.begin(); pit < objectPatches.end(); ++pit) {
			Sample* sample = patch2sample[(*pit)];
			sample->setObject(true);
		}
	}
	usingDynamicSvm = svmTraining->retrain(dynamicSvm, positiveTrainingPatches, negativeTrainingPatches);
}

std::vector<FdPatch*> SelfLearningWvmOeSvmModel::eliminate(const std::vector<FdPatch*>& patches,
		std::string detectorId) {
	std::vector<FdPatch*> remaining = patches;
	if (remaining.size() > 10) {
		std::sort(remaining.begin(), remaining.end(), FdPatch::SortByCertainty(detectorId));
		remaining.resize(10);
	}
	return remaining;
}

} /* namespace tracking */
