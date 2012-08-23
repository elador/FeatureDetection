/*
 * SelfLearningWvmSvmModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "tracking/SelfLearningWvmSvmModel.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "tracking/ChangableDetectorSvm.h"
#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/Sample.h"
#include "boost/make_shared.hpp"
#include <map>

using boost::make_shared;

namespace tracking {

SelfLearningWvmSvmModel::SelfLearningWvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm,
		shared_ptr<VDetectorVectorMachine> staticSvm, shared_ptr<ChangableDetectorSvm> dynamicSvm,
		shared_ptr<OverlapElimination> oe, shared_ptr<SvmTraining> svmTraining,
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

SelfLearningWvmSvmModel::SelfLearningWvmSvmModel(std::string configFilename, std::string negativesFilename) :
		wvm(make_shared<DetectorWVM>()),
		staticSvm(make_shared<DetectorSVM>()),
		dynamicSvm(make_shared<ChangableDetectorSvm>()),
		oe(make_shared<OverlapElimination>()),
		svmTraining(make_shared<FrameBasedSvmTraining>(5, 4, negativesFilename, 200)),
		usingDynamicSvm(false),
		positiveThreshold(0.85),
		negativeThreshold(0.4),
		selfLearningActive(true) {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	dynamicSvm->load(configFilename);
	oe->load(configFilename);
}

SelfLearningWvmSvmModel::~SelfLearningWvmSvmModel() {}

void SelfLearningWvmSvmModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
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
		bool useDynamicSvm = selfLearningActive && usingDynamicSvm;
		if (!useDynamicSvm)
			//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
			remainingPatches = eliminate(remainingPatches, wvm->getIdentifier());
		VDetectorVectorMachine& svm = useDynamicSvm ? *dynamicSvm : *staticSvm;
		svm.initPyramids(image);
		svm.initROI(image);
		std::vector<FdPatch*> objectPatches = svm.detect_on_patchvec(remainingPatches);
		for (std::vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
			FdPatch* patch = (*pit);
			Sample* sample = patch2sample[patch];
			double certainty = patch->certainty[svm.getIdentifier()];
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
		positiveTrainingPatches = takeDistinctBest(positiveTrainingPatches, 10, svm.getIdentifier());
		negativeTrainingPatches = takeDistinctWorst(negativeTrainingPatches, 10, svm.getIdentifier());
	}
	usingDynamicSvm = svmTraining->retrain(*dynamicSvm, positiveTrainingPatches, negativeTrainingPatches);
}

std::vector<FdPatch*> SelfLearningWvmSvmModel::eliminate(std::vector<FdPatch*> patches, std::string detectorId) {
	if (patches.size() > 10) {
		std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
		patches.resize(10);
	}
	return patches;
}

std::vector<FdPatch*> SelfLearningWvmSvmModel::takeDistinctBest(std::vector<FdPatch*> patches,
		unsigned int count, std::string detectorId) {
	std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
	return takeDistinct(patches, count);
}

std::vector<FdPatch*> SelfLearningWvmSvmModel::takeDistinctWorst(std::vector<FdPatch*> patches,
		unsigned int count, std::string detectorId) {
	std::sort(patches.begin(), patches.end(), FdPatch::SortByCertainty(detectorId));
	std::reverse(patches.begin(), patches.end());
	return takeDistinct(patches, count);
}

std::vector<FdPatch*> SelfLearningWvmSvmModel::takeDistinct(const std::vector<FdPatch*>& patches,
		unsigned int count) {
	if (patches.empty())
		return patches;
	std::vector<FdPatch*> remainingPatches;
	remainingPatches.reserve(count);
	std::vector<FdPatch*>::const_iterator pit = patches.begin();
	remainingPatches.push_back(*pit);
	pit++;
	for (; remainingPatches.size() < count && pit < patches.end(); ++pit) {
		if (*pit != remainingPatches.back())
			remainingPatches.push_back(*pit);
	}
	return remainingPatches;
}

} /* namespace tracking */
