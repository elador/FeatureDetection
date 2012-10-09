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
				positiveTrainingPatches(),
				negativeTrainingPatches() {}

SelfLearningWvmSvmModel::SelfLearningWvmSvmModel(std::string configFilename, std::string negativesFilename) :
		wvm(make_shared<DetectorWVM>()),
		staticSvm(make_shared<DetectorSVM>()),
		dynamicSvm(make_shared<ChangableDetectorSvm>()),
		oe(make_shared<OverlapElimination>()),
		svmTraining(boost::make_shared<FrameBasedSvmTraining>(5, 4, negativesFilename, 200)),
		usingDynamicSvm(false),
		positiveThreshold(0.85),
		negativeThreshold(0.4) {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	dynamicSvm->load(configFilename);
	oe->load(configFilename);
}

SelfLearningWvmSvmModel::~SelfLearningWvmSvmModel() {}

void SelfLearningWvmSvmModel::evaluate(FdImage* image, std::vector<Sample>& samples) {
//	if (usingDynamicSvm) { // TODO test of drift by not using wvm
//		positiveTrainingPatches.clear();
//		negativeTrainingPatches.clear();
//		dynamicSvm->initPyramids(image);
//		dynamicSvm->initROI(image);
//		for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
//			Sample& sample = *sit;
//			sample.setObject(false);
//			FdPatch* patch = dynamicSvm->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
//			if (patch == 0) {
//				sample.setWeight(0);
//			} else {
//				if (dynamicSvm->detect_on_patch(patch))
//					sample.setObject(true);
//				double certainty = patch->certainty[dynamicSvm->getIdentifier()];
//				if (certainty > positiveThreshold)
//					positiveTrainingPatches.push_back(patch);
//				else if (certainty < negativeThreshold)
//					negativeTrainingPatches.push_back(patch);
//				sample.setWeight(certainty);
//			}
//		}
//		positiveTrainingPatches = takeDistinctBest(positiveTrainingPatches, 10, dynamicSvm->getIdentifier());
//		negativeTrainingPatches = takeDistinctWorst(negativeTrainingPatches, 10, dynamicSvm->getIdentifier());
//	} else {
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
	positiveTrainingPatches.clear();
	negativeTrainingPatches.clear();
	if (!remainingPatches.empty()) {
		if (!usingDynamicSvm)
			//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
			remainingPatches = takeDistinctBest(remainingPatches, 10, wvm->getIdentifier());
		VDetectorVectorMachine& svm = usingDynamicSvm ? *dynamicSvm : *staticSvm;
		svm.initPyramids(image);
		svm.initROI(image);
		std::vector<FdPatch*> objectPatches = svm.detect_on_patchvec(remainingPatches);
		for (std::vector<FdPatch*>::iterator pit = objectPatches.begin(); pit < objectPatches.end(); ++pit) {
			std::vector<Sample*>& patchSamples = patch2samples[(*pit)];
			for (std::vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setObject(true);
			}
		}
		for (std::vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
			FdPatch* patch = (*pit);
			double certainty = patch->certainty[svm.getIdentifier()];
			if (certainty > positiveThreshold)
				positiveTrainingPatches.push_back(patch);
			else if (certainty < negativeThreshold)
				negativeTrainingPatches.push_back(patch);
			std::vector<Sample*>& patchSamples = patch2samples[patch];
			for (std::vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setWeight(2 * sample->getWeight() * certainty);
			}
			patchSamples.clear();
		}
		positiveTrainingPatches = takeDistinctBest(positiveTrainingPatches, 10, svm.getIdentifier());
		negativeTrainingPatches = takeDistinctWorst(negativeTrainingPatches, 10, svm.getIdentifier());
	}//}
}

void SelfLearningWvmSvmModel::reset() {
	svmTraining->reset(*dynamicSvm);
	usingDynamicSvm = false;
	positiveTrainingPatches.clear();
	negativeTrainingPatches.clear();
}

void SelfLearningWvmSvmModel::update() {
	usingDynamicSvm = svmTraining->retrain(*dynamicSvm, positiveTrainingPatches, negativeTrainingPatches);
	positiveTrainingPatches.clear();
	negativeTrainingPatches.clear();
}

void SelfLearningWvmSvmModel::update(FdImage* image, std::vector<Sample>& positiveSamples,
		std::vector<Sample>& negativeSamples) {
	addSamples(image, positiveTrainingPatches, positiveSamples);
	addSamples(image, negativeTrainingPatches, negativeSamples);
	update();
}

void SelfLearningWvmSvmModel::addSamples(FdImage* image, std::vector<FdPatch*>& patches, std::vector<Sample>& samples) {
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		FdPatch* patch = wvm->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
		if (patch != 0)
			patches.push_back(patch);
	}
}

} /* namespace tracking */
