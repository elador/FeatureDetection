/*
 * HeadLearningWvmSvmModel.cpp
 *
 *  Created on: 13.11.2012
 *      Author: poschmann
 */

#include "HeadLearningWvmSvmModel.h"
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
using namespace tracking;

HeadLearningWvmSvmModel::HeadLearningWvmSvmModel(shared_ptr<VDetectorVectorMachine> wvm,
		shared_ptr<VDetectorVectorMachine> staticSvm, shared_ptr<ChangableDetectorSvm> dynamicSvm,
		shared_ptr<OverlapElimination> oe, shared_ptr<SvmTraining> svmTraining) :
				wvm(wvm),
				staticSvm(staticSvm),
				dynamicSvm(dynamicSvm),
				oe(oe),
				svmTraining(svmTraining),
				useDynamicSvm(false),
				wasUsingDynamicSvm(false),
				fdImage(new FdImage()) {}

HeadLearningWvmSvmModel::HeadLearningWvmSvmModel(std::string configFilename, std::string negativesFilename) :
		wvm(make_shared<DetectorWVM>()),
		staticSvm(make_shared<DetectorSVM>()),
		dynamicSvm(make_shared<ChangableDetectorSvm>()),
		oe(make_shared<OverlapElimination>()),
		svmTraining(boost::make_shared<FrameBasedSvmTraining>(5, 4, negativesFilename, 200)),
		useDynamicSvm(false),
		wasUsingDynamicSvm(false),
		fdImage(new FdImage()) {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	dynamicSvm->load(configFilename);
	oe->load(configFilename);
}

HeadLearningWvmSvmModel::~HeadLearningWvmSvmModel() {
	delete fdImage;
}

void HeadLearningWvmSvmModel::evaluate(cv::Mat& image, std::vector<Sample>& samples) {
	delete fdImage;
	fdImage = new FdImage();
	fdImage->load(&image);
	if (useDynamicSvm) {
		dynamicSvm->initPyramids(fdImage);
		dynamicSvm->initROI(fdImage);
		for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			Sample& sample = *sit;
			sample.setObject(false);
			FdPatch* patch = dynamicSvm->extractPatchToPyramid(fdImage, sample.getX(), sample.getY(), sample.getSize());
			if (patch == 0) {
				sample.setWeight(0);
			} else {
				if (dynamicSvm->detectOnPatch(patch))
					sample.setObject(true);
				sample.setWeight(patch->certainty[dynamicSvm->getIdentifier()]);
			}
		}
	} else {
		wvm->initPyramids(fdImage);
		wvm->initROI(fdImage);
		std::vector<FdPatch*> remainingPatches;
		std::map<FdPatch*, std::vector<Sample*> > patch2samples;
		for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			Sample& sample = *sit;
			sample.setObject(false);
			int faceX = sample.getX();
			int faceY = sample.getY() + (int)(0.1 * sample.getSize());
			int faceSize = (int)(0.6 * sample.getSize());
			FdPatch* patch = wvm->extractPatchToPyramid(fdImage, faceX, faceY, faceSize);
			if (patch == 0) {
				sample.setWeight(0);
			} else {
				if (wvm->detectOnPatch(patch)) {
					remainingPatches.push_back(patch);
					patch2samples[patch].push_back(&sample);
				}
				sample.setWeight(0.5 * patch->certainty[wvm->getIdentifier()]);
			}
		}
		if (!remainingPatches.empty()) {
			if (!useDynamicSvm)
				//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
				remainingPatches = takeDistinctBest(remainingPatches, 10, wvm->getIdentifier());
			VDetectorVectorMachine& svm = useDynamicSvm ? *dynamicSvm : *staticSvm;
			svm.initPyramids(fdImage);
			svm.initROI(fdImage);
			std::vector<FdPatch*> objectPatches = svm.detectOnPatchvec(remainingPatches);
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
				std::vector<Sample*>& patchSamples = patch2samples[patch];
				for (std::vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
					Sample* sample = (*sit);
					sample->setWeight(2 * sample->getWeight() * certainty);
				}
				patchSamples.clear();
			}
		}
	}
	wasUsingDynamicSvm = useDynamicSvm;
}

void HeadLearningWvmSvmModel::reset() {
	svmTraining->reset(*dynamicSvm);
	useDynamicSvm = false;
}

void HeadLearningWvmSvmModel::update() {
	const std::vector<FdPatch*> empty;
	useDynamicSvm = svmTraining->retrain(*dynamicSvm, empty, empty);
}

void HeadLearningWvmSvmModel::update(cv::Mat& image, std::vector<Sample>& positiveSamples,
		std::vector<Sample>& negativeSamples) {
	useDynamicSvm = svmTraining->retrain(*dynamicSvm,
			getPatches(fdImage, positiveSamples), getPatches(fdImage, negativeSamples));
}

std::vector<FdPatch*> HeadLearningWvmSvmModel::getPatches(FdImage* image, std::vector<Sample>& samples) {
	std::vector<FdPatch*> patches;
	patches.reserve(samples.size());
	for (std::vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		Sample& sample = *sit;
		FdPatch* patch = wvm->extractPatchToPyramid(image, sample.getX(), sample.getY(), sample.getSize());
		if (patch != 0)
			patches.push_back(patch);
	}
	return patches;
}
