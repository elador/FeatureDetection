/*
 * LearningWvmSvmModel.cpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#include "tracking/LearningWvmSvmModel.h"
#include "tracking/Sample.h"
#include "classification/FeatureVector.h"
#include "classification/FeatureExtractor.h"
#include "classification/HistEqFeatureExtractor.h"
#include "classification/LibSvmClassifier.h"
#include "classification/LibSvmTraining.h"
#include "classification/FrameBasedSvmTraining.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "FdImage.h"
#include "FdPatch.h"
#include "boost/make_shared.hpp"
#include "boost/unordered_map.hpp"
#include <utility>

using boost::make_shared;
using boost::unordered_map;
using std::pair;

namespace tracking {

LearningWvmSvmModel::LearningWvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> staticSvm,
		shared_ptr<LibSvmClassifier> dynamicSvm, shared_ptr<OverlapElimination> oe,
		shared_ptr<LibSvmTraining> svmTraining) :
				featureExtractor(featureExtractor),
				wvm(wvm),
				staticSvm(staticSvm),
				dynamicSvm(dynamicSvm),
				oe(oe),
				svmTraining(svmTraining),
				useDynamicSvm(false),
				wasUsingDynamicSvm(false) {}

LearningWvmSvmModel::LearningWvmSvmModel(string configFilename, string negativesFilename) :
		featureExtractor(make_shared<HistEqFeatureExtractor>()),
		wvm(make_shared<DetectorWVM>()),
		staticSvm(make_shared<DetectorSVM>()),
		dynamicSvm(make_shared<LibSvmClassifier>()),
		oe(make_shared<OverlapElimination>()),
		svmTraining(make_shared<FrameBasedSvmTraining>(5, 4, negativesFilename, 200)),
		useDynamicSvm(false),
		wasUsingDynamicSvm(false) {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	oe->load(configFilename);
}

LearningWvmSvmModel::~LearningWvmSvmModel() {}

void LearningWvmSvmModel::evaluate(Mat& image, vector<Sample>& samples) {
	featureExtractor->init(image);
	if (useDynamicSvm) {
		unordered_map<shared_ptr<FeatureVector>, pair<bool, double> > results;
		for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			sit->setObject(false);
			shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sit->getX(), sit->getY(), sit->getSize());
			if (!featureVector) {
				sit->setWeight(0);
			} else {
				pair<bool, double> result;
				unordered_map<shared_ptr<FeatureVector>, pair<bool, double> >::iterator rit = results.find(featureVector);
				if (rit == results.end()) {
					result = dynamicSvm->classify(*featureVector);
					pair<const shared_ptr<FeatureVector>, pair<bool, double> > entry = make_pair(featureVector, result);
					results.insert(entry);
				} else {
					result = rit->second;
				}
				sit->setObject(result.first);
				sit->setWeight(result.second);
			}
		}
	} else {
		FdImage* fdImage = new FdImage();
		fdImage->load(&image);
		wvm->initPyramids(fdImage);
		wvm->initROI(fdImage);
		vector<FdPatch*> remainingPatches;
		unordered_map<FdPatch*, vector<Sample*> > patch2samples;
		for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			Sample& sample = *sit;
			sample.setObject(false);
			FdPatch* patch = wvm->extractPatchToPyramid(fdImage, sample.getX(), sample.getY(), sample.getSize());
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
			//remainingPatches = oe->eliminate(remainingPatches, wvm->getIdentifier());
			remainingPatches = takeDistinctBest(remainingPatches, 10, wvm->getIdentifier());
			staticSvm->initPyramids(fdImage);
			staticSvm->initROI(fdImage);
			vector<FdPatch*> objectPatches = staticSvm->detectOnPatchvec(remainingPatches);
			for (vector<FdPatch*>::iterator pit = objectPatches.begin(); pit < objectPatches.end(); ++pit) {
				vector<Sample*>& patchSamples = patch2samples[(*pit)];
				for (vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
					Sample* sample = (*sit);
					sample->setObject(true);
				}
			}
			for (vector<FdPatch*>::iterator pit = remainingPatches.begin(); pit < remainingPatches.end(); ++pit) {
				FdPatch* patch = (*pit);
				double certainty = patch->certainty[staticSvm->getIdentifier()];
				vector<Sample*>& patchSamples = patch2samples[patch];
				for (vector<Sample*>::iterator sit = patchSamples.begin(); sit < patchSamples.end(); ++sit) {
					Sample* sample = (*sit);
					sample->setWeight(2 * sample->getWeight() * certainty);
				}
				patchSamples.clear();
			}
		}
		delete fdImage;
	}
	wasUsingDynamicSvm = useDynamicSvm;
}

void LearningWvmSvmModel::reset() {
	svmTraining->reset(*dynamicSvm);
	useDynamicSvm = false;
}

void LearningWvmSvmModel::update() {
	const vector<shared_ptr<FeatureVector> > empty;
	useDynamicSvm = svmTraining->retrain(*dynamicSvm, empty, empty);
}

void LearningWvmSvmModel::update(vector<Sample>& positiveSamples, vector<Sample>& negativeSamples) {
	useDynamicSvm = svmTraining->retrain(*dynamicSvm,
			getTrainingSamples(positiveSamples), getTrainingSamples(negativeSamples));
}

vector<shared_ptr<FeatureVector> > LearningWvmSvmModel::getTrainingSamples(vector<Sample>& samples) {
	vector<shared_ptr<FeatureVector> > trainingSamples;
	trainingSamples.reserve(samples.size());
	for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sit->getX(), sit->getY(), sit->getSize());
		if (featureVector)
			trainingSamples.push_back(featureVector);
	}
	return trainingSamples;
}

} /* namespace tracking */
