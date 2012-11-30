/*
 * SelfLearningWvmSvmModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "tracking/SelfLearningWvmSvmModel.h"
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
#include <algorithm>

using boost::make_shared;
using boost::unordered_map;
using std::make_pair;

namespace tracking {

static bool compareFeatureProbabilityPairs(pair<shared_ptr<FeatureVector>, double> a,
		pair<shared_ptr<FeatureVector>, double> b) {
	return a.second < b.second;
}

SelfLearningWvmSvmModel::SelfLearningWvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<VDetectorVectorMachine> wvm, shared_ptr<VDetectorVectorMachine> staticSvm,
		shared_ptr<LibSvmClassifier> dynamicSvm, shared_ptr<OverlapElimination> oe, shared_ptr<LibSvmTraining> svmTraining,
		double positiveThreshold, double negativeThreshold) :
				featureExtractor(featureExtractor),
				wvm(wvm),
				staticSvm(staticSvm),
				dynamicSvm(dynamicSvm),
				oe(oe),
				svmTraining(svmTraining),
				useDynamicSvm(false),
				wasUsingDynamicSvm(false),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				positiveTrainingSamples(),
				negativeTrainingSamples() {}

SelfLearningWvmSvmModel::SelfLearningWvmSvmModel(string configFilename, string negativesFilename) :
		featureExtractor(make_shared<HistEqFeatureExtractor>()),
		wvm(make_shared<DetectorWVM>()),
		staticSvm(make_shared<DetectorSVM>()),
		dynamicSvm(make_shared<LibSvmClassifier>()),
		oe(make_shared<OverlapElimination>()),
		svmTraining(make_shared<FrameBasedSvmTraining>(5, 4, negativesFilename, 200)),
		useDynamicSvm(false),
		wasUsingDynamicSvm(false),
		positiveThreshold(0.85),
		negativeThreshold(0.4),
		positiveTrainingSamples(),
		negativeTrainingSamples() {
	wvm->load(configFilename);
	staticSvm->load(configFilename);
	oe->load(configFilename);
}

SelfLearningWvmSvmModel::~SelfLearningWvmSvmModel() {}

void SelfLearningWvmSvmModel::evaluate(Mat& image, vector<Sample>& samples) {
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
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
				if (result.second > positiveThreshold)
					positiveTrainingSamples.push_back(make_pair(featureVector, result.second));
				else if (result.second < negativeThreshold)
					negativeTrainingSamples.push_back(make_pair(featureVector, result.second));
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
				if (certainty > positiveThreshold) {
					Sample& sample = *patch2samples[(*pit)].front();
					shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sample.getX(), sample.getY(), sample.getSize());
					if (featureVector)
						positiveTrainingSamples.push_back(make_pair(featureVector, certainty));
				} else if (certainty < negativeThreshold) {
					Sample& sample = *patch2samples[(*pit)].front();
					shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sample.getX(), sample.getY(), sample.getSize());
					if (featureVector)
						negativeTrainingSamples.push_back(make_pair(featureVector, certainty));
				}
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
	std::sort(positiveTrainingSamples.begin(), positiveTrainingSamples.end(), compareFeatureProbabilityPairs);
	std::sort(negativeTrainingSamples.begin(), negativeTrainingSamples.end(), compareFeatureProbabilityPairs);
	std::reverse(negativeTrainingSamples.begin(), negativeTrainingSamples.end());
	if (positiveTrainingSamples.size() > 10)
		positiveTrainingSamples.resize(10);
	if (negativeTrainingSamples.size() > 10)
		negativeTrainingSamples.resize(10);
	wasUsingDynamicSvm = useDynamicSvm;
}

void SelfLearningWvmSvmModel::reset() {
	svmTraining->reset(*dynamicSvm);
	useDynamicSvm = false;
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
}

void SelfLearningWvmSvmModel::update() {
	vector<shared_ptr<FeatureVector> > positiveSamples;
	positiveSamples.reserve(positiveTrainingSamples.size());
	vector<pair<shared_ptr<FeatureVector>, double> >::iterator fvpit;
	for (fvpit = positiveTrainingSamples.begin(); fvpit != positiveTrainingSamples.end(); ++fvpit)
		positiveSamples.push_back(fvpit->first);

	vector<shared_ptr<FeatureVector> > negativeSamples;
	negativeSamples.reserve(negativeTrainingSamples.size());
	for (fvpit = negativeTrainingSamples.begin(); fvpit != negativeTrainingSamples.end(); ++fvpit)
		negativeSamples.push_back(fvpit->first);

	useDynamicSvm = svmTraining->retrain(*dynamicSvm, positiveSamples, negativeSamples);
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
}

void SelfLearningWvmSvmModel::update(vector<Sample>& additionalPositiveSamples,
		vector<Sample>& additionalNegativeSamples) {
	vector<shared_ptr<FeatureVector> > positiveSamples;
	positiveSamples.reserve(positiveTrainingSamples.size() + additionalPositiveSamples.size());
	addTrainingSamples(positiveSamples, positiveTrainingSamples);
	addTrainingSamples(positiveSamples, additionalPositiveSamples);

	vector<shared_ptr<FeatureVector> > negativeSamples;
	negativeSamples.reserve(negativeTrainingSamples.size() + additionalNegativeSamples.size());
	addTrainingSamples(negativeSamples, negativeTrainingSamples);
	addTrainingSamples(negativeSamples, additionalNegativeSamples);

	useDynamicSvm = svmTraining->retrain(*dynamicSvm, positiveSamples, negativeSamples);
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
}

void SelfLearningWvmSvmModel::addTrainingSamples(vector<shared_ptr<FeatureVector> >& trainingSamples,
		vector<pair<shared_ptr<FeatureVector>, double> >& pairs) {
	for (vector<pair<shared_ptr<FeatureVector>, double> >::iterator fvpit = pairs.begin(); fvpit != pairs.end(); ++fvpit)
		trainingSamples.push_back(fvpit->first);
}

void SelfLearningWvmSvmModel::addTrainingSamples(vector<shared_ptr<FeatureVector> >& trainingSamples,
		vector<Sample>& samples) {
	for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sit->getX(), sit->getY(), sit->getSize());
		if (featureVector)
			trainingSamples.push_back(featureVector);
	}
}

} /* namespace tracking */
