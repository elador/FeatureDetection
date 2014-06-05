/*
 * WvmSvmModel.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "condensation/WvmSvmModel.hpp"
#include "condensation/Sample.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <utility>
#include <functional>

using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::FeatureExtractor;
using classification::ProbabilisticWvmClassifier;
using classification::ProbabilisticSvmClassifier;
using detection::ClassifiedPatch;
using boost::make_indirect_iterator;
using std::pair;
using std::vector;
using std::greater;
using std::shared_ptr;
using std::make_shared;
using std::unordered_map;

namespace condensation {

WvmSvmModel::WvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<ProbabilisticWvmClassifier> wvm, shared_ptr<ProbabilisticSvmClassifier> svm) :
		featureExtractor(featureExtractor), wvm(wvm), svm(svm), cache() {}

void WvmSvmModel::update(shared_ptr<VersionedImage> image) {
	cache.clear();
	featureExtractor->update(image);
}

void WvmSvmModel::evaluate(Sample& sample) const {
	shared_ptr<Patch> patch = featureExtractor->extract(sample.getX(), sample.getY(), sample.getWidth(), sample.getHeight());
	if (!patch) {
		sample.setTarget(false);
		sample.setWeight(0);
	} else {
		pair<bool, double> wvmResult;
		auto resIt = cache.find(patch);
		if (resIt == cache.end()) {
			wvmResult = wvm->getProbability(patch->getData());
			cache.emplace(patch, wvmResult);
		} else {
			wvmResult = resIt->second;
		}
		if (wvmResult.first) {
			pair<bool, double> svmResult = svm->getProbability(patch->getData());
			sample.setTarget(svmResult.first);
			sample.setWeight(wvmResult.second * svmResult.second);
		} else {
			sample.setTarget(false);
			sample.setWeight(0.5 * wvmResult.second);
		}
	}
}

void WvmSvmModel::evaluate(shared_ptr<VersionedImage> image, vector<shared_ptr<Sample>>& samples) {
	update(image);
	vector<shared_ptr<ClassifiedPatch>> remainingPatches;
	unordered_map<shared_ptr<Patch>, vector<Sample*>> patch2samples;
	for (shared_ptr<Sample> sample : samples) {
		sample->setTarget(false);
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getWidth(), sample->getHeight());
		if (!patch) {
			sample->setWeight(0);
		} else {
			pair<bool, double> result;
			auto resIt = cache.find(patch);
			if (resIt == cache.end()) {
				result = wvm->getProbability(patch->getData());
				if (result.first)
					remainingPatches.push_back(make_shared<ClassifiedPatch>(patch, result));
				cache.emplace(patch, result);
			} else {
				result = resIt->second;
			}
			if (result.first)
				patch2samples[patch].push_back(&(*sample));
			sample->setWeight(0.5 * result.second);
		}
	}
	if (!remainingPatches.empty()) {
		// TODO overlap elimination instead?
		if (remainingPatches.size() > 8) {
			sort(make_indirect_iterator(remainingPatches.begin()), make_indirect_iterator(remainingPatches.end()), greater<ClassifiedPatch>());
			remainingPatches.resize(8);
		}
		for (auto patchWithProb = remainingPatches.cbegin(); patchWithProb != remainingPatches.cend(); ++patchWithProb) {
			shared_ptr<Patch> patch = (*patchWithProb)->getPatch();
			pair<bool, double> result = svm->getProbability(patch->getData());
			vector<Sample*>& patchSamples = patch2samples[patch];
			for (auto sit = patchSamples.begin(); sit != patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setTarget(result.first);
				sample->setWeight(2 * sample->getWeight() * result.second);
			}
			patchSamples.clear();
		}
	}
}

} /* namespace condensation */
