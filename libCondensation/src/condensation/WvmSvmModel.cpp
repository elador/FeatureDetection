/*
 * WvmSvmModel.cpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#include "condensation/WvmSvmModel.h"
#include "condensation/Sample.h"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <unordered_map>
#include <utility>
#include <functional>

using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;
using detection::ClassifiedPatch;
using boost::make_indirect_iterator;
using std::unordered_map;
using std::pair;
using std::make_shared;
using std::greater;

namespace condensation {

WvmSvmModel::WvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<ProbabilisticWvmClassifier> wvm, shared_ptr<ProbabilisticSvmClassifier> svm) :
		featureExtractor(featureExtractor), wvm(wvm), svm(svm) {}

WvmSvmModel::~WvmSvmModel() {}

void WvmSvmModel::evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
	featureExtractor->update(image);
	// TODO das folgende macht nur dann sinn, wenn featureExtractor bereits duplikate erkennt und schonmal extrahiertes rausgibt
	// TODO oder eigene hash-methode bereitstellen, die auf patch arbeitet (x, y, w, h) -> duplikate werden erst hier erkannt
	// TODO ClassifiedPatch?
	unordered_map<shared_ptr<Patch>, pair<bool, double>> results;
	vector<shared_ptr<ClassifiedPatch>> remainingPatches;
	unordered_map<shared_ptr<Patch>, vector<Sample*>> patch2samples;
	for (auto sample = samples.begin(); sample != samples.end(); ++sample) {
		sample->setObject(false);
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getSize(), sample->getSize());
		if (!patch) {
			sample->setWeight(0);
		} else {
			pair<bool, double> result;
			auto resIt = results.find(patch);
			if (resIt == results.end()) {
				// TODO iimg-filter anwenden?
				result = wvm->classify(patch->getData());
				if (result.first)
					remainingPatches.push_back(make_shared<ClassifiedPatch>(patch, result));
				results.insert(make_pair(patch, result));
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
		if (remainingPatches.size() > 10) {
			sort(make_indirect_iterator(remainingPatches.begin()), make_indirect_iterator(remainingPatches.end()), greater<ClassifiedPatch>());
			remainingPatches.resize(10);
		}
		for (auto patchWithProb = remainingPatches.cbegin(); patchWithProb != remainingPatches.cend(); ++patchWithProb) {
			shared_ptr<Patch> patch = (*patchWithProb)->getPatch();
			pair<bool, double> result = svm->classify(patch->getData());
			vector<Sample*>& patchSamples = patch2samples[patch];
			for (auto sit = patchSamples.begin(); sit != patchSamples.end(); ++sit) {
				Sample* sample = (*sit);
				sample->setObject(result.first);
				sample->setWeight(2 * sample->getWeight() * result.second);
			}
			patchSamples.clear();
		}
	}
}

} /* namespace condensation */
