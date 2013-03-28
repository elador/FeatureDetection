/*
 * SingleClassifierModel.cpp
 *
 *  Created on: 28.03.2013
 *      Author: poschmann
 */

#include "condensation/SingleClassifierModel.hpp"
#include "condensation/Sample.h"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/ProbabilisticClassifier.hpp"
#include <unordered_map>
#include <utility>

using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;
using std::make_shared;
using std::unordered_map;
using std::pair;

namespace condensation {

SingleClassifierModel::SingleClassifierModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<ProbabilisticClassifier> classifier) :
				featureExtractor(featureExtractor), classifier(classifier) {}

SingleClassifierModel::~SingleClassifierModel() {}

void SingleClassifierModel::evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
	featureExtractor->update(image);
	unordered_map<shared_ptr<Patch>, pair<bool, double>> results;
	for (auto sample = samples.begin(); sample != samples.end(); ++sample) {
		sample->setObject(false);
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getSize(), sample->getSize());
		if (!patch) {
			sample->setWeight(0);
		} else {
			pair<bool, double> result;
			auto resIt = results.find(patch);
			if (resIt == results.end()) {
				result = classifier->classify(patch->getData());
				results.emplace(patch, result);
			} else {
				result = resIt->second;
			}
			sample->setObject(result.first);
			sample->setWeight(result.second);
		}
	}
}

} /* namespace condensation */
