/*
 * DualClassifierModel.cpp
 *
 *  Created on: 03.07.2013
 *      Author: poschmann
 */

#include "DualClassifierModel.hpp"
#include "condensation/Sample.hpp"
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

DualClassifierModel::DualClassifierModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<ProbabilisticClassifier> classifier, shared_ptr<FeatureExtractor> filterFeatureExtractor, shared_ptr<ProbabilisticClassifier> filter) :
				featureExtractor(featureExtractor), classifier(classifier), filterFeatureExtractor(filterFeatureExtractor), filter(filter) {}

DualClassifierModel::~DualClassifierModel() {}

void DualClassifierModel::evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
	filterFeatureExtractor->update(image);
	featureExtractor->update(image);
	unordered_map<shared_ptr<Patch>, pair<bool, double>> results;
	unordered_map<shared_ptr<Patch>, pair<bool, double>> filterResults;
	for (auto sample = samples.begin(); sample != samples.end(); ++sample) {
		sample->setObject(false);
		shared_ptr<Patch> filterPatch = filterFeatureExtractor->extract(sample->getX(), sample->getY(), sample->getWidth(), sample->getHeight());
		if (!filterPatch) {
			sample->setWeight(0);
		} else {
			pair<bool, double> result;
			auto resIt = filterResults.find(filterPatch);
			if (resIt == filterResults.end()) {
				result = filter->classify(filterPatch->getData());
				filterResults.emplace(filterPatch, result);
			} else {
				result = resIt->second;
			}
			if (result.first) {
				shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getWidth(), sample->getHeight());

				if (!patch) {
					sample->setWeight(0);
				} else {
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
			} else {
				sample->setObject(false);
				sample->setWeight(0.5 * result.second);
			}
		}
	}
}

} /* namespace condensation */
