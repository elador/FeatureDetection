/*
 * ConfidenceBasedExampleManagement.cpp
 *
 *  Created on: 25.11.2013
 *      Author: poschmann
 */

#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/BinaryClassifier.hpp"
#include <utility>

using cv::Mat;
using std::pair;
using std::make_pair;
using std::vector;
using std::shared_ptr;

namespace classification {

ConfidenceBasedExampleManagement::ConfidenceBasedExampleManagement(
		const shared_ptr<BinaryClassifier>& classifier, bool positive, size_t capacity, size_t requiredSize) :
				VectorBasedExampleManagement(capacity, requiredSize), classifier(classifier), positive(positive), keep(1) {}

void ConfidenceBasedExampleManagement::setFirstExamplesToKeep(size_t keep) {
	this->keep = keep;
}

void ConfidenceBasedExampleManagement::add(const vector<Mat>& newExamples) {
	// compute confidences of existing and new training examples and sort
	vector<pair<size_t, double>> existingConfidences;
	existingConfidences.reserve(examples.size());
	for (size_t i = keep; i < examples.size(); ++i) {
		pair<bool, double> result = classifier->getConfidence(examples[i]);
		double score = result.second;
		if (positive ^ result.first)
			score = -score;
		existingConfidences.push_back(make_pair(i, score));
	}
	vector<pair<size_t, double>> newConfidences;
	newConfidences.reserve(newExamples.size());
	for (size_t i = 0; i < newExamples.size(); ++i) {
		pair<bool, double> result = classifier->getConfidence(newExamples[i]);
		double score = result.second;
		if (positive ^ result.first)
			score = -score;
		newConfidences.push_back(make_pair(i, score));
	}
	sort(existingConfidences.begin(), existingConfidences.end(), [](pair<size_t, double> a, pair<size_t, double> b) {
		return a.second > b.second; // descending order (high confidence first)
	});
	sort(newConfidences.begin(), newConfidences.end(), [](pair<size_t, double> a, pair<size_t, double> b) {
		return a.second < b.second; // ascending order (low confidence first)
	});
	// add new examples until there is no more space, then replace existing examples that have higher confidence than new examples
	auto newConfidence = newConfidences.cbegin();
	while (examples.size() < examples.capacity() && newConfidence != newConfidences.cend()) {
		examples.push_back(newExamples[newConfidence->first]);
		++newConfidence;
	}
	auto existingConfidence = existingConfidences.cbegin();
	while (existingConfidence != existingConfidences.cend()
			&& newConfidence != newConfidences.cend()
			&& newConfidence->second < existingConfidence->second) {
		examples[existingConfidence->first] = newExamples[newConfidence->first];
		++existingConfidence;
		++newConfidence;
	}
}

} /* namespace classification */
