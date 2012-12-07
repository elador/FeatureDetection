/*
 * FixedSizeLibSvmTraining.cpp
 *
 *  Created on: 07.12.2012
 *      Author: poschmann
 */

#include "classification/FixedSizeLibSvmTraining.h"
#include "classification/FeatureVector.h"
#include <utility>
#include <algorithm>

using std::pair;
using std::make_pair;
using std::sort;

namespace classification {

static bool compareProbabilities(pair<int, double> a, pair<int, double> b) {
	return a.second > b.second;
}

FixedSizeLibSvmTraining::FixedSizeLibSvmTraining(unsigned int positiveExamples, unsigned int negativeExamples,
		unsigned int minPositiveExamples) :
				dimensions(0), positiveExamples(), negativeExamples(), negativeInsertPosition(0),
				minPositiveExamples(minPositiveExamples) {
	this->positiveExamples.reserve(positiveExamples);
	this->negativeExamples.reserve(negativeExamples);
}

FixedSizeLibSvmTraining::~FixedSizeLibSvmTraining() {
	freeExamples(positiveExamples);
	freeExamples(negativeExamples);
}

void FixedSizeLibSvmTraining::reset(LibSvmClassifier& classifier) {
	freeExamples(positiveExamples);
	freeExamples(negativeExamples);
}

bool FixedSizeLibSvmTraining::retrain(LibSvmClassifier& classifier,
		const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
		const vector<shared_ptr<FeatureVector> >& newNegativeExamples) {
	if (!newPositiveExamples.empty())
		dimensions = newPositiveExamples.front()->getSize();
	addPositiveExamples(newPositiveExamples, classifier);
	addNegativeExamples(newNegativeExamples);
	if (positiveExamples.empty() || negativeExamples.empty())
		return false;
	bool success = train(classifier, dimensions, positiveExamples, negativeExamples);
	return success && positiveExamples.size() >= minPositiveExamples;
}

void FixedSizeLibSvmTraining::addPositiveExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
		LibSvmClassifier& classifier) {
	// compute probabilities of current positive examples and sort in ascending order
	vector<pair<int, double> > probabilities;
	probabilities.reserve(positiveExamples.size());
	for (unsigned int i = 1; i < positiveExamples.size(); ++i)
		probabilities.push_back(make_pair(i, computeSvmOutput(classifier.getModel(), positiveExamples[i])));
	sort(probabilities.begin(), probabilities.end(), compareProbabilities);
	// add new positive examples as long as no examples have to be removed
	vector<shared_ptr<FeatureVector> >::const_iterator eit = newPositiveExamples.begin();
	for (; eit != newPositiveExamples.end() && positiveExamples.size() < positiveExamples.capacity(); ++eit)
		positiveExamples.push_back(createNode(*(*eit)));
	// replace existing examples (beginning with the high probability ones) by new examples
	vector<pair<int, double> >::iterator pit = probabilities.begin();
	for (; eit != newPositiveExamples.end() && pit != probabilities.end(); ++eit, ++pit) {
		delete[] positiveExamples[pit->first];
		positiveExamples[pit->first] = createNode(*(*eit));
	}
}

void FixedSizeLibSvmTraining::addNegativeExamples(const vector<shared_ptr<FeatureVector> >& newNegativeExamples) {
	// add new negative examples as long as there is space available
	vector<shared_ptr<FeatureVector> >::const_iterator eit = newNegativeExamples.begin();
	for (; eit != newNegativeExamples.end() && negativeExamples.size() < negativeExamples.capacity(); ++eit)
		negativeExamples.push_back(createNode(*(*eit)));
	// replace the oldest negative examples by new ones
	for (; eit != newNegativeExamples.end(); ++eit) {
		delete[] negativeExamples[negativeInsertPosition];
		negativeExamples[negativeInsertPosition] = createNode(*(*eit));
		++negativeInsertPosition;
		if (negativeInsertPosition == negativeExamples.size())
			negativeInsertPosition = 0;
	}
}

} /* namespace classification */
