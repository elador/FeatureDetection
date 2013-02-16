/*
 * WvmClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#include "classification/WvmClassifier.h"
#include "classification/FeatureVector.h"
#include "VDetectorVectorMachine.h"
#include "FdPatch.h"

using std::make_pair;

namespace classification {

WvmClassifier::WvmClassifier(shared_ptr<VDetectorVectorMachine> wvm) : wvm(wvm) {}

WvmClassifier::~WvmClassifier() {}

pair<bool, double> WvmClassifier::classify(const FeatureVector& featureVector) const {
	FdPatch* patch = new FdPatch;
	patch->data = new unsigned char[featureVector.getSize()];
	for (unsigned int i = 0; i < featureVector.getSize(); ++i)
		patch->data[i] = (unsigned char)(255 * featureVector[i] + 0.5f);

	bool positive = wvm->classify(patch);
	double probability = patch->certainty[wvm->getIdentifier()];

	delete[] patch->data;
	patch->data = NULL;
	delete patch;

	return make_pair(positive, probability);
}

} /* namespace classification */
