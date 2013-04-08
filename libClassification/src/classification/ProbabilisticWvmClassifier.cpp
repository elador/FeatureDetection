/*
 * ProbabilisticWvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"

#include "mat.h"
#include <iostream>

using std::make_pair;
using std::make_shared;

namespace classification {

ProbabilisticWvmClassifier::ProbabilisticWvmClassifier(shared_ptr<WvmClassifier> wvm, double logisticA, double logisticB) :
		wvm(wvm), logisticA(logisticA), logisticB(logisticB) {}

ProbabilisticWvmClassifier::~ProbabilisticWvmClassifier() {}

pair<bool, double> ProbabilisticWvmClassifier::classify( const Mat& featureVector ) const
{
	pair<int, double> levelAndDistance = wvm->computeHyperplaneDistance(featureVector);
	// Do sigmoid stuff:
	// NOTE Patrik: Here we calculate the probability for all WVM patches, also of those that
	//      did not run up to the last filter. Those probabilities are wrong, but for the face-
	//      detector it seems to work out ok. However, for the eye-detector, the probability-maps
	//      don't look good. See below for the code.
	double probability = 1.0f / (1.0f + exp(logisticA + logisticB * levelAndDistance.second));
	return make_pair(wvm->classify(levelAndDistance), probability);

	/* TODO: In case of the WVM, we could again distinguish if we calculate the ("wrong") probability
	 *			of all patches or only the ones that pass the last stage (correct probability), and
	 *			set all others to zero.
	//fp->certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
	// We ran till the REAL LAST filter (not just the numUsedFilters one):
	if(filter_level+1 == this->numLinFilters && fout >= this->hierarchicalThresholds[filter_level]) {
		certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
		return true;
	}
	// We didn't run up to the last filter (or only up to the numUsedFilters one)
	if(this->calculateProbabilityOfAllPatches==true) {
		certainty = 1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
		return false;
	} else {
		certainty = 0.0f;
		return false;
	}
	*/
}

shared_ptr<ProbabilisticWvmClassifier> ProbabilisticWvmClassifier::loadMatlab(const string& classifierFilename, const string& thresholdsFilename)
{
	// Load sigmoid stuff:
	double logisticA, logisticB;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[DetWVM] : Unable to open the file (wrong format?):" << std::endl << thresholdsFilename << std::endl;
		exit(EXIT_FAILURE);	// TODO Exceptions/Logger, like in non-probabilistic file
	} 

	//printf("fd_ReadDetector(): read posterior_svm parameter for probabilistic SVM output\n");
	//read posterior_wrvm parameter for probabilistic WRVM output
	//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
	pmxarray = matGetVariable(pmatfile, "posterior_wrvm");
	if (pmxarray == 0) {
		std::cout << "[DetWVM] WARNING: Unable to find the vector posterior_wrvm, disable prob. SVM output;" << std::endl;
		// TODO prob. output cannot be disabled in the new version of this lib -> throw exception or log info message or something
		logisticA = logisticB = 0;
	} else {
		double* matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		if (dim[1] != 2) {
			std::cout << "[DetWVM] WARNING: Size of vector posterior_wrvm !=2, disable prob. WRVM output;" << std::endl;
			// TODO same as previous TODO
			logisticA = logisticB = 0;
		} else {
			logisticB = matdata[0];
			logisticA = matdata[1];
		}
		mxDestroyArray(pmxarray);
	}
	matClose(pmatfile);

	// Load the detector and thresholds:
	shared_ptr<WvmClassifier> wvm = WvmClassifier::loadMatlab(classifierFilename, thresholdsFilename);

	return make_shared<ProbabilisticWvmClassifier>(wvm, logisticA, logisticB);
}

} /* namespace classification */
