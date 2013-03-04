/*
 * ProbabilisticSvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "mat.h"
#include <iostream>

using std::make_pair;
using std::make_shared;

namespace classification {

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier(shared_ptr<SvmClassifier> svm, double logisticA, double logisticB) :
		svm(svm), logisticA(logisticA), logisticB(logisticB) {}

ProbabilisticSvmClassifier::~ProbabilisticSvmClassifier() {}

pair<bool, double> ProbabilisticSvmClassifier::classify(const Mat& featureVector) const {
	double hyperplaneDistance = svm->computeHyperplaneDistance(featureVector);
	double probability = 1.0f / (1.0f + exp(logisticA + logisticB * hyperplaneDistance));
	return make_pair(svm->classify(hyperplaneDistance), probability);
}

shared_ptr<ProbabilisticSvmClassifier> ProbabilisticSvmClassifier::loadMatlab(const string& classifierFilename, const string& thresholdsFilename)
{
	// Load sigmoid stuff:
	double logisticA, logisticB;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		std::cout << "[DetSVM] Unable to open the file (wrong format?):" << std::endl <<  thresholdsFilename << std::endl;
		// THROW
	} else {
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO 2012: is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		//TODO new2013: the posterior_svm is the same for all thresholds file, so the thresholds file is not needed for SVMs. The posterior_svm should go into the classifier.mat!
		pmxarray = matGetVariable(pmatfile, "posterior_svm");
		if (pmxarray == 0) {
			std::cout << "[DetSVM] WARNING: Unable to find the vector posterior_svm, disable prob. SVM output;" << std::endl;
			// TODO prob. output cannot be disabled in the new version of this lib -> throw exception or log info message or something
			logisticA = logisticB = 0;
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				std::cout << "[DetSVM] WARNING: Size of vector posterior_svm !=2, disable prob. SVM output;" << std::endl;
				// TODO same as previous TODO
				logisticA = logisticB = 0;
			} else {
				logisticB = matdata[0];
				logisticA = matdata[1];
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}

	// Load the detector and thresholds:
	shared_ptr<SvmClassifier> svm = SvmClassifier::loadMatlab(classifierFilename, thresholdsFilename);

	return make_shared<ProbabilisticSvmClassifier>(svm, logisticA, logisticB);
}

} /* namespace classification */
