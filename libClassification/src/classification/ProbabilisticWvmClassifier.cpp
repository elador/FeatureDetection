/*
 * ProbabilisticWvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"
#include "logging/LoggerFactory.hpp"
#ifdef WITH_MATLAB_CLASSIFIER
	#include "mat.h"
#endif
#include <iostream>
#include <stdexcept>

using logging::Logger;
using logging::LoggerFactory;
using cv::Mat;
using boost::property_tree::ptree;
using std::pair;
using std::string;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;
using std::runtime_error;

namespace classification {

ProbabilisticWvmClassifier::ProbabilisticWvmClassifier(shared_ptr<WvmClassifier> wvm, double logisticA, double logisticB) :
		wvm(wvm), logisticA(logisticA), logisticB(logisticB) {}

bool ProbabilisticWvmClassifier::classify(const Mat& featureVector) const {
	return wvm->classify(featureVector);
}

pair<bool, double> ProbabilisticWvmClassifier::getConfidence(const Mat& featureVector) const {
	return wvm->getConfidence(featureVector);
}

pair<bool, double> ProbabilisticWvmClassifier::getProbability(const Mat& featureVector) const {
	return getProbability(wvm->computeHyperplaneDistance(featureVector));
}

pair<bool, double> ProbabilisticWvmClassifier::getProbability(pair<int, double> levelAndDistance) const {
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

shared_ptr<ProbabilisticWvmClassifier> ProbabilisticWvmClassifier::load(const ptree& subtree)
{
	pair<double, double> sigmoidParams = loadSigmoidParamsFromMatlab(subtree.get<string>("thresholdsFile"));
	// Load the detector and thresholds:
	shared_ptr<WvmClassifier> wvm = WvmClassifier::loadFromMatlab(subtree.get<string>("classifierFile"), subtree.get<string>("thresholdsFile"));
	shared_ptr<ProbabilisticWvmClassifier> pwvm = make_shared<ProbabilisticWvmClassifier>(wvm, sigmoidParams.first, sigmoidParams.second);

	pwvm->getWvm()->setLimitReliabilityFilter(subtree.get("threshold", 0.0f));

	return pwvm;
}

shared_ptr<ProbabilisticWvmClassifier> ProbabilisticWvmClassifier::loadFromMatlab(const string& classifierFilename, const string& thresholdsFilename)
{
	pair<double, double> sigmoidParams = loadSigmoidParamsFromMatlab(thresholdsFilename);
	// Load the detector and thresholds:
	shared_ptr<WvmClassifier> wvm = WvmClassifier::loadFromMatlab(classifierFilename, thresholdsFilename);
	return make_shared<ProbabilisticWvmClassifier>(wvm, sigmoidParams.first, sigmoidParams.second);
}

pair<double, double> ProbabilisticWvmClassifier::loadSigmoidParamsFromMatlab(const string& thresholdsFilename)
{
	Logger logger = Loggers->getLogger("classification");

#ifdef WITH_MATLAB_CLASSIFIER
	// Load sigmoid stuff:
	double logisticA, logisticB;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		throw invalid_argument("ProbabilisticWvmClassifier: Unable to open the thresholds-file: " + thresholdsFilename);
	}

	//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
	pmxarray = matGetVariable(pmatfile, "posterior_wrvm");
	if (pmxarray == 0) {
		throw runtime_error("ProbabilisticWvmClassifier: Unable to find the vector posterior_wrvm. If you don't want probabilistic output, don't use a probabilistic classifier.");
		//logisticA = logisticB = 0; // TODO: Does it make sense to continue?
	}
	else {
		double* matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		if (dim[1] != 2) {
			throw runtime_error("ProbabilisticWvmClassifier: Size of vector posterior_wrvm !=2. If you don't want probabilistic output, don't use a probabilistic classifier.");
			//logisticA = logisticB = 0; // TODO: Does it make sense to continue?
		}
		else {
			logisticB = matdata[0];
			logisticA = matdata[1];
		}
		// Todo: delete *matdata and *dim??
		mxDestroyArray(pmxarray);
	}
	matClose(pmatfile);

	return make_pair(logisticA, logisticB);
#else
	string errorMessage("ProbabilisticWvmClassifier: Cannot load a Matlab classifier, library compiled without support for Matlab. Please re-run CMake with WITH_MATLAB_CLASSIFIER enabled.");
	logger.error(errorMessage);
	throw std::runtime_error(errorMessage);
#endif
}

} /* namespace classification */
