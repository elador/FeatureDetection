/*
 * ProbabilisticRvmClassifier.cpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticRvmClassifier.hpp"
#include "classification/RvmClassifier.hpp"
#include "logging/LoggerFactory.hpp"
#ifdef WITH_MATLAB_CLASSIFIER
	#include "mat.h"
#endif
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include <iostream>
#include <stdexcept>

using logging::Logger;
using logging::LoggerFactory;
using cv::Mat;
using boost::filesystem::path;
using boost::property_tree::ptree;
using std::pair;
using std::string;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;
using std::runtime_error;
using std::logic_error;

namespace classification {

ProbabilisticRvmClassifier::ProbabilisticRvmClassifier() :
		rvm(), logisticA(0), logisticB(0) {}

ProbabilisticRvmClassifier::ProbabilisticRvmClassifier(shared_ptr<RvmClassifier> rvm, double logisticA, double logisticB) :
		rvm(rvm), logisticA(logisticA), logisticB(logisticB) {}

bool ProbabilisticRvmClassifier::classify(const Mat& featureVector) const {
	return rvm->classify(featureVector);
}

pair<bool, double> ProbabilisticRvmClassifier::getConfidence(const Mat& featureVector) const {
	return rvm->getConfidence(featureVector);
}

pair<bool, double> ProbabilisticRvmClassifier::getProbability(const Mat& featureVector) const {
	return getProbability(rvm->computeHyperplaneDistance(featureVector));
}

pair<bool, double> ProbabilisticRvmClassifier::getProbability(pair<int, double> levelAndDistance) const {
	// Do sigmoid stuff:
	// NOTE Patrik: Here we calculate the probability for all RVM patches, also of those that
	//      did not run up to the last filter. Those probabilities are wrong, but for the face-
	//      detector it seems to work out ok. However, for the eye-detector, the probability-maps
	//      don't look good. See below for the code.
	double probability = 1.0f / (1.0f + exp(logisticA + logisticB * levelAndDistance.second));
	return make_pair(rvm->classify(levelAndDistance), probability);
}

void ProbabilisticRvmClassifier::setLogisticParameters(double logisticA, double logisticB) {
	this->logisticA = logisticA;
	this->logisticB = logisticB;
}

shared_ptr<ProbabilisticRvmClassifier> ProbabilisticRvmClassifier::load(const ptree& subtree)
{
	path classifierFile = subtree.get<path>("classifierFile");
	if (classifierFile.extension() == ".mat") {
		pair<double, double> logisticParams = ProbabilisticRvmClassifier::loadSigmoidParamsFromMatlab(subtree.get<path>("thresholdsFile").string());
		shared_ptr<RvmClassifier> rvm = RvmClassifier::load(subtree);
		return make_shared<ProbabilisticRvmClassifier>(rvm, logisticParams.first, logisticParams.second);
		// TODO: Re-think this whole loading logic. As soon as we call loadMatlab here, we lose the ptree, but
		// we want the RvmClassifier to be able to load additional parameters like a "threshold" or "numUsedVectors".
		// Option 1: Make a pair<float sigmA, float sigmB> loadSigmFromML(...); <- changed to this now
		//        2: Here, first call rvm = RvmClassifier::loadConfig(subtree), loads everything.
		//			 Then, here, call prvm = ProbRvmClass::loadMatlab/ProbabilisticStuff(rvm, thresholdsFile);
	} else {
		throw logic_error("ProbabilisticRvmClassifier: Only loading of .mat RVMs is supported. If you want to load a non-cascaded RVM, use an SvmClassifier.");
	}
}

pair<double, double> ProbabilisticRvmClassifier::loadSigmoidParamsFromMatlab(const string& logisticFilename)
{
	Logger logger = Loggers->getLogger("classification");

#ifdef WITH_MATLAB_CLASSIFIER
	// Load sigmoid stuff:
	double logisticA, logisticB;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(logisticFilename.c_str(), "r");
	if (pmatfile == 0) {
		throw invalid_argument("ProbabilisticRvmClassifier: Unable to open the thresholds-file: " + logisticFilename);
	}

	//TODO is there a case (when svm+wvm from same training-data) when there exists only a posterior_svm, and I should use this here?
	pmxarray = matGetVariable(pmatfile, "posterior_wrvm");
	if (pmxarray == 0) {
		throw runtime_error("ProbabilisticRvmClassifier: Unable to find the vector posterior_wrvm. If you don't want probabilistic output, don't use a probabilistic classifier.");
		//logisticA = logisticB = 0; // TODO: Does it make sense to continue?
	}
	else {
		double* matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		if (dim[1] != 2) {
			throw runtime_error("ProbabilisticRvmClassifier: Size of vector posterior_wrvm !=2. If you don't want probabilistic output, don't use a probabilistic classifier.");
			//logisticA = logisticB = 0; // TODO: Does it make sense to continue?
		}
		else {
			logisticB = matdata[0];
			logisticA = matdata[1];
		}
		mxDestroyArray(pmxarray);
	}
	matClose(pmatfile);

	return make_pair(logisticA, logisticB);
#else
	string errorMessage("ProbabilisticRvmClassifier: Cannot load a Matlab classifier, library compiled without support for Matlab. Please re-run CMake with WITH_MATLAB_CLASSIFIER enabled.");
	logger.error(errorMessage);
	throw std::runtime_error(errorMessage);
#endif
}

} /* namespace classification */
