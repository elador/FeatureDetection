/*
 * ProbabilisticSvmClassifier.cpp
 *
 *  Created on: 25.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
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
using std::runtime_error;

namespace classification {

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier(shared_ptr<Kernel> kernel, double logisticA, double logisticB) :
		svm(make_shared<SvmClassifier>(kernel)), logisticA(logisticA), logisticB(logisticB) {}

ProbabilisticSvmClassifier::ProbabilisticSvmClassifier(shared_ptr<SvmClassifier> svm, double logisticA, double logisticB) :
		svm(svm), logisticA(logisticA), logisticB(logisticB) {}

bool ProbabilisticSvmClassifier::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

pair<bool, double> ProbabilisticSvmClassifier::getConfidence(const Mat& featureVector) const {
	return svm->getConfidence(featureVector);
}

pair<bool, double> ProbabilisticSvmClassifier::getProbability(const Mat& featureVector) const {
	return getProbability(svm->computeHyperplaneDistance(featureVector));
}

pair<bool, double> ProbabilisticSvmClassifier::getProbability(double hyperplaneDistance) const {
	double fABp = logisticA + logisticB * hyperplaneDistance;
	double probability = fABp >= 0 ? exp(-fABp) / (1.0 + exp(-fABp)) : 1.0 / (1.0 + exp(fABp));
	return make_pair(svm->classify(hyperplaneDistance), probability);
}

void ProbabilisticSvmClassifier::setLogisticParameters(double logisticA, double logisticB) {
	this->logisticA = logisticA;
	this->logisticB = logisticB;
}

void ProbabilisticSvmClassifier::store(std::ofstream& file) {
	svm->store(file);
	file << "Logistic " << logisticA << ' ' << logisticB << '\n';
}

std::shared_ptr<ProbabilisticSvmClassifier> ProbabilisticSvmClassifier::load(std::ifstream& file) {
	shared_ptr<SvmClassifier> svm = SvmClassifier::load(file);
	string tmp;
	double logisticA, logisticB;
	file >> tmp; // "Logistic"
	file >> logisticA;
	file >> logisticB;
	return make_shared<ProbabilisticSvmClassifier>(svm, logisticA, logisticB);
}

shared_ptr<ProbabilisticSvmClassifier> ProbabilisticSvmClassifier::load(const ptree& subtree)
{
	path classifierFile = subtree.get<path>("classifierFile");
	shared_ptr<ProbabilisticSvmClassifier> psvm;
	if (classifierFile.extension() == ".mat") {
		psvm = loadFromMatlab(classifierFile.string(), subtree.get<string>("thresholdsFile"));
	} else {
		shared_ptr<SvmClassifier> svm = SvmClassifier::loadFromText(classifierFile.string()); // Todo: Make a ProbabilisticSvmClassifier::loadFromText(...)
		if (subtree.get("logisticA", 0.0) == 0.0 || subtree.get("logisticB", 0.0) == 0.0) {
			std::cout << "Warning, one or both sigmoid parameters not set in config, using default sigmoid parameters." << std::endl; // TODO use logger
		}
		psvm = make_shared<ProbabilisticSvmClassifier>(svm);
	}

	// If the user sets the logistic parameters in the config, overwrite the current settings with it
	double logisticA = subtree.get("logisticA", 0.0); // TODO: This is not so good, better use the try/catch of get(...). Because: What if the user actually sets a value of 0.0 in the config...
	double logisticB = subtree.get("logisticB", 0.0);
	if (logisticA != 0.0 && logisticB != 0.0) {
		psvm->setLogisticParameters(logisticA, logisticB);
	}

	psvm->getSvm()->setThreshold(subtree.get("threshold", 0.0f));

	return psvm;
}

shared_ptr<ProbabilisticSvmClassifier> ProbabilisticSvmClassifier::loadFromMatlab(const string& classifierFilename, const string& logisticFilename)
{
	pair<double, double> sigmoidParams = loadSigmoidParamsFromMatlab(logisticFilename);
	// Load the detector and thresholds:
	shared_ptr<SvmClassifier> svm = SvmClassifier::loadFromMatlab(classifierFilename);
	return make_shared<ProbabilisticSvmClassifier>(svm, sigmoidParams.first, sigmoidParams.second);
}

pair<double, double> ProbabilisticSvmClassifier::loadSigmoidParamsFromMatlab(const string& logisticFilename)
{
	Logger logger = Loggers->getLogger("classification");

#ifdef WITH_MATLAB_CLASSIFIER
	// Load sigmoid stuff:
	double logisticA = 0;
	double logisticB = 0;
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	pmatfile = matOpen(logisticFilename.c_str(), "r");
	if (pmatfile == 0) {
		throw runtime_error("ProbabilisticSvmClassifier: Unable to open the logistic file to read the logistic parameters (wrong format?):" + logisticFilename);
	}
	else {
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO 2012: is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		//TODO new2013: the posterior_svm is the same for all thresholds file, so the thresholds file is not needed for SVMs. The posterior_svm should go into the classifier.mat!
		pmxarray = matGetVariable(pmatfile, "posterior_svm");
		if (pmxarray == 0) {
			std::cout << "[DetSVM] WARNING: Unable to find the vector posterior_svm, disable prob. SVM output;" << std::endl;
			// TODO prob. output cannot be disabled in the new version of this lib -> throw exception or log info message or something
			logisticA = logisticB = 0;
		}
		else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				std::cout << "[DetSVM] WARNING: Size of vector posterior_svm !=2, disable prob. SVM output;" << std::endl;
				// TODO same as previous TODO
				logisticA = logisticB = 0;
			}
			else {
				logisticB = matdata[0];
				logisticA = matdata[1];
			}
			// TODO delete *matdata and *dim? -> No I don't think so, no 'new'
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}

	return make_pair(logisticA, logisticB);
#else
	string errorMessage("ProbabilisticSvmClassifier: Cannot load a Matlab classifier, library compiled without support for Matlab. Please re-run CMake with WITH_MATLAB_CLASSIFIER enabled.");
	logger.error(errorMessage);
	throw std::runtime_error(errorMessage);
#endif
}

} /* namespace classification */
