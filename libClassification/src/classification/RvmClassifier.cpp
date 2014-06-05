/*
 * RvmClassifier.cpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */

#include "classification/RvmClassifier.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "logging/LoggerFactory.hpp"
#ifdef WITH_MATLAB_CLASSIFIER
	#include "mat.h"
#endif
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"
#include <stdexcept>
#include <fstream>

using boost::filesystem::path;
using logging::Logger;
using logging::LoggerFactory;
using cv::Mat;
using boost::lexical_cast;
using boost::property_tree::ptree;
using std::pair;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::make_pair;
using std::invalid_argument;
using std::runtime_error;
using std::logic_error;

namespace classification {

RvmClassifier::RvmClassifier(shared_ptr<Kernel> kernel, bool cascadedCoefficients) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() { // TODO coefficients() probably different... vector<vector<... how?

	//useCaching = !cascadedCoefficients;
}

RvmClassifier::~RvmClassifier() {}

bool RvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

pair<bool, double> RvmClassifier::getConfidence(const Mat& featureVector) const {
	return getConfidence(computeHyperplaneDistance(featureVector));

}

pair<bool, double> RvmClassifier::getConfidence(pair<int, double> levelAndDistance) const {
	if (classify(levelAndDistance))
		return make_pair(true, levelAndDistance.second);
	else
		return make_pair(false, -levelAndDistance.second);
}

bool RvmClassifier::classify(pair<int, double> levelAndDistance) const {
	// TODO the following todo was moved here from the end of the getHyperplaneDistance function (was in classify before)
	// TODO: filter statistics, nDropedOutAsNonFace[filter_level]++;
	// We ran till the REAL LAST filter (not just the numUsedFilters one), save the certainty
	int filterLevel = levelAndDistance.first;
	double hyperplaneDistance = levelAndDistance.second;
	return filterLevel + 1 == this->numFiltersToUse && hyperplaneDistance >= this->hierarchicalThresholds[filterLevel];
}

pair<int, double> RvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	int filterLevel = -1;
	double hyperplaneDistance = 0;
	vector<double> filterEvalCache(this->numFiltersToUse);
	do {
		++filterLevel;
//		hyperplaneDistance = computeHyperplaneDistance(featureVector, filterLevel);
		hyperplaneDistance = computeHyperplaneDistanceCached(featureVector, filterLevel, filterEvalCache);
	} while (hyperplaneDistance >= hierarchicalThresholds[filterLevel] && filterLevel + 1 < this->numFiltersToUse); // TODO check the logic of this... may have a look at the WvmClassifier
	return make_pair(filterLevel, hyperplaneDistance);
}

double RvmClassifier::computeHyperplaneDistance(const Mat& featureVector, const size_t filterLevel) const {
	double distance = -bias;
	for (size_t i = 0; i <= filterLevel; ++i)
		distance += coefficients[filterLevel][i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

double RvmClassifier::computeHyperplaneDistanceCached(const Mat& featureVector, const size_t filterLevel, vector<double>& filterEvalCache) const {
	
	if (filterEvalCache.size() == filterLevel && filterLevel != 0) {
		double distance = filterEvalCache[filterLevel-1];
		distance += coefficients[filterLevel][filterLevel] * kernel->compute(featureVector, supportVectors[filterLevel]);
		filterEvalCache.push_back(distance);
		return distance;
	}

	filterEvalCache.clear();

	double distance = -bias;
	for (size_t i=0; i<=filterLevel; ++i) {
		distance += coefficients[filterLevel][i] * kernel->compute(featureVector, supportVectors[i]);
	}

	filterEvalCache.push_back(distance);

	return distance;
}

unsigned int RvmClassifier::getNumFiltersToUse(void) const
{
	return numFiltersToUse;
}

void RvmClassifier::setNumFiltersToUse(const unsigned int numFilters)
{
	if (numFilters == 0 || numFilters > this->coefficients.size()) {
		numFiltersToUse = this->coefficients.size();
	} else {
		numFiltersToUse = numFilters;
	}
}

shared_ptr<RvmClassifier> RvmClassifier::load(const ptree& subtree)
{
	path classifierFile = subtree.get<path>("classifierFile");
	if (classifierFile.extension() == ".mat") {
		shared_ptr<RvmClassifier> rvm = loadFromMatlab(classifierFile.string(), subtree.get<string>("thresholdsFile"));
		// Do some stuff, e.g.
		//int numFiltersToUse = subtree.get<int>("");
		//Number filters to use
		//wvm->numUsedFilters=280;	// Todo make dynamic (from script)
		return rvm;
	}
	else {
		throw logic_error("ProbabilisticRvmClassifier: Only loading of .mat RVMs is supported. If you want to load a non-cascaded RVM, use an SvmClassifier.");
	}
}

shared_ptr<RvmClassifier> RvmClassifier::loadFromMatlab(const string& classifierFilename, const string& thresholdsFilename)
{
	Logger logger = Loggers->getLogger("classification");

#ifdef WITH_MATLAB_CLASSIFIER
	logger.info("Loading RVM classifier from Matlab file: " + classifierFilename);

	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(classifierFilename.c_str(), "r");
	if (pmatfile == NULL) {
		throw invalid_argument("RvmClassifier: Could not open the provided classifier filename: " + classifierFilename);
	}

	pmxarray = matGetVariable(pmatfile, "num_hk");
	if (pmxarray == 0) {
		throw runtime_error("RvmClassifier: There is a no num_hk in the classifier file.");
		// TODO (concerns the whole class): I think we leak memory here (all the MATFile and double pointers etc.)?
	}
	matdata = mxGetPr(pmxarray);
	int numFilter = (int)matdata[0];
	mxDestroyArray(pmxarray);
	logger.debug("Found " + lexical_cast<string>(numFilter)+" reduced set vectors (RSVs).");

	float nonlinThreshold;
	int nonLinType;
	float basisParam;
	int polyPower;
	float divisor;
	pmxarray = matGetVariable(pmatfile, "param_nonlin1_rvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		nonlinThreshold = (float)matdata[0];
		nonLinType = (int)matdata[1];
		basisParam = (float)(matdata[2] / 65025.0); // because the training images gray level values were divided by 255
		polyPower = (int)matdata[3];
		divisor = (float)matdata[4];
		mxDestroyArray(pmxarray);
	}
	else {
		pmxarray = matGetVariable(pmatfile, "param_nonlin1");
		if (pmxarray != 0) {
			matdata = mxGetPr(pmxarray);
			nonlinThreshold = (float)matdata[0];
			nonLinType = (int)matdata[1];
			basisParam = (float)(matdata[2] / 65025.0); // because the training images gray level values were divided by 255
			polyPower = (int)matdata[3];
			divisor = (float)matdata[4];
			mxDestroyArray(pmxarray);
		}
		else {
			throw runtime_error("RvmClassifier: Could not find kernel parameters and bias.");
			// TODO tidying up?!
		}
	}
	shared_ptr<Kernel> kernel;
	if (nonLinType == 1) { // polynomial kernel
		kernel.reset(new PolynomialKernel(1 / divisor, basisParam / divisor, polyPower));
	}
	else if (nonLinType == 2) { // RBF kernel
		kernel.reset(new RbfKernel(basisParam));
	}
	else {
		throw runtime_error("RvmClassifier: Unsupported kernel type. Currently, only polynomial and RBF kernels are supported.");
		// TODO We should also throw/print the unsupported nonLinType value to the user
	}
	// TODO: logger.debug("Loaded kernel with params... ");

	shared_ptr<RvmClassifier> rvm = make_shared<RvmClassifier>(kernel);
	rvm->bias = nonlinThreshold;

	logger.debug("Reading the " + lexical_cast<string>(numFilter)+" non-linear filters support_hk* and weight_hk* ...");
	char str[100];
	sprintf(str, "support_hk%d", 1);
	pmxarray = matGetVariable(pmatfile, str);
	if (pmxarray == 0) {
		throw runtime_error("RvmClassifier: Unable to find the matrix 'support_hk1' in the classifier file.");
	}
	if (mxGetNumberOfDimensions(pmxarray) != 2) {
		throw runtime_error("RvmClassifier: The matrix 'support_hk1' in the classifier file should have 2 dimensions.");
	}
	const mwSize *dim = mxGetDimensions(pmxarray);
	int filterSizeY = (int)dim[0]; // height
	int filterSizeX = (int)dim[1]; // width TODO check if this is right with eg 24x16
	mxDestroyArray(pmxarray);

	// Alloc space for SV's and alphas (weights)
	rvm->supportVectors.reserve(numFilter);
	rvm->coefficients.reserve(numFilter);

	// Read the reduced vectors and coefficients (weights):
	for (int i = 0; i < numFilter; ++i) {
		sprintf(str, "support_hk%d", i + 1);
		pmxarray = matGetVariable(pmatfile, str);
		if (pmxarray == 0) {
			throw runtime_error("RvmClassifier: Unable to find the matrix 'support_hk" + lexical_cast<string>(i + 1) + "' in the classifier file.");
		}
		if (mxGetNumberOfDimensions(pmxarray) != 2) {
			throw runtime_error("RvmClassifier: The matrix 'support_hk" + lexical_cast<string>(i + 1) + "' in the classifier file should have 2 dimensions.");
		}

		// Read the reduced-vector:
		matdata = mxGetPr(pmxarray);
		int k = 0;
		Mat supportVector(filterSizeY, filterSizeX, CV_32F);
		float* values = supportVector.ptr<float>(0);
		for (int x = 0; x < filterSizeX; ++x)	// column-major order (ML-convention)
			for (int y = 0; y < filterSizeY; ++y)
				values[y * filterSizeX + x] = static_cast<float>(matdata[k++]); // because the training images gray level values were divided by 255
		rvm->supportVectors.push_back(supportVector);
		mxDestroyArray(pmxarray);

		sprintf(str, "weight_hk%d", i + 1);
		pmxarray = matGetVariable(pmatfile, str);
		if (pmxarray != 0) {
			const mwSize *dim = mxGetDimensions(pmxarray);
			if ((dim[1] != i + 1) && (dim[0] != i + 1)) {
				throw runtime_error("RvmClassifier: The matrix " + lexical_cast<string>(str)+" in the classifier file should have a dimensions 1x" + lexical_cast<string>(i + 1) + " or " + lexical_cast<string>(i + 1) + "x1");
			}
			vector<float> coefficientsForFilter;
			matdata = mxGetPr(pmxarray);
			for (int j = 0; j <= i; ++j) {
				coefficientsForFilter.push_back(static_cast<float>(matdata[j]));
			}
			rvm->coefficients.push_back(coefficientsForFilter);
			mxDestroyArray(pmxarray);
		}
	}	// end for over numHKs
	logger.debug("Vectors and weights successfully read.");

	if (matClose(pmatfile) != 0) {
		logger.warn("RvmClassifier: Could not close file " + classifierFilename);
		// TODO What is this? An error? Info? Throw an exception?
	}
	logger.info("RVM successfully read.");


	//MATFile *mxtFile = matOpen(args->threshold, "r");
	logger.info("Loading RVM thresholds from Matlab file: " + thresholdsFilename);
	pmatfile = matOpen(thresholdsFilename.c_str(), "r");
	if (pmatfile == 0) {
		throw runtime_error("RvmClassifier: Unable to open the thresholds file (wrong format?):" + thresholdsFilename);
	}
	else {
		pmxarray = matGetVariable(pmatfile, "hierar_thresh");
		if (pmxarray == 0) {
			throw runtime_error("RvmClassifier: Unable to find the matrix hierar_thresh in the thresholds file.");
		}
		else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			for (int o = 0; o < (int)dim[1]; ++o) {
				rvm->hierarchicalThresholds.push_back(static_cast<float>(matdata[o]));
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}

	if (rvm->hierarchicalThresholds.size() != rvm->coefficients.size()) {
		throw runtime_error("RvmClassifier: Something seems to be wrong, hierarchicalThresholds.size() != coefficients.size(): " + lexical_cast<string>(rvm->hierarchicalThresholds.size()) + "!=" + lexical_cast<string>(rvm->coefficients.size()));
	}

	logger.info("RVM thresholds successfully read.");
	rvm->setNumFiltersToUse(numFilter);
	return rvm;
#else
	string errorMessage("RvmClassifier: Cannot load a Matlab classifier, library compiled without support for Matlab. Please re-run CMake with WITH_MATLAB_CLASSIFIER enabled.");
	logger.error(errorMessage);
	throw std::runtime_error(errorMessage);
#endif
}

} /* namespace classification */
