/*
 * SvmClassifier.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/SvmClassifier.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "logging/LoggerFactory.hpp"
#include "mat.h"
#include <iostream>
#include <stdexcept>

using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using std::make_shared;
using std::make_pair;
using std::cout;
using std::invalid_argument;
using std::runtime_error;

namespace classification {

SvmClassifier::SvmClassifier(shared_ptr<Kernel> kernel) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() {}

SvmClassifier::~SvmClassifier() {}

bool SvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

bool SvmClassifier::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

double SvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	double distance = -bias;
	for (size_t i = 0; i < supportVectors.size(); ++i)
		distance += coefficients[i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

void SvmClassifier::setSvmParameters(vector<Mat> supportVectors, vector<float> coefficients, double bias) {
	this->supportVectors = supportVectors;
	this->coefficients = coefficients;
	this->bias = bias;
}

shared_ptr<SvmClassifier> SvmClassifier::loadMatlab(const string& classifierFilename, const string& thresholdsFilename)
{
	Logger logger = Loggers->getLogger("classification");
	logger.info("Loading SVM classifier from matlab file: " + classifierFilename);

	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(classifierFilename.c_str(), "r");
	if (pmatfile == NULL) {
		throw invalid_argument("SvmClassifier: Could not open the provided classifier filename: " + classifierFilename);
	}

	pmxarray = matGetVariable(pmatfile, "param_nonlin1");
	if (pmxarray == 0) {
		throw runtime_error("SvmClassifier: There is a no param_nonlin1 in the classifier file.");
		// TODO (concerns the whole class): I think we leak memory here (all the MATFile and double pointers etc.)?
	}
	matdata = mxGetPr(pmxarray);
	// TODO we don't need all of those (added by peter: or do we? see polynomial kernel type)
	float nonlinThreshold = (float)matdata[0];
	int nonLinType        = (int)matdata[1];
	float basisParam      = (float)(matdata[2]/65025.0); // because the training image's graylevel values were divided by 255
	int polyPower         = (int)matdata[3];
	float divisor         = (float)matdata[4];
	mxDestroyArray(pmxarray);

	shared_ptr<Kernel> kernel;
	if (nonLinType == 1) { // polynomial kernel
		kernel.reset(new PolynomialKernel(1 / divisor, basisParam / divisor, polyPower));
	} else if (nonLinType == 2) { // RBF kernel
		kernel.reset(new RbfKernel(basisParam));
	} else {
		throw runtime_error("SvmClassifier: Unsupported kernel type. Currently, only polynomial and RBF kernels are supported.");
		// TODO We should also throw/print the unsupported nonLinType value to the user
	}

	shared_ptr<SvmClassifier> svm = make_shared<SvmClassifier>(kernel);
	svm->bias = nonlinThreshold;

	pmxarray = matGetVariable(pmatfile, "support_nonlin1");
	if (pmxarray == 0) {
		throw runtime_error("SvmClassifier: There is a nonlinear SVM in the file, but the matrix support_nonlin1 is lacking.");
	} 
	if (mxGetNumberOfDimensions(pmxarray) != 3) {
		throw runtime_error("SvmClassifier: The matrix support_nonlin1 in the file should have 3 dimensions.");
	}
	const mwSize *dim = mxGetDimensions(pmxarray);
	int numSV = (int)dim[2];
	matdata = mxGetPr(pmxarray);

	int filter_size_x = (int)dim[1];
	int filter_size_y = (int)dim[0];

	// Alloc space for SV's and alphas (weights)
	svm->supportVectors.reserve(numSV);
	svm->coefficients.reserve(numSV);

	int k = 0;
	for (int sv = 0; sv < numSV; ++sv) {
		Mat supportVector(filter_size_y, filter_size_x, CV_8U);
		uchar* values = supportVector.ptr<uchar>(0);
		for (int x = 0; x < filter_size_x; ++x)	// column-major order (ML-convention)
			for (int y = 0; y < filter_size_y; ++y)
				values[y * filter_size_x + x] = static_cast<uchar>(255.0 * matdata[k++]); // because the training images gray level values were divided by 255
		svm->supportVectors.push_back(supportVector);
	}
	mxDestroyArray(pmxarray);

	pmxarray = matGetVariable(pmatfile, "weight_nonlin1");
	if (pmxarray == 0) {
		throw runtime_error("SvmClassifier: There is a nonlinear SVM in the file but the matrix threshold_nonlin is lacking.");
	}
	matdata = mxGetPr(pmxarray);
	for (int sv = 0; sv < numSV; ++sv)
		svm->coefficients.push_back(static_cast<float>(matdata[sv]));
	mxDestroyArray(pmxarray);

	if (matClose(pmatfile) != 0) {
		logger.warn("SvmClassifier: Could not close file " + classifierFilename);
		// TODO What is this? An error? Info? Throw an exception?
	}

	logger.info("SVM successfully read.");

	return svm;
}

} /* namespace classification */
