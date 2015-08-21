/*
 * SvmClassifier.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/SvmClassifier.hpp"
#include "classification/HistogramIntersectionKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "logging/LoggerFactory.hpp"
#ifdef WITH_MATLAB_CLASSIFIER
	#include "mat.h"
#endif
#include <stdexcept>

using logging::Logger;
using logging::LoggerFactory;
using cv::Mat;
using std::pair;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::make_pair;
using std::invalid_argument;
using std::runtime_error;

namespace classification {

SvmClassifier::SvmClassifier(shared_ptr<Kernel> kernel) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() {}

bool SvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

pair<bool, double> SvmClassifier::getConfidence(const Mat& featureVector) const {
	return getConfidence(computeHyperplaneDistance(featureVector));
}

bool SvmClassifier::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

pair<bool, double> SvmClassifier::getConfidence(double hyperplaneDistance) const {
	if (classify(hyperplaneDistance))
		return make_pair(true, hyperplaneDistance);
	else
		return make_pair(false, -hyperplaneDistance);
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

void SvmClassifier::store(std::ofstream& file) {
	if (!file)
		throw runtime_error("SvmClassifier: Cannot write into stream");

	file << "Kernel ";
	if (dynamic_cast<LinearKernel*>(getKernel().get()))
		file << "Linear\n";
	else if (PolynomialKernel* kernel = dynamic_cast<PolynomialKernel*>(getKernel().get()))
		file << "Polynomial " << kernel->getDegree() << ' ' << kernel->getConstant() << ' ' << kernel->getAlpha() << '\n';
	else if (RbfKernel* kernel = dynamic_cast<RbfKernel*>(getKernel().get()))
		file << "RBF " << kernel->getGamma() << '\n';
	else if (dynamic_cast<HistogramIntersectionKernel*>(getKernel().get()))
		file << "HIK\n";
	else
		throw runtime_error("SvmClassifier: cannot write kernel parameters (unknown kernel type)");

	file << "Bias " << getBias() << '\n';
	file << "Coefficients " << getCoefficients().size() << '\n';
	for (float coefficient : getCoefficients())
		file << coefficient << '\n';
	const Mat& vector = getSupportVectors().front();
	int count = getSupportVectors().size();
	int rows = vector.rows;
	int cols = vector.cols;
	int channels = vector.channels();
	int depth = vector.depth();
	file << "SupportVectors " << count << ' ' << rows << ' ' << cols << ' ' << channels << ' ' << depth << '\n';
	switch (depth) {
		case CV_8U: storeSupportVectors<uchar>(file, getSupportVectors()); break;
		case CV_32S: storeSupportVectors<int32_t>(file, getSupportVectors()); break;
		case CV_32F: storeSupportVectors<float>(file, getSupportVectors()); break;
		case CV_64F: storeSupportVectors<double>(file, getSupportVectors()); break;
		default: throw runtime_error(
				"SvmClassifier: cannot store support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}
}

shared_ptr<SvmClassifier> SvmClassifier::load(std::ifstream& file) {
	if (!file)
		throw runtime_error("SvmClassifier: Cannot read from stream");
	string tmp;

	file >> tmp; // "Kernel"
	shared_ptr<Kernel> kernel;
	string kernelType;
	file >> kernelType;
	if (kernelType == "Linear") {
		kernel.reset(new LinearKernel());
	} else if (kernelType == "Polynomial") {
		int degree;
		double constant, scale;
		file >> degree >> constant >> scale;
		kernel.reset(new PolynomialKernel(scale, constant, degree));
	} else if (kernelType == "RBF") {
		double gamma;
		file >> gamma;
		kernel.reset(new RbfKernel(gamma));
	} else if (kernelType == "HIK") {
		kernel.reset(new HistogramIntersectionKernel());
	} else {
		throw runtime_error("SvmClassifier: Invalid kernel type: " + kernelType);
	}
	shared_ptr<SvmClassifier> svm = make_shared<SvmClassifier>(kernel);

	file >> tmp; // "Bias"
	file >> svm->bias;

	size_t count;
	file >> tmp; // "Coefficients"
	file >> count;
	svm->coefficients.resize(count);
	for (size_t i = 0; i < count; ++i)
		file >> svm->coefficients[i];

	int rows, cols, channels, depth;
	file >> tmp; // "SupportVectors"
	file >> count; // should be the same as above
	file >> rows;
	file >> cols;
	file >> channels;
	file >> depth;
	switch (depth) {
		case CV_8U: loadSupportVectors<uchar>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_32S: loadSupportVectors<int32_t>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_32F: loadSupportVectors<float>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		case CV_64F: loadSupportVectors<double>(file, count, rows, cols, channels, depth, svm->supportVectors); break;
		default: throw runtime_error(
				"SvmClassifier: cannot load support vectors of depth other than CV_8U, CV_32S, CV_32F or CV_64F");
	}

	return svm;
}

shared_ptr<SvmClassifier> SvmClassifier::loadFromText(const string& classifierFilename)
{
	Logger logger = Loggers->getLogger("classification");
	logger.info("Loading SVM classifier from text file: " + classifierFilename);

	std::ifstream file(classifierFilename.c_str());
	if (!file.is_open())
		throw runtime_error("SvmClassifier: Invalid classifier file");

	string line;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");

	// read kernel parameters
	shared_ptr<Kernel> kernel;
	std::istringstream lineStream(line);
	if (lineStream.good() && !lineStream.fail()) {
		string kernelType;
		lineStream >> kernelType;
		if (kernelType != "FullPolynomial")
			throw runtime_error("SvmClassifier: Invalid kernel type: " + kernelType);
		int degree;
		double constant, scale;
		lineStream >> degree >> constant >> scale;
		kernel.reset(new PolynomialKernel(scale, constant, degree));
	}

	shared_ptr<SvmClassifier> svm = make_shared<SvmClassifier>(kernel);

	int svCount;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "Number of SV : %d", &svCount);

	int dimensionCount;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "Dim of SV : %d", &dimensionCount);

	float bias;
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	std::sscanf(line.c_str(), "B0 : %f", &bias);
	svm->bias = bias;

	// coefficients
	svm->coefficients.resize(svCount);
	for (int i = 0; i < svCount; ++i) {
		float alpha;
		int index;
		if (!std::getline(file, line))
			throw runtime_error("SvmClassifier: Invalid classifier file");
		std::sscanf(line.c_str(), "alphas[%d]=%f", &index, &alpha);
		svm->coefficients[index] = alpha;
	}

	// read line containing "Support vectors: "
	if (!std::getline(file, line))
		throw runtime_error("SvmClassifier: Invalid classifier file");
	// read support vectors
	svm->supportVectors.reserve(svCount);
	for (int i = 0; i < svCount; ++i) {
		Mat vector(1, dimensionCount, CV_32F);
		if (!std::getline(file, line))
			throw runtime_error("SvmClassifier: Invalid classifier file");
		std::istringstream lineStream(line);
		if (!lineStream.good() || lineStream.fail())
			throw runtime_error("SvmClassifier: Invalid classifier file");
		float* values = vector.ptr<float>(0);
		for (int j = 0; j < dimensionCount; ++j)
			lineStream >> values[j];
		svm->supportVectors.push_back(vector);
	}
	// TODO: Note: We never close the file?
	logger.info("SVM successfully read.");

	return svm;
}

shared_ptr<SvmClassifier> SvmClassifier::loadFromMatlab(const string& classifierFilename)
{
	Logger logger = Loggers->getLogger("classification");

#ifdef WITH_MATLAB_CLASSIFIER
	logger.info("Loading SVM classifier from Matlab file: " + classifierFilename);

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
	int nonLinType = (int)matdata[1];
	float basisParam = (float)(matdata[2] / 65025.0); // because the training image's graylevel values were divided by 255
	int polyPower = (int)matdata[3];
	float divisor = (float)matdata[4];
	mxDestroyArray(pmxarray);

	shared_ptr<Kernel> kernel;
	if (nonLinType == 1) { // polynomial kernel
		kernel.reset(new PolynomialKernel(1 / divisor, basisParam / divisor, polyPower));
	}
	else if (nonLinType == 2) { // RBF kernel
		kernel.reset(new RbfKernel(basisParam));
	}
	else {
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
#else
	string errorMessage("SvmClassifier: Cannot load a Matlab classifier, library compiled without support for Matlab. Please re-run CMake with WITH_MATLAB_CLASSIFIER enabled.");
	logger.error(errorMessage);
	throw std::runtime_error(errorMessage);
#endif
}

} /* namespace classification */
