/*
 * LibSvmClassifier.cpp
 *
 *  Created on: 21.11.2013
 *      Author: poschmann
 */

#include "libsvm/LibSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/EmptyExampleManagement.hpp"
#include "svm.h"
#include <fstream>
#include <stdexcept>

using classification::Kernel;
using classification::SvmClassifier;
using classification::ProbabilisticSvmClassifier;
using classification::ExampleManagement;
using classification::UnlimitedExampleManagement;
using classification::EmptyExampleManagement;
using cv::Mat;
using std::move;
using std::string;
using std::vector;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;

namespace libsvm {

LibSvmClassifier::LibSvmClassifier(shared_ptr<Kernel> kernel, double cnu, bool oneClass, bool compensateImbalance, bool probabilistic) :
		LibSvmClassifier(make_shared<SvmClassifier>(kernel), cnu, oneClass, compensateImbalance, probabilistic) {}

LibSvmClassifier::LibSvmClassifier(shared_ptr<SvmClassifier> svm, double cnu, bool oneClass, bool compensateImbalance, bool probabilistic) :
		TrainableSvmClassifier(svm),
		compensateImbalance(compensateImbalance),
		probabilistic(probabilistic),
		probabilisticSvm(make_shared<ProbabilisticSvmClassifier>(svm)),
		utils(),
		param(),
		positiveExamples(new UnlimitedExampleManagement()),
		negativeExamples(new UnlimitedExampleManagement()),
		staticNegativeExamples() {
	if (oneClass && compensateImbalance)
		throw invalid_argument("LibSvmClassifier: a one-class SVM cannot have unbalanced data it needs to compensate for");
	if (oneClass && probabilistic)
		throw invalid_argument("LibSvmClassifier: a one-class SVM cannot have probabilistic output (yet)");
	if (oneClass)
		negativeExamples.reset(new EmptyExampleManagement());
	createParameters(svm->getKernel(), cnu, oneClass);
}

void LibSvmClassifier::createParameters(const shared_ptr<Kernel> kernel, double cnu, bool oneClass) {
	param.reset(new struct svm_parameter);
	param->cache_size = 100;
	param->eps = 1e-4;
	if (oneClass) {
		param->nu = cnu;
		param->svm_type = ONE_CLASS;
	} else {
		param->C = cnu;
		param->svm_type = C_SVC;
	}
	if (compensateImbalance) {
		param->nr_weight = 2;
		param->weight_label = (int*)malloc(param->nr_weight * sizeof(int));
		param->weight_label[0] = +1;
		param->weight_label[1] = -1;
		param->weight = (double*)malloc(param->nr_weight * sizeof(double));
		param->weight[0] = 1;
		param->weight[1] = 1;
	} else {
		param->nr_weight = 0;
		param->weight_label = nullptr;
		param->weight = nullptr;
	}
	param->shrinking = 0;
	param->probability = probabilistic ? 1 : 0;
	param->gamma = 0; // necessary for kernels that do not use this parameter
	param->degree = 0; // necessary for kernels that do not use this parameter
	utils.setKernelParams(*kernel, param.get());
}

void LibSvmClassifier::loadStaticNegatives(const string& negativesFilename, int maxNegatives, double scale) {
	staticNegativeExamples.reserve(maxNegatives);
	int negatives = 0;
	vector<double> values;
	double value;
	char separator;
	string line;
	std::ifstream file(negativesFilename.c_str());
	if (file.is_open()) {
		while (file.good() && negatives < maxNegatives) {
			if (!std::getline(file, line))
				break;
			negatives++;
			// read values from line
			values.clear();
			std::istringstream lineStream(line);
			while (lineStream.good() && !lineStream.fail()) {
				lineStream >> value >> separator;
				values.push_back(value);
			}
			// create node
			unique_ptr<struct svm_node[], NodeDeleter> data(new struct svm_node[values.size() + 1], utils.getNodeDeleter());
			for (size_t i = 0; i < values.size(); ++i) {
				data[i].index = i;
				data[i].value = scale * values[i];
			}
			data[values.size()].index = -1;
			staticNegativeExamples.push_back(move(data));
		}
	}
}

bool LibSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (newPositiveExamples.empty() && newNegativeExamples.empty()) // no new training data available -> no new training necessary
		return usable;
	positiveExamples->add(newPositiveExamples);
	negativeExamples->add(newNegativeExamples);
	if (positiveExamples->hasRequiredSize() && negativeExamples->hasRequiredSize())
		usable = train();
	return usable;
}

bool LibSvmClassifier::train() {
	vector<unique_ptr<struct svm_node[], NodeDeleter>> positiveExamples = move(createNodes(this->positiveExamples.get()));
	vector<unique_ptr<struct svm_node[], NodeDeleter>> negativeExamples = move(createNodes(this->negativeExamples.get()));
	if (compensateImbalance) {
		double positiveCount = positiveExamples.size();
		double negativeCount = negativeExamples.size() + staticNegativeExamples.size();
		param->weight[0] = negativeCount / positiveCount;
		param->weight[1] = positiveCount / negativeCount;
	}
	unique_ptr<struct svm_problem, ProblemDeleter> problem = move(createProblem(
			positiveExamples, negativeExamples, staticNegativeExamples));
	const char* message = svm_check_parameter(problem.get(), param.get());
	if (message != 0)
		throw invalid_argument(string("LibSvmClassifier: invalid SVM parameters: ") + message);
	unique_ptr<struct svm_model, ModelDeleter> model(svm_train(problem.get(), param.get()));
	svm->setSvmParameters(
			utils.extractSupportVectors(model.get()),
			utils.extractCoefficients(model.get()),
			utils.extractBias(model.get()));
	if (probabilistic) {
		// order of A and B in libSVM is reverse of order in ProbabilisticSvmClassifier
		// therefore ProbabilisticSvmClassifier.logisticA = libSVM.logisticB and vice versa
		probabilisticSvm->setLogisticParameters(utils.extractLogisticParamB(model.get()), utils.extractLogisticParamA(model.get()));
	}
	return true;
}

vector<unique_ptr<struct svm_node[], NodeDeleter>> LibSvmClassifier::createNodes(ExampleManagement* examples) {
	vector<unique_ptr<struct svm_node[], NodeDeleter>> nodes;
	nodes.reserve(examples->size());
	for (auto iterator = examples->iterator(); iterator->hasNext();)
		nodes.push_back(move(utils.createNode(iterator->next())));
	return move(nodes);
}

unique_ptr<struct svm_problem, ProblemDeleter> LibSvmClassifier::createProblem(
		const vector<unique_ptr<struct svm_node[], NodeDeleter>>& positiveExamples,
		const vector<unique_ptr<struct svm_node[], NodeDeleter>>& negativeExamples,
		const vector<unique_ptr<struct svm_node[], NodeDeleter>>& staticNegativeExamples) {
	unique_ptr<struct svm_problem, ProblemDeleter> problem(new struct svm_problem);
	problem->l = positiveExamples.size() + negativeExamples.size() + staticNegativeExamples.size();
	problem->y = new double[problem->l];
	problem->x = new struct svm_node *[problem->l];
	size_t i = 0;
	for (auto& example : positiveExamples) {
		problem->y[i] = 1;
		problem->x[i] = example.get();
		++i;
	}
	for (auto& example : negativeExamples) {
		problem->y[i] = -1;
		problem->x[i] = example.get();
		++i;
	}
	for (auto& example : staticNegativeExamples) {
		problem->y[i] = -1;
		problem->x[i] = example.get();
		++i;
	}
	return move(problem);
}

void LibSvmClassifier::reset() {
	usable = false;
	svm->setSvmParameters(vector<Mat>(), vector<float>(), 0.0);
	positiveExamples->clear();
	negativeExamples->clear();
}

} /* namespace libsvm */
