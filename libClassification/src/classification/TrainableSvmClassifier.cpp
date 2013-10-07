/*
 * TrainableSvmClassifier.cpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#include "classification/TrainableSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/Kernel.hpp"
#include "svm.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

using std::invalid_argument;
using std::make_shared;
using std::make_pair;
using std::move;

namespace classification {

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<SvmClassifier> svm, double constraintsViolationCosts) :
		utils(), svm(svm), constraintsViolationCosts(constraintsViolationCosts),
		usable(false), staticNegativeExamples(), param(), problem(), model() {
	createParameters(svm->getKernel(), constraintsViolationCosts);
}

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts) :
		utils(), svm(make_shared<SvmClassifier>(kernel)), constraintsViolationCosts(constraintsViolationCosts),
		usable(false), staticNegativeExamples(), param(), problem(), model() {
	createParameters(kernel, constraintsViolationCosts);
}

TrainableSvmClassifier::~TrainableSvmClassifier() {}

bool TrainableSvmClassifier::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

void TrainableSvmClassifier::createParameters(const shared_ptr<Kernel> kernel, double constraintsViolationCosts) {
	param.reset(new struct svm_parameter);
	param->svm_type = C_SVC;
	param->C = constraintsViolationCosts;
	param->cache_size = 100;
	param->eps = 1e-3;
	param->nr_weight = 2;
	param->weight_label = (int*)malloc(param->nr_weight * sizeof(int));
	param->weight_label[0] = +1;
	param->weight_label[1] = -1;
	param->weight = (double*)malloc(param->nr_weight * sizeof(double));
	param->weight[0] = 1;
	param->weight[1] = 1;
	param->shrinking = 0;
	param->probability = 0;
	param->degree = 0; // necessary for kernels that do not use this parameter
	kernel->setLibSvmParams(param.get());
}

double TrainableSvmClassifier::computeSvmOutput(const struct svm_node *x) const {
	return utils.computeSvmOutput(model.get(), x);
}

void TrainableSvmClassifier::loadStaticNegatives(const string& negativesFilename, int maxNegatives, double scale) {
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
			for (unsigned int i = 0; i < values.size(); ++i) {
				data[i].index = i;
				data[i].value = scale * values[i];
			}
			data[values.size()].index = -1;
			staticNegativeExamples.push_back(move(data));
		}
	}
}

bool TrainableSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (newPositiveExamples.empty() && newNegativeExamples.empty()) // no new training data available -> no new training necessary
		return usable;
	addExamples(newPositiveExamples, newNegativeExamples);
	if (isRetrainingReasonable()) {
		train();
		usable = true;
	}
	return usable;
}

void TrainableSvmClassifier::reset() {
	usable = false;
	clearExamples();
}

void TrainableSvmClassifier::train() {
	param->weight[0] = getPositiveCount();
	param->weight[1] = getNegativeCount() + staticNegativeExamples.size();
	problem = move(createProblem());
	const char* message = svm_check_parameter(problem.get(), param.get());
	if (message != 0)
		throw invalid_argument(string("invalid SVM parameters: ") + message);
	model.reset(svm_train(problem.get(), param.get()));
	updateSvmParameters();
}

unique_ptr<struct svm_problem, ProblemDeleter> TrainableSvmClassifier::createProblem() {
	unique_ptr<struct svm_problem, ProblemDeleter> problem(new struct svm_problem);
	unsigned int count = getPositiveCount() + getNegativeCount() + staticNegativeExamples.size();
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = fillProblem(problem.get());
	for (auto example = staticNegativeExamples.cbegin(); example != staticNegativeExamples.cend(); ++example) {
		problem->y[i] = -1;
		problem->x[i] = example->get();
		++i;
	}
	return move(problem);
}

void TrainableSvmClassifier::updateSvmParameters() {
	svm->setSvmParameters(
			utils.extractSupportVectors(model.get()),
			utils.extractCoefficients(model.get()),
			utils.extractBias(model.get()));
}

} /* namespace classification */
