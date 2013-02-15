/*
 * LibSvmTraining.cpp
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#include "classification/LibSvmTraining.h"
#include "classification/LibSvmClassifier.h"
#include "classification/FeatureVector.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace classification {

LibSvmTraining::LibSvmTraining(shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) : staticNegativeExamples(),
				parameterBuilder(parameterBuilder), sigmoidParameterComputation(sigmoidParameterComputation) {}

LibSvmTraining::~LibSvmTraining() {
	freeExamples(staticNegativeExamples);
}

void LibSvmTraining::freeExamples(vector<struct svm_node *>& examples) {
	for (vector<struct svm_node *>::iterator sit = examples.begin(); sit < examples.end(); ++sit)
		delete[] (*sit);
	examples.clear();
}

void LibSvmTraining::readStaticNegatives(const std::string negativesFilename, int maxNegatives) {
	staticNegativeExamples.reserve(maxNegatives);
	int negatives = 0;
	vector<int> values;
	int value;
	char separator;
	std::string line;
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
			// create nodes
			struct svm_node* data = new struct svm_node[values.size() + 1];
			for (unsigned int i = 0; i < values.size(); ++i) {
				data[i].index = i;
				data[i].value = values[i] / 255.0;
			}
			data[values.size()].index = -1;
			staticNegativeExamples.push_back(data);
		}
	}
}

struct svm_node *LibSvmTraining::createNode(const FeatureVector& vector) {
	struct svm_node* node = new struct svm_node[vector.getSize() + 1];
	for (unsigned int i = 0; i < vector.getSize(); ++i) {
		node[i].index = i;
		node[i].value = vector[i];
	}
	node[vector.getSize()].index = -1;
	return node;
}

double LibSvmTraining::computeSvmOutput(const struct svm_model *model, const struct svm_node *x) {
	double* dec_values = new double[1];
	svm_predict_values(model, x, dec_values);
	double svmOutput = dec_values[0];
	delete[] dec_values;
	return svmOutput;
}

bool LibSvmTraining::train(LibSvmClassifier& svm, unsigned int dimensions,
		vector<struct svm_node *>& positiveExamples, vector<struct svm_node *>& negativeExamples) {
	struct svm_parameter *param = createParameters(
			positiveExamples.size(), negativeExamples.size() + staticNegativeExamples.size());
	struct svm_problem *problem = createProblem(positiveExamples, negativeExamples, staticNegativeExamples);
	const char* message = svm_check_parameter(problem, param);
	if (message != 0) {
		std::cerr << "invalid SVM parameters: " << message << std::endl;
		svm_destroy_param(param);
		delete param;
		delete[] problem->x;
		delete[] problem->y;
		delete problem;
		return false;
	}
	struct svm_model *model = svm_train(problem, param);
	unsigned int negativeCount = negativeExamples.size() + staticNegativeExamples.size();
	changeSvmParameters(svm, dimensions, model, problem, positiveExamples.size(), negativeCount);
	svm_destroy_param(param);
	delete param;
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
	return true;
}

struct svm_parameter *LibSvmTraining::createParameters(unsigned int positiveCount, unsigned int negativeCount) {
	return parameterBuilder->createParameters(positiveCount, negativeCount);
}

struct svm_problem *LibSvmTraining::createProblem(vector<struct svm_node *>& positiveExamples,
		vector<struct svm_node *>& negativeExamples, vector<struct svm_node *>& staticNegativeExamples) {
	struct svm_problem *problem = new struct svm_problem;
	unsigned int count = positiveExamples.size() + negativeExamples.size() + staticNegativeExamples.size();
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = 0;
	vector<struct svm_node *>::iterator sit;
	for (sit = positiveExamples.begin(); sit < positiveExamples.end(); ++sit) {
		problem->y[i] = 1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = negativeExamples.begin(); sit < negativeExamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = staticNegativeExamples.begin(); sit < staticNegativeExamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	return problem;
}

void LibSvmTraining::changeSvmParameters(LibSvmClassifier& svm, unsigned int dimensions, struct svm_model *model,
		struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount) {
	pair<double, double> sigmoidParams = sigmoidParameterComputation->computeSigmoidParameters(
			model, problem->x, positiveCount, problem->x + positiveCount, negativeCount);
	svm.setModel(dimensions, model, sigmoidParams.first, sigmoidParams.second);
}

} /* namespace classification */
