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
		svm(svm), constraintsViolationCosts(constraintsViolationCosts),
		usable(false), dimensions(0), node2example(), nodeDeleter(node2example), staticNegativeExamples(), matType(CV_32F) {
	createParameters(svm->getKernel(), constraintsViolationCosts);
}

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts) :
		svm(make_shared<SvmClassifier>(kernel)), constraintsViolationCosts(constraintsViolationCosts),
		usable(false), dimensions(0), node2example(), nodeDeleter(node2example), staticNegativeExamples(), matType(CV_32F) {
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
	param->weight_label = (int*)malloc(2 * sizeof(int));
	param->weight_label[0] = +1;
	param->weight_label[1] = -1;
	param->weight = (double*)malloc(2 * sizeof(double));
	param->weight[0] = 1;
	param->weight[1] = 1;
	param->shrinking = 0;
	param->probability = 0;
	param->degree = 0; // necessary for kernels that do not use this parameter
	kernel->setLibSvmParams(param.get());
}

pair<double, double> TrainableSvmClassifier::computeMeanSvmOutputs() {
	double positiveOutputSum = 0;
	double negativeOutputSum = 0;
	unsigned int positiveCount = getPositiveCount();
	unsigned int negativeCount = problem->l - positiveCount;
	unsigned int totalCount = problem->l;
	unsigned int i = 0;
	for (; i < positiveCount; ++i)
		positiveOutputSum += computeSvmOutput(problem->x[i]);
	for (; i < totalCount; ++i)
		negativeOutputSum += computeSvmOutput(problem->x[i]);
	return make_pair(positiveOutputSum / positiveCount, negativeOutputSum / negativeCount);
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
			unique_ptr<struct svm_node[], NodeDeleter> data(new struct svm_node[values.size() + 1], nodeDeleter);
			for (unsigned int i = 0; i < values.size(); ++i) {
				data[i].index = i;
				data[i].value = scale * values[i];
			}
			data[values.size()].index = -1;
			staticNegativeExamples.push_back(move(data));
		}
	}
}

unique_ptr<struct svm_node[], NodeDeleter> TrainableSvmClassifier::createNode(const Mat& vector) {
	unsigned int size = vector.total();
	unique_ptr<struct svm_node[], NodeDeleter> node(new struct svm_node[size + 1], nodeDeleter);
	matType = vector.type();
	if (matType == CV_8U)
		fillNode<uchar>(node.get(), size, vector);
	else if (matType == CV_32F)
		fillNode<float>(node.get(), size, vector);
	else if (matType == CV_64F)
		fillNode<double>(node.get(), size, vector);
	else
		throw invalid_argument("TrainableSvmClassifier: vector has to be of type CV_8U, CV_32F, or CV_64F to create a node of");
	node[size].index = -1;
	node2example.insert(make_pair(node.get(), vector));
	return move(node);
}

double TrainableSvmClassifier::computeSvmOutput(const struct svm_node *x) {
	double* dec_values = new double[1]; // TODO einfluss, wenn das zwischengespeichert wird?
	svm_predict_values(model.get(), x, dec_values);
	double svmOutput = dec_values[0];
	delete[] dec_values;
	return svmOutput;
}

bool TrainableSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (!newPositiveExamples.empty())
		dimensions = newPositiveExamples[0].total();
	if (newPositiveExamples.empty() && newNegativeExamples.empty()) // no new training data available -> no new training necessary
		return usable;// TODO die abfrage nur dann sinnvoll, wenn dieser fall erwartet wird (eigentlich müsste ich in, wenn kein objekt gefunden wurde, gar nicht erst retrain aufrufen -> classifier sollte dann nach möglichkeit eh nix machen)
	addExamples(newPositiveExamples, newNegativeExamples);
	if (isRetrainingReasonable())
		usable = train();
	return usable;
//	return isRetrainingReasonable() && train(); TODO raus
}

void TrainableSvmClassifier::reset() {
	usable = false;
	clearExamples();
}

bool TrainableSvmClassifier::train() {
	param->weight[0] = getPositiveCount();
	param->weight[1] = getNegativeCount() + staticNegativeExamples.size();
	problem = move(createProblem());
	const char* message = svm_check_parameter(problem.get(), param.get());
	if (message != 0) {// TODO exception? just logging? wenn exception, dann kann rückgabewert weg
		std::cerr << "invalid SVM parameters: " << message << std::endl;
		return false;
	}
	model.reset(svm_train(problem.get(), param.get()));
	updateSvmParameters();
	return true;
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
	vector<Mat> supportVectors;
	vector<float> coefficients;
	supportVectors.reserve(model->l);
	coefficients.reserve(model->l);
	for (int i = 0; i < model->l; ++i) {
		supportVectors.push_back(getSupportVector(model->SV[i]));
		coefficients.push_back(model->sv_coef[0][i]);
	}
	svm->setSvmParameters(supportVectors, coefficients, model->rho[0]);
}

Mat TrainableSvmClassifier::getSupportVector(const struct svm_node *node) {
	Mat& vector = node2example[node];
	if (vector.empty()) {
		vector.create(1, dimensions, matType);
		if (matType == CV_8U)
			fillMat<uchar>(vector, node);
		else if (matType == CV_32F)
			fillMat<float>(vector, node);
		else if (matType == CV_64F)
			fillMat<double>(vector, node);
		else
			throw invalid_argument("TrainableSvmClassifier: vectors have to be of type CV_8U, CV_32F, or CV_64F");
	}
	return vector;
}

template<class T>
void TrainableSvmClassifier::fillNode(struct svm_node *node, unsigned int size, const Mat& vector) {
	// TODO has to be continuous
	const T* values = vector.ptr<T>(0);
	for (unsigned int i = 0; i < size; ++i) {
		node[i].index = i;
		node[i].value = values[i];
	}
}

template<class T>
void TrainableSvmClassifier::fillMat(Mat& vector, const struct svm_node *node) {
	T* values = vector.ptr<T>(0);
	for (int i = 0; i < dimensions; ++i) {
		if (node->index == i) {
			values[i] = static_cast<T>(node->value);
			++node;
		} else {
			values[i] = 0;
		}
	}
}

NodeDeleter::NodeDeleter(unordered_map<const struct svm_node*, Mat>& map) : map(map) {}

NodeDeleter::NodeDeleter(const NodeDeleter& other) : map(other.map) {}

NodeDeleter& NodeDeleter::operator=(const NodeDeleter& other) {
	map = other.map;
	return *this;
}

void NodeDeleter::operator()(struct svm_node *node) const {
	map.erase(node);
	delete[] node;
}

void ParameterDeleter::operator()(struct svm_parameter *param) const {
	svm_destroy_param(param);
	delete param;
}

void ProblemDeleter::operator()(struct svm_problem *problem) const {
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
}

void ModelDeleter::operator()(struct svm_model *model) const {
	svm_free_and_destroy_model(&model);
}

} /* namespace classification */
