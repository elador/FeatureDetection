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
		svm(svm), constraintsViolationCosts(constraintsViolationCosts), usable(false), dimensions(0), staticNegativeExamples() {
	createParameters(svm->getKernel(), constraintsViolationCosts);
}

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts) :
		svm(make_shared<SvmClassifier>(kernel)), constraintsViolationCosts(constraintsViolationCosts),
		usable(false), dimensions(0), staticNegativeExamples() {
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
	vector<int> values;
	int value;
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
			// create nodes
			unique_ptr<struct svm_node[]> data(new struct svm_node[values.size() + 1]);
			for (unsigned int i = 0; i < values.size(); ++i) {
				data[i].index = i;
				data[i].value = scale * values[i];
			}
			data[values.size()].index = -1;
			staticNegativeExamples.push_back(move(data));
		}
	}
}

unique_ptr<struct svm_node[]> TrainableSvmClassifier::createNode(const Mat& vector) {
	unsigned int size = vector.total();
	unique_ptr<struct svm_node[]> node(new struct svm_node[size + 1]);
	if (vector.type() == CV_8U)
		fillNode<uchar>(node.get(), size, vector);
	else if (vector.type() == CV_32F)
		fillNode<float>(node.get(), size, vector);
	else if (vector.type() == CV_64F)
		fillNode<double>(node.get(), size, vector);
	else
		throw invalid_argument("TrainableSvmClassifier: vector has to be of type CV_8U, CV_32F, or CV_64F to create a node of");
	node[size].index = -1;
	return move(node);
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
//	return isRetrainingReasonable() && train();
}

void TrainableSvmClassifier::reset() {
	usable = false;
	clearExamples();
}

bool TrainableSvmClassifier::train() {
	// TODO prüfen, ob und inwiefern sinnvoll - auch mal vertauschen oder jeweils 1 reinschreiben und testen
	param->weight[0] = getPositiveCount();
	param->weight[1] = getNegativeCount() + staticNegativeExamples.size();
	problem = move(createProblem());
	const char* message = svm_check_parameter(problem.get(), param.get());
	if (message != 0) {// TODO exception? only logging? wenn exception, dann kann rückgabewert weg
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
		Mat supportVector(1, dimensions, CV_32F);
		float* values = supportVector.ptr<float>(0);
		const svm_node *sv = model->SV[i];
		for (int j = 0; j < dimensions; ++j) {
			if (sv->index == j) {
				values[j] = sv->value;
				++sv;
			} else {
				values[j] = 0;
			}
		}
		supportVectors.push_back(supportVector);
		coefficients.push_back(model->sv_coef[0][i]);
	}
	svm->setSvmParameters(supportVectors, coefficients, model->rho[0]);
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
