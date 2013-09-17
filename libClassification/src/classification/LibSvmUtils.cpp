/*
 * LibSvmUtils.cpp
 *
 *  Created on: 12.09.2013
 *      Author: poschmann
 */

#include "classification/LibSvmUtils.hpp"
#include "svm.h"
#include <stdexcept>

using std::invalid_argument;

namespace classification {

LibSvmUtils::LibSvmUtils() :
		matType(CV_32F), dimensions(0), node2example(), nodeDeleter(node2example) {}

LibSvmUtils::~LibSvmUtils() {}

NodeDeleter LibSvmUtils::getNodeDeleter() const {
	return nodeDeleter;
}

unique_ptr<struct svm_node[], NodeDeleter> LibSvmUtils::createNode(const Mat& vector) const {
	matType = vector.type();
	dimensions = vector.total();
	unique_ptr<struct svm_node[], NodeDeleter> node(new struct svm_node[dimensions + 1], nodeDeleter);
	if (matType == CV_8U)
		fillNode<uchar>(node.get(), vector, dimensions);
	else if (matType == CV_32F)
		fillNode<float>(node.get(), vector, dimensions);
	else if (matType == CV_64F)
		fillNode<double>(node.get(), vector, dimensions);
	else
		throw invalid_argument("TrainableSvmClassifier: vector has to be of type CV_8U, CV_32F, or CV_64F to create a node of");
	node[dimensions].index = -1;
	node2example.emplace(node.get(), vector);
	return move(node);
}

Mat LibSvmUtils::getVector(const struct svm_node *node) const {
	Mat& vector = node2example[node];
	if (vector.empty()) {
		vector.create(1, dimensions, matType);
		if (matType == CV_8U)
			fillMat<uchar>(vector, node, dimensions);
		else if (matType == CV_32F)
			fillMat<float>(vector, node, dimensions);
		else if (matType == CV_64F)
			fillMat<double>(vector, node, dimensions);
		else
			throw invalid_argument("TrainableSvmClassifier: vectors have to be of type CV_8U, CV_32F, or CV_64F");
	}
	return vector;
}

double LibSvmUtils::computeSvmOutput(struct svm_model *model, const struct svm_node *x) const {
	double* dec_values = new double[1];
	svm_predict_values(model, x, dec_values);
	double svmOutput = dec_values[0];
	delete[] dec_values;
	return svmOutput;
}

vector<Mat> LibSvmUtils::extractSupportVectors(struct svm_model *model) const {
	vector<Mat> supportVectors;
	supportVectors.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		supportVectors.push_back(getVector(model->SV[i]));
	return supportVectors;
}

vector<float> LibSvmUtils::extractCoefficients(struct svm_model *model) const {
	vector<float> coefficients;
	coefficients.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		coefficients.push_back(model->sv_coef[0][i]);
	return coefficients;
}

double LibSvmUtils::extractBias(struct svm_model *model) const {
	return model->rho[0];
}

template<class T>
void LibSvmUtils::fillNode(struct svm_node *node, const Mat& vector, int size) const {
	if (!vector.isContinuous())
		throw invalid_argument("TrainableSvmClassifier: vector has to be continuous");
	const T* values = vector.ptr<T>();
	for (int i = 0; i < size; ++i) {
		node[i].index = i;
		node[i].value = values[i];
	}
}

template<class T>
void LibSvmUtils::fillMat(Mat& vector, const struct svm_node *node, int size) const {
	T* values = vector.ptr<T>();
	for (int i = 0; i < size; ++i) {
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
