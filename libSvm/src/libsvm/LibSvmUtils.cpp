/*
 * LibSvmUtils.cpp
 *
 *  Created on: 12.09.2013
 *      Author: poschmann
 */

#include "libsvm/LibSvmUtils.hpp"
#include "libsvm/LibSvmKernelParamSetter.hpp"
#include "classification/Kernel.hpp"
#include "svm.h"
#include <stdexcept>

using classification::Kernel;
using cv::Mat;
using std::vector;
using std::unique_ptr;
using std::unordered_map;
using std::invalid_argument;

namespace libsvm {

LibSvmUtils::LibSvmUtils() :
		matRows(-1), matCols(-1), matType(CV_32FC1), matDepth(CV_32F), dimensions(0), node2example(), nodeDeleter(node2example) {}

LibSvmUtils::~LibSvmUtils() {}

NodeDeleter LibSvmUtils::getNodeDeleter() const {
	return nodeDeleter;
}

unique_ptr<struct svm_node[], NodeDeleter> LibSvmUtils::createNode(const Mat& vector) const {
	matRows = vector.rows;
	matCols = vector.cols;
	matType = vector.type();
	matDepth = vector.depth();
	dimensions = vector.total() * vector.channels();
	unique_ptr<struct svm_node[], NodeDeleter> node(new struct svm_node[dimensions + 1], nodeDeleter);
	if (matDepth == CV_8U)
		fillNode<uchar>(node.get(), vector, dimensions);
	else if (matDepth == CV_32F)
		fillNode<float>(node.get(), vector, dimensions);
	else if (matDepth == CV_64F)
		fillNode<double>(node.get(), vector, dimensions);
	else
		throw invalid_argument("LibSvmUtils: vector has to be of depth CV_8U, CV_32F, or CV_64F to create a node of");
	node[dimensions].index = -1;
	node2example.emplace(node.get(), vector);
	return move(node);
}

template<class T>
void LibSvmUtils::fillNode(struct svm_node *node, const Mat& vector, int size) const {
	if (!vector.isContinuous())
		throw invalid_argument("LibSvmUtils: vector has to be continuous");
	const T* values = vector.ptr<T>();
	for (int i = 0; i < size; ++i) {
		node[i].index = i + 1;
		node[i].value = values[i];
	}
}

Mat LibSvmUtils::getVector(const struct svm_node *node) const {
	Mat& vector = node2example[node];
	if (vector.empty()) {
		vector.create(matRows, matCols, matType);
		if (matDepth == CV_8U)
			fillMat<uchar>(vector, node, dimensions);
		else if (matDepth == CV_32F)
			fillMat<float>(vector, node, dimensions);
		else if (matDepth == CV_64F)
			fillMat<double>(vector, node, dimensions);
		else
			throw invalid_argument("LibSvmUtils: vectors have to be of depth CV_8U, CV_32F, or CV_64F");
	}
	return vector;
}

template<class T>
void LibSvmUtils::fillMat(Mat& vector, const struct svm_node *node, int size) const {
	T* values = vector.ptr<T>();
	for (int i = 0; i < size; ++i) {
		if (node->index == i + 1) {
			values[i] = static_cast<T>(node->value);
			++node;
		} else {
			values[i] = 0;
		}
	}
}

void LibSvmUtils::setKernelParams(const Kernel& kernel, struct svm_parameter *params) const {
	LibSvmKernelParamSetter paramSetter(params);
	kernel.accept(paramSetter);
}

double LibSvmUtils::computeSvmOutput(struct svm_model *model, const struct svm_node *x) const {
	double* dec_values = new double[1];
	svm_predict_values(model, x, dec_values);
	double svmOutput = dec_values[0];
	delete[] dec_values;
	return svmOutput;
}

vector<Mat> LibSvmUtils::extractSupportVectors(struct svm_model *model) const {
	if (model->param.kernel_type == LINEAR && (matDepth == CV_32F || matDepth == CV_64F)) {
		vector<Mat> supportVectors(1);
		supportVectors[0] = Mat::zeros(matRows, matCols, matType);
		if (matDepth == CV_32F) {
			float* values = supportVectors[0].ptr<float>();
			for (int i = 0; i < model->l; ++i) {
				double coeff = model->sv_coef[0][i];
				svm_node *node = model->SV[i];
				while (node->index != -1) {
					values[node->index - 1] += static_cast<float>(coeff * node->value);
					++node;
				}
			}
		} else if (matDepth == CV_64F) {
			double* values = supportVectors[0].ptr<double>();
			for (int i = 0; i < model->l; ++i) {
				double coeff = model->sv_coef[0][i];
				svm_node *node = model->SV[i];
				while (node->index != -1) {
					values[node->index - 1] += coeff * node->value;
					++node;
				}
			}
		}
		return supportVectors;
	}
	vector<Mat> supportVectors;
	supportVectors.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		supportVectors.push_back(getVector(model->SV[i]));
	return supportVectors;
}

vector<float> LibSvmUtils::extractCoefficients(struct svm_model *model) const {
	if (model->param.kernel_type == LINEAR && (matDepth == CV_32F || matDepth == CV_64F)) {
		vector<float> coefficients(1);
		coefficients[0] = 1;
		return coefficients;
	}
	vector<float> coefficients;
	coefficients.reserve(model->l);
	for (int i = 0; i < model->l; ++i)
		coefficients.push_back(model->sv_coef[0][i]);
	return coefficients;
}

double LibSvmUtils::extractBias(struct svm_model *model) const {
	return model->rho[0];
}

double LibSvmUtils::extractLogisticParamA(struct svm_model *model) const {
	return model->probA[0];
}

double LibSvmUtils::extractLogisticParamB(struct svm_model *model) const {
	return model->probB[0];
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

} /* namespace libsvm */
