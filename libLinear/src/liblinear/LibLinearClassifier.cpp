/*
 * LibLinearClassifier.cpp
 *
 *  Created on: 03.12.2013
 *      Author: poschmann
 */

#include "liblinear/LibLinearClassifier.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/EmptyExampleManagement.hpp"
#include <fstream>
#include <stdexcept>

using classification::LinearKernel;
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

namespace liblinear {

LibLinearClassifier::LibLinearClassifier(double c, bool bias) :
		TrainableSvmClassifier(make_shared<LinearKernel>()), utils(), bias(bias), param(), staticNegativeExamples(),
		positiveExamples(new UnlimitedExampleManagement()), negativeExamples(new UnlimitedExampleManagement()) {
	param.reset(new struct parameter);
	param->solver_type = L2R_L2LOSS_SVC; // possible values: L2R_L2LOSS_SVC   L2R_L2LOSS_SVC_DUAL   L2R_L1LOSS_SVC_DUAL   MCSVM_CS   L1R_L2LOSS_SVC
	if (param->solver_type == L2R_L2LOSS_SVC)
		param->eps = 0.01;
	else
		param->eps = 0.1;
	param->C = c;
	param->p = 0;
	param->nr_weight = 0;
	param->weight_label = nullptr;
	param->weight = nullptr;
}

void LibLinearClassifier::loadStaticNegatives(const string& negativesFilename, int maxNegatives, double scale) {
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
			unique_ptr<struct feature_node[]> data(new struct feature_node[values.size() + 1]);
			for (size_t i = 0; i < values.size(); ++i) {
				data[i].index = i + 1;
				data[i].value = scale * values[i];
			}
			if (bias) {
				data[values.size()].index = values.size() + 1;
				data[values.size()].value = 1;
				data[values.size() + 1].index = -1;
			} else {
				data[values.size()].index = -1;
			}
			staticNegativeExamples.push_back(move(data));
		}
	}
}

bool LibLinearClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (newPositiveExamples.empty() && newNegativeExamples.empty()) // no new training data available -> no new training necessary
		return usable;
	positiveExamples->add(newPositiveExamples);
	negativeExamples->add(newNegativeExamples);
	if (positiveExamples->hasRequiredSize() && negativeExamples->hasRequiredSize())
		usable = train();
	return usable;
}

bool LibLinearClassifier::train() {
	vector<unique_ptr<struct feature_node[]>> positiveExamples = move(createNodes(this->positiveExamples.get()));
	vector<unique_ptr<struct feature_node[]>> negativeExamples = move(createNodes(this->negativeExamples.get()));
	unique_ptr<struct problem, ProblemDeleter> problem = move(createProblem(
			positiveExamples, negativeExamples, staticNegativeExamples, utils.getDimensions()));
	const char* message = check_parameter(problem.get(), param.get());
	if (message != 0)
		throw invalid_argument(string("LibLinearClassifier: invalid SVM parameters: ") + message);
	unique_ptr<struct model, ModelDeleter> model(::train(problem.get(), param.get()));
	svm->setSvmParameters(
			utils.extractSupportVectors(model.get()),
			utils.extractCoefficients(model.get()),
			utils.extractBias(model.get()));
	return true;
}

vector<unique_ptr<struct feature_node[]>> LibLinearClassifier::createNodes(ExampleManagement* examples) {
	vector<unique_ptr<struct feature_node[]>> nodes;
	nodes.reserve(examples->size());
	for (auto iterator = examples->iterator(); iterator->hasNext();)
		nodes.push_back(move(utils.createNode(iterator->next(), bias)));
	return move(nodes);
}

unique_ptr<struct problem, ProblemDeleter> LibLinearClassifier::createProblem(
		const vector<unique_ptr<struct feature_node[]>>& positiveExamples,
		const vector<unique_ptr<struct feature_node[]>>& negativeExamples,
		const vector<unique_ptr<struct feature_node[]>>& staticNegativeExamples,
		size_t dimensions) {
	unique_ptr<struct problem, ProblemDeleter> problem(new struct problem);
	problem->l = positiveExamples.size() + negativeExamples.size() + staticNegativeExamples.size();
	problem->n = bias ? dimensions + 1 : dimensions;
	problem->y = new double[problem->l];
	problem->x = new struct feature_node *[problem->l];
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
	problem->bias = bias ? 1 : -1;
	return move(problem);
}

void LibLinearClassifier::reset() {
	usable = false;
	svm->setSvmParameters(vector<Mat>(), vector<float>(), 0.0);
	positiveExamples->clear();
	negativeExamples->clear();
}

} /* namespace liblinear */
