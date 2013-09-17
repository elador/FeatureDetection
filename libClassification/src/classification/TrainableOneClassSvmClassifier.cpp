/*
 * TrainableOneClassSvmClassifier.cpp
 *
 *  Created on: 11.09.2013
 *      Author: poschmann
 */

#include "classification/TrainableOneClassSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/Kernel.hpp"
#include "svm.h"
#include <stdexcept>

using std::invalid_argument;
using std::make_shared;
using std::make_pair;
using std::move;
using std::string;

namespace classification {

TrainableOneClassSvmClassifier::TrainableOneClassSvmClassifier(shared_ptr<SvmClassifier> svm, double nu, int minExamples, int maxExamples) :
		utils(), svm(svm), usable(false), param(), problem(), model(), examples(), minExamples(minExamples) {
	examples.reserve(maxExamples);
	createParameters(svm->getKernel(), nu);
}

TrainableOneClassSvmClassifier::TrainableOneClassSvmClassifier(const shared_ptr<Kernel> kernel, double nu, int minExamples, int maxExamples) :
		utils(), svm(make_shared<SvmClassifier>(kernel)), usable(false), param(), problem(), model(), examples(), minExamples(minExamples) {
	examples.reserve(maxExamples);
	createParameters(kernel, nu);
}

TrainableOneClassSvmClassifier::~TrainableOneClassSvmClassifier() {}

void TrainableOneClassSvmClassifier::createParameters(const shared_ptr<Kernel> kernel, double nu) {
	param.reset(new struct svm_parameter);
	param->svm_type = ONE_CLASS;
	param->nu = nu;
	param->cache_size = 100;
	param->eps = 1e-3;
	param->nr_weight = 0;
	param->weight_label = (int*)malloc(param->nr_weight * sizeof(int));
	param->weight = (double*)malloc(param->nr_weight * sizeof(double));
	param->shrinking = 0;
	param->probability = 0;
	param->degree = 0; // necessary for kernels that do not use this parameter
	kernel->setLibSvmParams(param.get());
}

bool TrainableOneClassSvmClassifier::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

bool TrainableOneClassSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (newPositiveExamples.empty()) // no new training data available -> no new training necessary
		return usable;
	addExamples(newPositiveExamples);
	if (isRetrainingReasonable()) {
		train();
		usable = true;
	}
	return usable;
}

void TrainableOneClassSvmClassifier::addExamples(const vector<Mat>& newExamples) {
	// compute hyperplane distances of current positive examples and sort in descending order
	vector<pair<unsigned int, double>> distances;
	distances.reserve(examples.size());
	if (isUsable()) {
		for (unsigned int i = 1; i < examples.size(); ++i)
			distances.push_back(make_pair(i, utils.computeSvmOutput(model.get(), examples[i].get())));
		sort(distances.begin(), distances.end(), [](pair<unsigned int, double> a, pair<unsigned int, double> b) {
			return a.second > b.second;
		});
	} else {
		for (unsigned int i = 1; i < examples.size(); ++i)
			distances.push_back(make_pair(i, 0.5));
	}
	// add new positive examples as long as no examples have to be removed
	auto example = newExamples.cbegin();
	for (; examples.size() < examples.capacity() && example != newExamples.cend(); ++example)
		examples.push_back(move(utils.createNode(*example)));
	// replace existing examples (beginning with the high distance ones) with new examples
	auto distanceIndex = distances.cbegin();
	for (; example != newExamples.cend() && distanceIndex != distances.cend(); ++example, ++distanceIndex)
		examples[distanceIndex->first] = move(utils.createNode(*example));
}

bool TrainableOneClassSvmClassifier::isRetrainingReasonable() const {
	return examples.size() >= minExamples;
}

void TrainableOneClassSvmClassifier::train() {
	problem = move(createProblem());
	const char* message = svm_check_parameter(problem.get(), param.get());
	if (message != 0)
		throw invalid_argument(string("invalid SVM parameters: ") + message);
	model.reset(svm_train(problem.get(), param.get()));
	updateSvmParameters();
}

unique_ptr<struct svm_problem, ProblemDeleter> TrainableOneClassSvmClassifier::createProblem() {
	unique_ptr<struct svm_problem, ProblemDeleter> problem(new struct svm_problem);
	unsigned int count = examples.size();
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	fillProblem(problem.get());
	return move(problem);
}

void TrainableOneClassSvmClassifier::fillProblem(struct svm_problem *problem) const {
	for (size_t i = 0; i < examples.size(); ++i) {
		problem->y[i] = 1;
		problem->x[i] = examples[i].get();
	}
}

void TrainableOneClassSvmClassifier::updateSvmParameters() {
	svm->setSvmParameters(
			utils.extractSupportVectors(model.get()),
			utils.extractCoefficients(model.get()),
			utils.extractBias(model.get()));
}

void TrainableOneClassSvmClassifier::reset() {
	usable = false;
	examples.clear();
}

} /* namespace classification */
