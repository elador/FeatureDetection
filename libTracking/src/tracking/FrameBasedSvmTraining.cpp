/*
 * FrameBasedSvmTraining.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/ChangableDetectorSvm.h"
#include "FdPatch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>

namespace tracking {

FrameBasedSvmTraining::FrameBasedSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename,
		int negatives, shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) : frameLength(frameLength),
				minAvgSamples(minAvgSamples), staticNegativeSamples(), positiveSamples(frameLength),
				negativeSamples(frameLength), oldestEntry(0), sigmoidParameterComputation(sigmoidParameterComputation) {
	staticNegativeSamples.reserve(negatives);
	readStaticNegatives(negativesFilename, negatives);
}

FrameBasedSvmTraining::~FrameBasedSvmTraining() {
	std::vector<std::vector<struct svm_node *> >::iterator sit;
	for (sit = positiveSamples.begin(); sit < positiveSamples.end(); ++sit)
		freeSamples(*sit);
	for (sit = negativeSamples.begin(); sit < negativeSamples.end(); ++sit)
		freeSamples(*sit);
	freeSamples(staticNegativeSamples);
}

void FrameBasedSvmTraining::readStaticNegatives(const std::string negativesFilename, int maxNegatives) {
	int negatives = 0;
	std::vector<int> values;
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
			staticNegativeSamples.push_back(data);
		}
	}
}

int FrameBasedSvmTraining::getRequiredPositiveSampleCount() const {
	return (int)ceil(minAvgSamples * frameLength);
}

int FrameBasedSvmTraining::getPositiveSampleCount() const {
	return getSampleCount(positiveSamples);
}

int FrameBasedSvmTraining::getNegativeSampleCount() const {
	return getSampleCount(negativeSamples) + staticNegativeSamples.size();
}

bool FrameBasedSvmTraining::isTrainingReasonable() const {
	return getPositiveSampleCount() >= getRequiredPositiveSampleCount();
}

bool FrameBasedSvmTraining::retrain(ChangableDetectorSvm& svm, const std::vector<FdPatch*>& positivePatches,
		const std::vector<FdPatch*>& negativePatches) {
	addSamples(positivePatches, negativePatches);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FrameBasedSvmTraining::freeSamples(std::vector<struct svm_node *>& samples) {
	for (std::vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit)
		delete[] (*sit);
	samples.clear();
}

void FrameBasedSvmTraining::addSamples(const std::vector<FdPatch*>& positivePatches,
		const std::vector<FdPatch*>& negativePatches) {
	replaceSamples(positiveSamples[oldestEntry], positivePatches);
	replaceSamples(negativeSamples[oldestEntry], negativePatches);
	++oldestEntry;
	if (oldestEntry >= frameLength)
		oldestEntry = 0;
}

void FrameBasedSvmTraining::replaceSamples(std::vector<struct svm_node *>& samples,
		const std::vector<FdPatch*>& patches) {
	freeSamples(samples);
	for (std::vector<FdPatch*>::const_iterator pit = patches.begin(); pit < patches.end(); ++pit) {
		FdPatch* patch = *pit;
		unsigned int dataLength = patch->w * patch->h;
		struct svm_node* data = new struct svm_node[dataLength + 1];
		for (unsigned int i = 0; i < dataLength; ++i) {
			data[i].index = i;
			data[i].value = patch->data[i] / 255.0;
		}
		data[dataLength].index = -1;
		samples.push_back(data);
	}
}

int FrameBasedSvmTraining::getSampleCount(const std::vector<std::vector<struct svm_node *> >& samples) const {
	int count = 0;
	std::vector<std::vector<struct svm_node *> >::const_iterator it;
	for (it = samples.begin(); it < samples.end(); ++it)
		count += it->size();
	return count;
}

bool FrameBasedSvmTraining::train(ChangableDetectorSvm& svm) {
	unsigned int positiveCount = getPositiveSampleCount();
	unsigned int negativeCount = getNegativeSampleCount();
	unsigned int count = positiveCount + negativeCount;

	struct svm_parameter *param = createParameters(positiveCount, negativeCount);
	struct svm_problem *problem = createProblem(count);
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
	changeSvmParameters(svm, model, problem, positiveCount, negativeCount);
	svm_free_and_destroy_model(&model);
	svm_destroy_param(param);
	delete param;
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
	return true;
}

struct svm_parameter *FrameBasedSvmTraining::createParameters(unsigned int positiveCount, unsigned int negativeCount) {
	struct svm_parameter *param = new struct svm_parameter;
	param->svm_type = C_SVC;
	param->kernel_type = RBF;
	param->degree = 0;
	param->gamma = 0.05;
	param->cache_size = 100;
	param->eps = 1e-3;
	param->C = 1;
	param->nr_weight = 2;
	param->weight_label = (int*)malloc(2 * sizeof(int));
	param->weight_label[0] = +1;
	param->weight_label[1] = -1;
	param->weight = (double*)malloc(2 * sizeof(double));
	param->weight[0] = positiveCount;
	param->weight[1] = negativeCount;
	param->shrinking = 0;
	param->probability = 0;
	return param;
}

struct svm_problem *FrameBasedSvmTraining::createProblem(unsigned int count) {
	struct svm_problem *problem = new struct svm_problem;
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = 0;
	std::vector<std::vector<struct svm_node *> >::iterator sit;
	for (sit = positiveSamples.begin(); sit < positiveSamples.end(); ++sit) {
		std::vector<struct svm_node *>::iterator dit;
		for (dit = sit->begin(); dit < sit->end(); ++dit) {
			problem->y[i] = 1;
			problem->x[i] = *dit;
			i++;
		}
	}
	for (sit = negativeSamples.begin(); sit < negativeSamples.end(); ++sit) {
		std::vector<struct svm_node *>::iterator dit;
		for (dit = sit->begin(); dit < sit->end(); ++dit) {
			problem->y[i] = -1;
			problem->x[i] = *dit;
			i++;
		}
	}
	std::vector<struct svm_node *>::iterator dit;
	for (dit = staticNegativeSamples.begin(); dit < staticNegativeSamples.end(); ++dit) {
		problem->y[i] = -1;
		problem->x[i] = *dit;
		i++;
	}
	return problem;
}

void FrameBasedSvmTraining::changeSvmParameters(ChangableDetectorSvm& svm, struct svm_model *model,
		struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount) {
	int dimensions = svm.getDimensions();
	unsigned char** supportVectors = new unsigned char*[model->l];
	float* alphas = new float[model->l];
	for (int i = 0; i < model->l; ++i) {
		supportVectors[i] = new unsigned char[dimensions];
		const struct svm_node *svit = model->SV[i];
		for (int j = 0; j < dimensions; ++j) {
			if (j == svit->index) {
				supportVectors[i][j] = 255 * svit->value; // because the SVM operates on gray-scale values between 0 and 255
				++svit;
			} else {
				supportVectors[i][j] = 0;
			}
		}
		alphas[i] = model->sv_coef[0][i];
	}
	double rho = model->rho[0];
	double gamma = model->param.gamma / (255 * 255); // because the support vectors were multiplied by 255
	std::pair<double, double> sigmoidParams = sigmoidParameterComputation->computeSigmoidParameters(svm, model,
			problem->x, positiveCount, problem->x + positiveCount, negativeCount);
	svm.changeRbfParameters(model->l, supportVectors, alphas, rho, gamma, sigmoidParams.first, sigmoidParams.second);
}

} /* namespace tracking */
