/*
 * LibSvmTraining.cpp
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#include "tracking/LibSvmTraining.h"
#include "tracking/ChangableDetectorSvm.h"
#include <fstream>
#include <sstream>

namespace tracking {

LibSvmTraining::LibSvmTraining(shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
		staticNegativeSamples(), sigmoidParameterComputation(sigmoidParameterComputation) {}

LibSvmTraining::~LibSvmTraining() {
	freeSamples(staticNegativeSamples);
}

void LibSvmTraining::freeSamples(std::vector<struct svm_node *>& samples) {
	for (std::vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit)
		delete[] (*sit);
	samples.clear();
}

void LibSvmTraining::changeSvmParameters(ChangableDetectorSvm& svm, struct svm_model *model,
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
	svm.changeRbfParameters(model->l, supportVectors, alphas, rho, svm.getThreshold(), gamma,
			sigmoidParams.first, sigmoidParams.second);
}

void LibSvmTraining::readStaticNegatives(const std::string negativesFilename, int maxNegatives) {
	staticNegativeSamples.reserve(maxNegatives);
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

} /* namespace tracking */
