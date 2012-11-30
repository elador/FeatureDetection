/*
 * LibSvmTraining.cpp
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#include "classification/LibSvmTraining.h"
#include "classification/LibSvmClassifier.h"
#include <fstream>
#include <sstream>

namespace classification {

LibSvmTraining::LibSvmTraining(shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) : staticNegativeTrainingSamples(),
				parameterBuilder(parameterBuilder), sigmoidParameterComputation(sigmoidParameterComputation) {}

LibSvmTraining::~LibSvmTraining() {
	freeSamples(staticNegativeTrainingSamples);
}

void LibSvmTraining::freeSamples(std::vector<struct svm_node *>& samples) {
	for (std::vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit)
		delete[] (*sit);
	samples.clear();
}

struct svm_parameter *LibSvmTraining::createParameters(unsigned int positiveCount, unsigned int negativeCount) {
	return parameterBuilder->createParameters(positiveCount, negativeCount);
}

void LibSvmTraining::changeSvmParameters(LibSvmClassifier& svm, int dimensions, struct svm_model *model,
		struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount) {
	std::pair<double, double> sigmoidParams = sigmoidParameterComputation->computeSigmoidParameters(
			model, problem->x, positiveCount, problem->x + positiveCount, negativeCount);
	svm.setModel(dimensions, model, sigmoidParams.first, sigmoidParams.second);
}

void LibSvmTraining::readStaticNegatives(const std::string negativesFilename, int maxNegatives) {
	staticNegativeTrainingSamples.reserve(maxNegatives);
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
			staticNegativeTrainingSamples.push_back(data);
		}
	}
}

} /* namespace tracking */
