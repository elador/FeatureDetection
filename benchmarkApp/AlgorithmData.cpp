/*
 * AlgorithmData.cpp
 *
 *  Created on: 26.09.2013
 *      Author: poschmann
 */

#include "AlgorithmData.hpp"

AlgorithmData::AlgorithmData(
		string name, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
		float confidenceThreshold, size_t negatives, size_t initialNegatives) :
				name(name),
				extractor(extractor),
				classifier(classifier),
				confidenceThreshold(confidenceThreshold),
				negatives(negatives),
				initialNegatives(initialNegatives) {}

AlgorithmData::~AlgorithmData() {}
