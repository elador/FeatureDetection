/*
 * AlgorithmData.hpp
 *
 *  Created on: 26.09.2013
 *      Author: poschmann
 */

#ifndef ALGORITHMDATA_HPP_
#define ALGORITHMDATA_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include <string>
#include <memory>

using imageprocessing::FeatureExtractor;
using classification::TrainableProbabilisticClassifier;
using std::string;
using std::shared_ptr;

/**
 * Algorithm data for benchmarking.
 */
class AlgorithmData {
public:

	/**
	 * Constructs new algorithm data.
	 *
	 * @param[in] name Name of the algorithm data.
	 * @param[in] extractor Feature extractor.
	 * @param[in] classifier Trainable classifier.
	 * @param[in] confidenceThreshold Probability based confidence threshold for skipping learning on patches.
	 * @param[in] negatives Maximum amount of negative training examples per frame.
	 * @param[in] initialNegatives Amount of negative training examples in the first frame.
	 */
	AlgorithmData(string name, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
			float confidenceThreshold, size_t negatives, size_t initialNegatives);

	~AlgorithmData();

	string name;
	shared_ptr<FeatureExtractor> extractor;
	shared_ptr<TrainableProbabilisticClassifier> classifier;
	float confidenceThreshold;
	size_t negatives;
	size_t initialNegatives;
};

#endif /* ALGORITHMDATA_HPP_ */
