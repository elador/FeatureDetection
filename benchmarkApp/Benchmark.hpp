/*
 * Benchmark.hpp
 *
 *  Created on: 10.09.2013
 *      Author: poschmann
 */

#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include "AlgorithmData.hpp"
#include "imageio/LabeledImageSource.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
//#include "classification/TrainableOneClassSvmClassifier.hpp"
#include <memory>
#include <ostream>

using imageio::LabeledImageSource;
using imageprocessing::FeatureExtractor;
using classification::TrainableProbabilisticClassifier;
//using classification::TrainableOneClassSvmClassifier;
using cv::Rect_;
using std::vector;
using std::shared_ptr;
using std::ostream;

class Benchmark {
public:

	Benchmark(float sizeMin, float sizeMax, float sizeScale, float step, float allowedOverlap = 0.5f, string outputDir = "results");

	~Benchmark();

	/**
	 * Adds a new combination of a feature extractor and a classifier for benchmarking.
	 *
	 * @param[in] name Name of the feature extractor classifier combination.
	 * @param[in] extractor Feature extractor.
	 * @param[in] classifier Trainable classifier.
	 * @param[in] confidenceThreshold Probability based confidence threshold for skipping learning on patches.
	 * @param[in] negatives Maximum amount of negative training examples per frame.
	 * @param[in] initialNegatives Amount of negative training examples in the first frame.
	 */
	void add(string name, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
			float confidenceThreshold, size_t negatives, size_t initialNegatives);

	/**
	 * Runs the benchmark using the given image and ground truth source, evaluating the previously added feature extractor
	 * classifier combinations. For each extractor-classifier-combo there will be a text file that is named after the combo
	 * and this benchmark, containing the collected evaluated data.
	 *
	 * @param[in] name Name of the benchmark data.
	 * @param[in] source Source of image and ground truth data.
	 */
	void run(string name, shared_ptr<LabeledImageSource> source) const;

	/**
	 * TODO
	 *
	 * @param[in] source Source of image and annotations (ground truth).
	 * @param[in] extractor TODO
	 * @param[in] classifier TODO
	 * @param[in] confidenceThreshold
	 * @param[in] negatives
	 * @param[in] initialNegatives
	 * @param[in] frameOut
	 * @param[in] resultOut
	 */
	void run(shared_ptr<LabeledImageSource> source, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
			float confidenceThreshold, size_t negatives, size_t initialNegatives, ostream& frameOut, ostream& resultOut) const;

	//void runOneClass(shared_ptr<LabeledImageSource> source, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableOneClassSvmClassifier> classifier);

	/**
	 * @return Minimum size (height) of the patches relative to the image height.
	 */
	float getSizeMin() {
		return sizeMin;
	}

	/**
	 * @return Maximum size (height) of the patches relative to the image height.
	 */
	float getSizeMax() {
		return sizeMax;
	}

	/**
	 * @return Incremental scale factor of the size.
	 */
	float getSizeScale() {
		return sizeScale;
	}

	/**
	 * @return Shifting distance relative to the patch size.
	 */
	float getStep() {
		return step;
	}

private:

	/**
	 * Computes the overlap between two rectangular areas (e.g. the ground truth and a patch).
	 *
	 * @param[in] groundTruth TODO
	 * @param[in] patch TODO
	 * @return The overlap between TODO
	 */
	float computeOverlap(Rect_<float> groundTruth, Rect_<float> patch) const;

	float sizeMin;   ///< Minimum size (height) of the patches relative to the image height.
	float sizeMax;   ///< Maximum size (height) of the patches relative to the image height.
	float sizeScale; ///< Incremental scale factor of the size.
	float step;      ///< Shifting distance relative to the patch size.
	float allowedOverlap; ///< Maximum overlap with the ground truth allowed for negative patches.
	string outputDir; ///< Directory of the output files.
	vector<AlgorithmData> algorithmData; ///< Data of the extractor-classifier-combinations.
};

#endif /* BENCHMARK_HPP_ */
