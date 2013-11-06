/*
 * Benchmark.cpp
 *
 *  Created on: 10.09.2013
 *      Author: poschmann
 */

#include "Benchmark.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/Landmark.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "detection/ClassifiedPatch.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using imageio::LandmarkCollection;
using imageio::Landmark;
using imageprocessing::DirectPyramidFeatureExtractor;
using imageprocessing::Patch;
using detection::ClassifiedPatch;
using cv::Mat;
using std::vector;
using std::chrono::system_clock;
using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::minutes;
using std::chrono::duration_cast;
using std::make_shared;
using std::cout;
using std::endl;
using std::runtime_error;

Benchmark::Benchmark(float sizeMin, float sizeMax, float sizeScale, float step, float allowedOverlap, string outputDir) :
		sizeMin(sizeMin), sizeMax(sizeMax), sizeScale(sizeScale), step(step), allowedOverlap(allowedOverlap), outputDir(outputDir), algorithmData() {}

Benchmark::~Benchmark() {}

void Benchmark::add(string name, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
			float confidenceThreshold, size_t negatives, size_t initialNegatives) {
	algorithmData.push_back(AlgorithmData(name, extractor, classifier, confidenceThreshold, negatives, initialNegatives));
}

void Benchmark::run(string name, shared_ptr<LabeledImageSource> source) const {
	cout << "=== " << name << " ===" << endl;
	std::ofstream resultOut(outputDir + '/' + name);
	if (resultOut.fail())
		throw runtime_error("Could not create result file '" + outputDir + '/' + name + '\'');
	resultOut << name << endl;
	for (const AlgorithmData& data : algorithmData) {
		system_clock::time_point now = system_clock::now();
		std::time_t t_now = system_clock::to_time_t(now);
		struct tm* tm_now = std::localtime(&t_now);
		cout << '[' << std::setfill('0') << std::setw(2) << tm_now->tm_hour << ':'
				<< std::setfill('0') << std::setw(2) << tm_now->tm_min << ':'
				<< std::setfill('0') << std::setw(2) << tm_now->tm_sec << "] ";

		cout << data.name << ": ";
		cout.flush();
		resultOut << data.name << ": ";
		source->reset();
		data.classifier->reset();
		std::ofstream frameOut(outputDir + '/' + name + '-' + data.name);
		if (frameOut.fail())
			throw runtime_error("Could not create frame file '" + outputDir + '/' + name + '-' + data.name + '\'');

		steady_clock::time_point start = steady_clock::now();
		try {
			run(source, data.extractor, data.classifier, data.confidenceThreshold, data.negatives, data.initialNegatives, frameOut, resultOut);
		} catch (const std::exception& exc) {
			frameOut << exc.what() << endl;
			resultOut << exc.what() << endl;
			cout << exc.what() << " ";
		}
		steady_clock::time_point end = steady_clock::now();
		minutes duration = duration_cast<minutes>(end - start);
		cout << duration.count() << " min" << endl;
	}
}

void Benchmark::run(
		shared_ptr<LabeledImageSource> source, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableProbabilisticClassifier> classifier,
		float confidenceThreshold, size_t negatives, size_t initialNegatives, ostream& frameOut, ostream& resultOut) const {

	uint32_t frameCount = 0;
	uint64_t extractedPatchCountSum = 0; // total amount of extracted patches
	uint64_t classifiedPatchCountSum = 0; // total amount of classified patches
	uint64_t negativePatchCountSum = 0; // amount of negative patches of all frames except the first ones (where classifier is not usable)
	uint32_t positivePatchCountSum = 0;
	uint64_t updateTimeSum = 0;
	uint64_t extractTimeSum = 0;
	uint64_t falseRejectionsSum = 0;
	uint64_t falseAcceptancesSum = 0;
	uint64_t classifyTimeSum = 0;
	double locationAccuracySum = 0;
	float worstLocationAccuracy = 1;
	uint32_t positiveTrainingCountSum = 0;
	uint32_t negativeTrainingCountSum = 0;
	uint64_t trainingTimeSum = 0;
	cv::Size patchSize;

	bool initialized = false;
	float aspectRatio = 1;
	while (source->next()) {
		Mat image = source->getImage();
		const LandmarkCollection landmarks = source->getLandmarks();
		if (landmarks.isEmpty())
			continue;
		const shared_ptr<Landmark> landmark = landmarks.getLandmark();
		bool hasGroundTruth = landmark->isVisible();
		if (!hasGroundTruth && !initialized)
			continue;

		Rect_<float> groundTruth;
		if (hasGroundTruth) {
			groundTruth = landmark->getRect();
			aspectRatio = groundTruth.width / groundTruth.height;
			if (!initialized) {
				DirectPyramidFeatureExtractor* pyramidExtractor = dynamic_cast<DirectPyramidFeatureExtractor*>(extractor.get());
				if (pyramidExtractor != nullptr) {
					patchSize = pyramidExtractor->getPatchSize();
					double dimension = patchSize.width * patchSize.height;
					double patchHeight = sqrt(dimension / aspectRatio);
					double patchWidth = aspectRatio * patchHeight;
					pyramidExtractor->setPatchSize(cvRound(patchWidth), cvRound(patchHeight));
				}
				initialized = true;
			}
		}

		steady_clock::time_point extractionStart = steady_clock::now();
		extractor->update(image);
		steady_clock::time_point extractionBetween = steady_clock::now();
		shared_ptr<Patch> positivePatch;
		if (hasGroundTruth) {
			positivePatch = extractor->extract(
					cvRound(groundTruth.x + 0.5f * groundTruth.width),
					cvRound(groundTruth.y + 0.5f * groundTruth.height),
					cvRound(groundTruth.width),
					cvRound(groundTruth.height));
			if (!positivePatch)
				continue;
		}
		vector<shared_ptr<Patch>> negativePatches;
		vector<shared_ptr<Patch>> neutralPatches;
		int patchCount = hasGroundTruth ? 1 : 0;
		for (float size = sizeMin; size <= sizeMax; size *= sizeScale) {
			float realHeight = size * image.rows;
			float realWidth = aspectRatio * realHeight;
			int width = cvRound(realWidth);
			int height = cvRound(realHeight);
			int minX = width / 2;
			int minY = height / 2;
			int maxX = image.cols - width + width / 2;
			int maxY = image.rows - height + height / 2;
			float stepX = std::max(1.f, step * realWidth);
			float stepY = std::max(1.f, step * realHeight);
			for (float x = minX; cvRound(x) < maxX; x += stepX) {
				for (float y = minY; cvRound(y) < maxY; y += stepY) {
					shared_ptr<Patch> patch = extractor->extract(cvRound(x), cvRound(y), width, height);
					if (patch) {
						patchCount++;
						if (!hasGroundTruth || computeOverlap(groundTruth, patch->getBounds()) <= allowedOverlap)
							negativePatches.push_back(patch);
						else
							neutralPatches.push_back(patch);
					}
				}
			}
		}
		steady_clock::time_point extractionEnd = steady_clock::now();
		milliseconds updateDuration = duration_cast<milliseconds>(extractionBetween - extractionStart);
		milliseconds extractDuration = duration_cast<milliseconds>(extractionEnd - extractionBetween);

		frameOut << source->getName().filename().generic_string() << ": " << updateDuration.count() << "ms " << patchCount << " " << extractDuration.count() << "ms";
		frameCount++;
		extractedPatchCountSum += patchCount;
		updateTimeSum += updateDuration.count();
		extractTimeSum += extractDuration.count();

		if (classifier->isUsable()) {
			classifiedPatchCountSum += patchCount;
			if (hasGroundTruth)
				positivePatchCountSum++; // amount needed for false acceptance rate, therefore only positive patches that are tested
			negativePatchCountSum += negativePatches.size(); // amount needed for false acceptance rate, therefore only negative patches that are tested

			steady_clock::time_point classificationStart = steady_clock::now();
			// positive patch
			shared_ptr<ClassifiedPatch> classifiedPositivePatch;
			if (hasGroundTruth)
				classifiedPositivePatch = make_shared<ClassifiedPatch>(positivePatch, classifier->classify(positivePatch->getData()));
			// negative patches
			vector<shared_ptr<ClassifiedPatch>> classifiedNegativePatches;
			classifiedNegativePatches.reserve(negativePatches.size());
			for (const shared_ptr<Patch>& patch : negativePatches)
				classifiedNegativePatches.push_back(make_shared<ClassifiedPatch>(patch, classifier->classify(patch->getData())));
			// neutral patches
			vector<shared_ptr<ClassifiedPatch>> classifiedNeutralPatches;
			classifiedNeutralPatches.reserve(neutralPatches.size());
			for (const shared_ptr<Patch>& patch : neutralPatches)
				classifiedNeutralPatches.push_back(make_shared<ClassifiedPatch>(patch, classifier->classify(patch->getData())));
			steady_clock::time_point classificationEnd = steady_clock::now();
			milliseconds classificationDuration = duration_cast<milliseconds>(classificationEnd - classificationStart);

			int falseRejections = !hasGroundTruth || classifiedPositivePatch->isPositive() ? 0 : 1;
			int falseAcceptances = std::count_if(classifiedNegativePatches.begin(), classifiedNegativePatches.end(), [](shared_ptr<ClassifiedPatch>& patch) {
				return patch->isPositive();
			});

			frameOut << " " << falseRejections << "/" << (hasGroundTruth ? "1 " : "0 ") << falseAcceptances << "/" << negativePatches.size() << " " << classificationDuration.count() << "ms";
			falseRejectionsSum += falseRejections;
			falseAcceptancesSum += falseAcceptances;
			classifyTimeSum += classificationDuration.count();

			if (hasGroundTruth) {
				shared_ptr<ClassifiedPatch> bestNegativePatch = *std::max_element(classifiedNegativePatches.begin(), classifiedNegativePatches.end(), [](shared_ptr<ClassifiedPatch>& a, shared_ptr<ClassifiedPatch>& b) {
					return a->getProbability() > b->getProbability();
				});
				shared_ptr<ClassifiedPatch> bestNeutralPatch = *std::max_element(classifiedNeutralPatches.begin(), classifiedNeutralPatches.end(), [](shared_ptr<ClassifiedPatch>& a, shared_ptr<ClassifiedPatch>& b) {
					return a->getProbability() > b->getProbability();
				});
				float overlap = 0;
				if (classifiedPositivePatch->getProbability() >= bestNeutralPatch->getProbability() && classifiedPositivePatch->getProbability() >= bestNegativePatch->getProbability())
					overlap = computeOverlap(groundTruth, classifiedPositivePatch->getPatch()->getBounds());
				else if (bestNeutralPatch->getProbability() >= bestNegativePatch->getProbability())
					overlap = computeOverlap(groundTruth, bestNeutralPatch->getPatch()->getBounds());
				else
					overlap = computeOverlap(groundTruth, bestNegativePatch->getPatch()->getBounds());

				frameOut << " " << cvRound(100 * overlap) << "%";
				locationAccuracySum += overlap;
				worstLocationAccuracy = std::min(worstLocationAccuracy, overlap);
			}

			vector<shared_ptr<ClassifiedPatch>> negativeTrainingCandidates;
			for (shared_ptr<ClassifiedPatch>& patch : classifiedNegativePatches) {
				if (patch->isPositive() || patch->getProbability() > 1 - confidenceThreshold)
					negativeTrainingCandidates.push_back(patch);
			}
			if (negativeTrainingCandidates.size() > negatives) {
				std::partial_sort(negativeTrainingCandidates.begin(), negativeTrainingCandidates.begin() + negatives, negativeTrainingCandidates.end(), [](shared_ptr<ClassifiedPatch>& a, shared_ptr<ClassifiedPatch>& b) {
					return a->getProbability() > b->getProbability();
				});
				negativeTrainingCandidates.resize(negatives);
			}
			vector<Mat> negativeTrainingExamples;
			negativeTrainingExamples.reserve(negativeTrainingCandidates.size());
			for (const shared_ptr<ClassifiedPatch>& patch : negativeTrainingCandidates)
				negativeTrainingExamples.push_back(patch->getPatch()->getData());
			vector<Mat> positiveTrainingExamples;
			if (hasGroundTruth && (!classifiedPositivePatch->isPositive() || classifiedPositivePatch->getProbability() < confidenceThreshold))
				positiveTrainingExamples.push_back(classifiedPositivePatch->getPatch()->getData());

			steady_clock::time_point trainingStart = steady_clock::now();
			classifier->retrain(positiveTrainingExamples, negativeTrainingExamples);
			steady_clock::time_point trainingEnd = steady_clock::now();
			milliseconds trainingDuration = duration_cast<milliseconds>(trainingEnd - trainingStart);

			frameOut << " " << positiveTrainingExamples.size() << "+" << negativeTrainingExamples.size() << " " << trainingDuration.count() << "ms";
			positiveTrainingCountSum += positiveTrainingExamples.size();
			negativeTrainingCountSum += negativeTrainingExamples.size();
			trainingTimeSum += trainingDuration.count();
		} else {
			int step = negativePatches.size() / initialNegatives;
			int first = step / 2;
			vector<Mat> negativeTrainingExamples;
			negativeTrainingExamples.reserve(initialNegatives);
			for (size_t i = first; i < negativePatches.size(); i += step)
				negativeTrainingExamples.push_back(negativePatches[i]->getData());
			vector<Mat> positiveTrainingExamples;
			if (hasGroundTruth)
				positiveTrainingExamples.push_back(positivePatch->getData());

			steady_clock::time_point trainingStart = steady_clock::now();
			classifier->retrain(positiveTrainingExamples, negativeTrainingExamples);
			steady_clock::time_point trainingEnd = steady_clock::now();
			milliseconds trainingDuration = duration_cast<milliseconds>(trainingEnd - trainingStart);

			frameOut << " " << positiveTrainingExamples.size() << "+" << negativeTrainingExamples.size() << " " << trainingDuration.count() << "ms";
			positiveTrainingCountSum += positiveTrainingExamples.size();
			negativeTrainingCountSum += negativeTrainingExamples.size();
			trainingTimeSum += trainingDuration.count();
		}
		frameOut << endl;
	}

	DirectPyramidFeatureExtractor* pyramidExtractor = dynamic_cast<DirectPyramidFeatureExtractor*>(extractor.get());
	if (pyramidExtractor != nullptr) {
		pyramidExtractor->setPatchSize(patchSize.width, patchSize.height);
	}

	if (frameCount == 0) {
		frameOut << "no valid frames" << endl;
		resultOut << "no valid frames" << endl;
	} else if (frameCount == 1 || positivePatchCountSum < 1) {
		frameOut << "too few valid frames" << endl;
		resultOut << "too few valid frames" << endl;
	} else if (extractedPatchCountSum == 0) {
		frameOut << "no valid patches" << endl;
		resultOut << "no valid patches" << endl;
	} else if (negativePatchCountSum == 0) {
		frameOut << "no valid negative patches" << endl;
		resultOut << "no valid negative patches" << endl;
	} else {
		frameOut << frameCount << " " << extractedPatchCountSum << " " << classifiedPatchCountSum << " " << negativePatchCountSum << " " << positivePatchCountSum << " "
				<< updateTimeSum << "ms " << extractTimeSum << "ms " << falseRejectionsSum << " " << falseAcceptancesSum << " "
				<< classifyTimeSum << "ms " << cvRound(100 * locationAccuracySum / positivePatchCountSum) << "% " << cvRound(100 * worstLocationAccuracy) << "% "
				<< positiveTrainingCountSum << " " << negativeTrainingCountSum << " " << trainingTimeSum << "ms" << endl;

		float falseRejectionRate = static_cast<float>(falseRejectionsSum) / static_cast<float>(frameCount - 1);
		float falseAcceptanceRate = static_cast<float>(falseRejectionsSum) / static_cast<float>(negativePatchCountSum);
		float avgPositiveTrainingCount = static_cast<float>(positiveTrainingCountSum) / static_cast<float>(frameCount);
		float avgNegativeTrainingCount = static_cast<float>(negativeTrainingCountSum) / static_cast<float>(frameCount);
		uint64_t normalizedExtractionTime = updateTimeSum / frameCount + (1000 * extractTimeSum) / extractedPatchCountSum;
		uint64_t normalizedClassificationTime = (1000 * classifyTimeSum) / classifiedPatchCountSum;
		uint64_t normalizedTrainingTime = trainingTimeSum / frameCount;
		uint64_t normalizedFrameTime = normalizedExtractionTime + normalizedClassificationTime + normalizedTrainingTime;
		resultOut.setf(std::ios::fixed);
		resultOut.precision(2);
		resultOut << falseRejectionRate << " " << falseAcceptanceRate << " "
				<< cvRound(100 * locationAccuracySum / positivePatchCountSum) << "% " << cvRound(100 * worstLocationAccuracy) << "% "
				<< avgPositiveTrainingCount << " " << avgNegativeTrainingCount << " "
				<< normalizedFrameTime << "ms (" << normalizedExtractionTime << "ms " << normalizedClassificationTime << "ms " << normalizedTrainingTime << "ms)" << endl;
	}
}

float Benchmark::computeOverlap(Rect_<float> groundTruth, Rect_<float> patch) const {
	Rect_<float> intersection = groundTruth & patch;
	return intersection.area() / (groundTruth.area() + patch.area() - intersection.area());
}

/*void Benchmark::runOneClass(shared_ptr<LabeledImageSource> source, shared_ptr<FeatureExtractor> extractor, shared_ptr<TrainableOneClassSvmClassifier> classifier) {
	while (source->next()) {
		Mat image = source->getImage();
		const LandmarkCollection landmarks = source->getLandmarks();
		if (!landmarks.isEmpty()) {
			const Landmark& landmark = landmarks.getLandmark();
			if (landmark.isVisible()) {
				Rect_<float> groundTruth = landmark.getRect();
				float aspectRatio = groundTruth.width / groundTruth.height;
				// TODO im falle von pyramid extractor: patchsize setzen

				steady_clock::time_point extractionStart = steady_clock::now();
				extractor->update(image);
				steady_clock::time_point extractionBetween = steady_clock::now();
				shared_ptr<Patch> positivePatch = extractor->extract(
						cvRound(groundTruth.x + 0.5f * groundTruth.width),
						cvRound(groundTruth.y + 0.5f * groundTruth.height),
						cvRound(groundTruth.width),
						cvRound(groundTruth.height));
				vector<shared_ptr<Patch>> negativePatches;
				int patchCount = 1;
				for (float size = sizeMin; size <= sizeMax; size *= sizeScale) {
					float realHeight = size * image.rows;
					float realWidth = aspectRatio * realHeight;
					int width = cvRound(realWidth);
					int height = cvRound(realHeight);
					int minX = width / 2;
					int minY = height / 2;
					int maxX = image.cols - width + width / 2;
					int maxY = image.rows - height + height / 2;
					float stepX = std::max(1.f, step * realWidth);
					float stepY = std::max(1.f, step * realHeight);
					for (float x = minX; cvRound(x) < maxX; x += stepX) {
						for (float y = minY; cvRound(y) < maxY; y += stepY) {
							shared_ptr<Patch> patch = extractor->extract(cvRound(x), cvRound(y), width, height);
							if (patch) {
								patchCount++;
								Rect_<float> bounds = patch->getBounds();
								Rect_<float> intersection = groundTruth & bounds;
								float overlap = intersection.area() / (groundTruth.area() + bounds.area() - intersection.area());
								if (overlap <= 0.3f) // TODO parameter
									negativePatches.push_back(patch);
							}
						}
					}
				}
				steady_clock::time_point extractionEnd = steady_clock::now();
				milliseconds updateDuration = duration_cast<milliseconds>(extractionBetween - extractionStart);
				milliseconds extractDuration = duration_cast<milliseconds>(extractionEnd - extractionBetween);
//				milliseconds extractionDuration = duration_cast<milliseconds>(extractionEnd - extractionStart);
				int64_t normalizedExtractionTime = updateDuration.count() + 1000 * extractDuration.count() / patchCount;

				cout << source->getName().filename().generic_string() << ": " << patchCount << " " << normalizedExtractionTime << "ms";

				if (classifier->isUsable()) {
					steady_clock::time_point classificationStart = steady_clock::now();
					double positiveDistance = classifier->getSvm()->computeHyperplaneDistance(positivePatch->getData());
					shared_ptr<ClassifiedPatch> classifiedPositivePatch = make_shared<ClassifiedPatch>(positivePatch, classifier->getSvm()->classify(positiveDistance));
					vector<shared_ptr<ClassifiedPatch>> classifiedNegativePatches;
					classifiedNegativePatches.reserve(negativePatches.size());
					double negativeDistanceSum = 0;
					double minNegativeDistance = -1000;
					double maxNegativeDistance = 1000;
					for (const shared_ptr<Patch>& patch : negativePatches) {
						double negativeDistance = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
						classifiedNegativePatches.push_back(make_shared<ClassifiedPatch>(patch, classifier->getSvm()->classify(negativeDistance)));
						negativeDistanceSum += negativeDistance;
						if (negativeDistance > minNegativeDistance)
							minNegativeDistance = negativeDistance;
						if (negativeDistance < maxNegativeDistance)
							maxNegativeDistance = negativeDistance;
					}
					double avgNegativeDistance = negativeDistanceSum / negativePatches.size();
					steady_clock::time_point classificationEnd = steady_clock::now();
					milliseconds classificationDuration = duration_cast<milliseconds>(classificationEnd - classificationStart);
					int64_t normalizedClassificationTime = 1000 * classificationDuration.count() / (negativePatches.size() + 1);

					int falseRejections = classifiedPositivePatch->isPositive() ? 0 : 1;
					int falseAcceptances = std::count_if(classifiedNegativePatches.begin(), classifiedNegativePatches.end(), [](shared_ptr<ClassifiedPatch>& patch) {
						return patch->isPositive();
					});

					cout << " " << falseRejections << "/1 " << falseAcceptances << "/" << negativePatches.size() << " " << normalizedClassificationTime << "ms";

					vector<Mat> negativeTrainingExamples;
					vector<Mat> positiveTrainingExamples;
//					if (!classifiedPositivePatch->isPositive() || classifiedPositivePatch->getProbability() < 0.99) // TODO parameter
						positiveTrainingExamples.push_back(classifiedPositivePatch->getPatch()->getData());

					steady_clock::time_point trainingStart = steady_clock::now();
					classifier->retrain(positiveTrainingExamples, negativeTrainingExamples);
					steady_clock::time_point trainingEnd = steady_clock::now();
					milliseconds trainingDuration = duration_cast<milliseconds>(trainingEnd - trainingStart);

					cout << " " << positiveTrainingExamples.size() << " " << trainingDuration.count() << "ms" << endl;
					cout << positiveDistance << " " << minNegativeDistance << " " << avgNegativeDistance << " " << maxNegativeDistance;
				} else {
					vector<Mat> negativeTrainingExamples;
					vector<Mat> positiveTrainingExamples;
					positiveTrainingExamples.push_back(positivePatch->getData());

					steady_clock::time_point trainingStart = steady_clock::now();
					classifier->retrain(positiveTrainingExamples, negativeTrainingExamples);
					steady_clock::time_point trainingEnd = steady_clock::now();
					milliseconds trainingDuration = duration_cast<milliseconds>(trainingEnd - trainingStart);

					cout << " " << positiveTrainingExamples.size() << " " << trainingDuration.count() << "ms";
				}
				cout << endl;
			}
		}
	}
}*/
