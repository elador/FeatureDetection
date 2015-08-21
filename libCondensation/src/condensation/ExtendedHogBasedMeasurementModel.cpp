/*
 * ExtendedHogBasedMeasurementModel.cpp
 *
 *  Created on: 13.01.2014
 *      Author: poschmann
 */

#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
#include "condensation/Sample.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include "imageprocessing/GradientBinningFilter.hpp"
#include "imageprocessing/ExtendedHogFilter.hpp"
#include "imageprocessing/CompleteExtendedHogFilter.hpp"
#include "imageprocessing/ConvolutionFilter.hpp"
#include "imageprocessing/CellBasedPyramidFeatureExtractor.hpp"
#include "imageprocessing/ExtendedHogFeatureExtractor.hpp"
#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/BinaryClassifier.hpp"
#include "classification/ProbabilisticClassifier.hpp"
#include <stdexcept>

using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::ImagePyramid;
using imageprocessing::ImagePyramidLayer;
using imageprocessing::GrayscaleFilter;
using imageprocessing::GradientFilter;
using imageprocessing::GradientBinningFilter;
using imageprocessing::ExtendedHogFilter;
using imageprocessing::CompleteExtendedHogFilter;
using imageprocessing::ConvolutionFilter;
using imageprocessing::FeatureExtractor;
using imageprocessing::CellBasedPyramidFeatureExtractor;
using imageprocessing::ExtendedHogFeatureExtractor;
using classification::LinearKernel;
using classification::ProbabilisticSvmClassifier;
using classification::TrainableProbabilisticSvmClassifier;
using cv::Mat;
using cv::Rect;
using std::pair;
using std::vector;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;
using std::runtime_error;
using std::invalid_argument;
using std::unordered_map;
using std::round;

namespace condensation {

ExtendedHogBasedMeasurementModel::ExtendedHogBasedMeasurementModel(shared_ptr<TrainableProbabilisticSvmClassifier> classifier) :
		cellSize(5), cellCount(35), signedAndUnsigned(false), interpolateBins(false), interpolateCells(true), octaveLayerCount(5),
		rejectionThreshold(-1.5), useSlidingWindow(true), conservativeReInit(false),
		negativeExampleCount(10), initialNegativeExampleCount(50), randomExampleCount(50), negativeScoreThreshold(-1.0f),
		positiveOverlapThreshold(0.5), negativeOverlapThreshold(0.5),
		adaptation(Adaptation::POSITION), adaptationThreshold(0.75), exclusionThreshold(0.0),
		convolutionFilter(make_shared<ConvolutionFilter>(CV_32F)), basePyramid(), heatPyramid(),
		featureExtractor(), positiveFeatureExtractor(), heatExtractor(),
		classifier(classifier->getProbabilisticSvm()), trainable(classifier),
		cellRowCount(), cellColumnCount(), minWidth(), maxWidth(),
		initialized(false), usable(false), targetLost(false),
		generator(), uniformIntDistribution(), normalDistribution(),
		initialFeatures(), trajectoryFeatures(), pastFeatureExtractors(),
		trajectoryToLearn(), frameIndex(0), learned() {
	if (!dynamic_cast<LinearKernel*>(this->classifier->getSvm()->getKernel().get()))
		throw invalid_argument("ExtendedHogBasedMeasurementKernel: the SVM must use a LinearKernel");
}

ExtendedHogBasedMeasurementModel::ExtendedHogBasedMeasurementModel(
		shared_ptr<TrainableProbabilisticSvmClassifier> classifier, shared_ptr<ImagePyramid> basePyramid) :
				cellSize(5), cellCount(35), signedAndUnsigned(false), interpolateBins(false), interpolateCells(true), octaveLayerCount(5),
				rejectionThreshold(-1.5), useSlidingWindow(true), conservativeReInit(false),
				negativeExampleCount(10), initialNegativeExampleCount(50), randomExampleCount(50), negativeScoreThreshold(-1.0f),
				positiveOverlapThreshold(0.5), negativeOverlapThreshold(0.5),
				adaptation(Adaptation::POSITION), adaptationThreshold(0.75), exclusionThreshold(0.0),
				convolutionFilter(make_shared<ConvolutionFilter>(CV_32F)), basePyramid(basePyramid), heatPyramid(),
				featureExtractor(), positiveFeatureExtractor(), heatExtractor(),
				classifier(classifier->getProbabilisticSvm()), trainable(classifier),
				cellRowCount(), cellColumnCount(), minWidth(), maxWidth(),
				initialized(false), usable(false), targetLost(false),
				generator(), uniformIntDistribution(), normalDistribution(),
				initialFeatures(), trajectoryFeatures(), pastFeatureExtractors(),
				trajectoryToLearn(), frameIndex(0), learned() {
	if (!dynamic_cast<LinearKernel*>(this->classifier->getSvm()->getKernel().get()))
		throw invalid_argument("ExtendedHogBasedMeasurementKernel: the SVM must use a LinearKernel");
}

void ExtendedHogBasedMeasurementModel::update(shared_ptr<VersionedImage> image) {
	if (useSlidingWindow) {
		positiveFeatureExtractor->update(image);
		featureExtractor->update(image);
		heatExtractor->update(image);
	} else {
		featureExtractor->update(image);
	}
}

void ExtendedHogBasedMeasurementModel::evaluate(shared_ptr<VersionedImage> image, vector<shared_ptr<Sample>>& samples) {
	update(image);
	if (!useSlidingWindow) {
		for (shared_ptr<Sample> sample : samples)
			evaluate(*sample);
	} else { // use sliding window
		pair<double, Rect> peak = getHeatPeak();
		if (targetLost) {
			double peakScore = peak.first;
			if (classifier->getSvm()->classify(peakScore) && (!conservativeReInit || peakScore > adaptationThreshold)) {
				// re-initialize tracker at location of score peak
				int clusterId = Sample::getNextClusterId();
				for (shared_ptr<Sample>& sample : samples) {
					sample->setX(peak.second.x + peak.second.width / 2 + 0.2 * peak.second.width * normalDistribution(generator));
					sample->setY(peak.second.y + peak.second.height / 2 + 0.2 * peak.second.width * normalDistribution(generator));
					sample->setSize(peak.second.width * (1 + 0.2 * normalDistribution(generator)));
					sample->setVx(0.1 * peak.second.width * normalDistribution(generator));
					sample->setVy(0.1 * peak.second.width * normalDistribution(generator));
					sample->setVSize(1 + 0.1 * normalDistribution(generator));
					sample->setClusterId(clusterId);
					sample->resetAncestor();
					evaluate(*sample);
				}
			} else { // target was lost and could not be re-initialized
				for (shared_ptr<Sample>& sample : samples) {
					sample->setWeight(0);
					sample->setScore(0);
					sample->setTarget(false);
				}
			}
		} else { // target was not lost
			double bestScore = std::numeric_limits<double>::lowest();
			for (shared_ptr<Sample> sample : samples) {
				evaluate(*sample);
				bestScore = std::max(bestScore, sample->getScore());
			}
			double peakScore = peak.first;
			double initialFeaturesScore = classifier->getSvm()->computeHyperplaneDistance(initialFeatures);
			double scoreThreshold = 0.5 * (bestScore + initialFeaturesScore);
			if (conservativeReInit)
				scoreThreshold = std::max(scoreThreshold, adaptationThreshold);
			if (bestScore < initialFeaturesScore && classifier->getSvm()->classify(peakScore) && peakScore > scoreThreshold) {
				// re-initialize tracker at location of score peak
				trajectoryFeatures.clear();
				trajectoryToLearn.clear();
				pastFeatureExtractors.clear();
				int clusterId = Sample::getNextClusterId();
				for (shared_ptr<Sample>& sample : samples) {
					sample->setX(peak.second.x + peak.second.width / 2 + 0.2 * peak.second.width * normalDistribution(generator));
					sample->setY(peak.second.y + peak.second.height / 2 + 0.2 * peak.second.width * normalDistribution(generator));
					sample->setSize(peak.second.width * (1 + 0.2 * normalDistribution(generator)));
					sample->setVx(0.1 * peak.second.width * normalDistribution(generator));
					sample->setVy(0.1 * peak.second.width * normalDistribution(generator));
					sample->setVSize(1 + 0.1 * normalDistribution(generator));
					sample->setClusterId(clusterId);
					sample->resetAncestor();
					evaluate(*sample);
				}
			}
		}
	}
}

void ExtendedHogBasedMeasurementModel::evaluate(Sample& sample) const {
	if (!useSlidingWindow) {
		shared_ptr<Patch> patch = featureExtractor->extract(sample.getX(), sample.getY(), sample.getWidth(), sample.getHeight());
		if (!patch) {
			sample.setTarget(false);
			sample.setWeight(0);
			sample.setScore(0);
		} else {
			double score = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
			pair<bool, double> result = classifier->getProbability(score);
			double globalLikelihood = result.second;
			sample.setWeight(sample.getWeight() * globalLikelihood);
			sample.setScore(score);
			if (targetLost)
				sample.setTarget(result.first);
			else
				sample.setTarget(true);
		}
	} else {
		shared_ptr<Patch> patch = featureExtractor->extract(sample.getX(), sample.getY(), sample.getWidth(), sample.getHeight());
		if (!patch) {
			sample.setTarget(false);
			sample.setWeight(0);
			sample.setScore(0);
		} else {
			shared_ptr<Patch> heatPatch = heatExtractor->extract(sample.getX(), sample.getY(), sample.getWidth(), sample.getHeight());
			double score = heatPatch->getData().at<float>(cellRowCount / 2, cellColumnCount / 2);
			pair<bool, double> result = classifier->getProbability(score);
			double globalLikelihood = result.second;
			sample.setWeight(sample.getWeight() * globalLikelihood);
			sample.setScore(score);
			if (targetLost)
				sample.setTarget(result.first);
			else
				sample.setTarget(score > rejectionThreshold);
		}
	}
}

bool ExtendedHogBasedMeasurementModel::isValid(const Sample& target,
		const vector<shared_ptr<Sample>>& samples, shared_ptr<VersionedImage> image) {
	shared_ptr<Patch> patch = positiveFeatureExtractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
	return patch && classifier->getSvm()->computeHyperplaneDistance(patch->getData()) > rejectionThreshold;
}

bool ExtendedHogBasedMeasurementModel::isUsable() const {
	return usable;
}

bool ExtendedHogBasedMeasurementModel::initialize(shared_ptr<VersionedImage> image, Sample& target) {
	if (!initialized) {
		double aspectRatio = static_cast<double>(target.getHeight()) / static_cast<double>(target.getWidth());
		if (aspectRatio < 1) { // height is less than width, so determine height first
			cellRowCount = cvRound(sqrt(aspectRatio * cellCount));
			cellColumnCount = cvRound(cellRowCount / aspectRatio);
		} else { // width is less than or equal to height, so determine width first
			cellColumnCount = cvRound(sqrt(cellCount / aspectRatio));
			cellRowCount = cvRound(aspectRatio * cellColumnCount);
		}
		double newAspectRatio = static_cast<double>(cellRowCount) / static_cast<double>(cellColumnCount);
		if (newAspectRatio < aspectRatio) // less height at same width - so size of target has to be increased
			target.setSize(cvRound(aspectRatio * target.getSize() / newAspectRatio));
		Sample::setAspectRatio(cellColumnCount, cellRowCount);

		double imageAspectRatio = static_cast<double>(image->getData().rows) / static_cast<double>(image->getData().cols);
		minWidth = cellSize * cellColumnCount;
		if (aspectRatio > imageAspectRatio) { // height is the limiting factor when rescaling
			size_t maxHeight = image->getData().rows;
			maxWidth = static_cast<size_t>(maxHeight / aspectRatio);
		} else { // width is the limiting factor when rescaling
			maxWidth = image->getData().cols;
		}

		shared_ptr<CompleteExtendedHogFilter> hogFilter;
		if (signedAndUnsigned)
			hogFilter = make_shared<CompleteExtendedHogFilter>(cellSize, 18, true, true, interpolateBins, interpolateCells, 0.2);
		else
			hogFilter = make_shared<CompleteExtendedHogFilter>(cellSize, 9, false, true, interpolateBins, interpolateCells, 0.48);

		if (basePyramid) {
			positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(basePyramid, hogFilter, cellColumnCount, cellRowCount);
			int patchWidth = positiveFeatureExtractor->getPatchWidth();
			minWidth = std::max(minWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMaxScaleFactor())));
			maxWidth = std::min(maxWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMinScaleFactor())));
		} else { // no base pyramid given
			positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(
					hogFilter, cellColumnCount, cellRowCount, minWidth, maxWidth, octaveLayerCount);
		}

		if (useSlidingWindow) {
			shared_ptr<ImagePyramid> featurePyramid = make_shared<ImagePyramid>(positiveFeatureExtractor->getPyramid());
			featurePyramid->addLayerFilter(hogFilter);
			heatPyramid = make_shared<ImagePyramid>(featurePyramid);
			heatPyramid->addLayerFilter(convolutionFilter);
			featureExtractor = make_shared<CellBasedPyramidFeatureExtractor>(featurePyramid, cellSize, cellColumnCount, cellRowCount);
			heatExtractor = make_shared<CellBasedPyramidFeatureExtractor>(heatPyramid, cellSize, cellColumnCount, cellRowCount);
		} else {
			featureExtractor = positiveFeatureExtractor;
		}

		// TODO alternative with other hog filter for non-sliding-window - why worse (especially on ball video)?
//		if (useSlidingWindow) {
//			shared_ptr<CompleteExtendedHogFilter> hogFilter;
//			if (signedAndUnsigned)
//				hogFilter = make_shared<CompleteExtendedHogFilter>(cellSize, 18, true, true, interpolateBins, interpolateCells, 0.2);
//			else
//				hogFilter = make_shared<CompleteExtendedHogFilter>(cellSize, 9, false, true, interpolateBins, interpolateCells, 0.48);
//			if (basePyramid) {
//				positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(basePyramid, hogFilter, cellColumnCount, cellRowCount);
//				int patchWidth = positiveFeatureExtractor->getPatchWidth();
//				minWidth = std::max(minWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMaxScaleFactor())));
//				maxWidth = std::min(maxWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMinScaleFactor())));
//			} else { // no base pyramid given
//				positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(
//						hogFilter, cellColumnCount, cellRowCount, minWidth, maxWidth, octaveLayerCount);
//			}
//			shared_ptr<ImagePyramid> featurePyramid = make_shared<ImagePyramid>(positiveFeatureExtractor->getPyramid());
//			featurePyramid->addLayerFilter(hogFilter);
//			heatPyramid = make_shared<ImagePyramid>(featurePyramid);
//			heatPyramid->addLayerFilter(convolutionFilter);
//			featureExtractor = make_shared<CellBasedPyramidFeatureExtractor>(featurePyramid, cellSize, cellColumnCount, cellRowCount);
//			heatExtractor = make_shared<CellBasedPyramidFeatureExtractor>(heatPyramid, cellSize, cellColumnCount, cellRowCount);
//		} else {
//			shared_ptr<GradientFilter> gradientFilter = make_shared<GradientFilter>(1, 0);
//			shared_ptr<GradientBinningFilter> binningFilter;
//			shared_ptr<ExtendedHogFilter> hogFilter;
//			if (signedAndUnsigned) {
//				binningFilter = make_shared<GradientBinningFilter>(18, true, interpolateBins);
//				hogFilter = make_shared<ExtendedHogFilter>(18, cellSize, interpolateCells, true, 0.2);
//			} else {
//				binningFilter = make_shared<GradientBinningFilter>(9, false, interpolateBins);
//				hogFilter = make_shared<ExtendedHogFilter>(9, cellSize, interpolateCells, false, 0.48);
//			}
//			if (basePyramid) {
//				shared_ptr<ImagePyramid> featurePyramid = make_shared<ImagePyramid>(basePyramid);
//				featurePyramid->addLayerFilter(gradientFilter);
//				featurePyramid->addLayerFilter(binningFilter);
//				positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(featurePyramid, hogFilter, cellColumnCount, cellRowCount);
//				int patchWidth = positiveFeatureExtractor->getPatchWidth();
//				minWidth = std::max(minWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMaxScaleFactor())));
//				maxWidth = std::min(maxWidth, static_cast<size_t>(round(patchWidth / basePyramid->getMinScaleFactor())));
//			} else {
//				positiveFeatureExtractor = make_shared<ExtendedHogFeatureExtractor>(
//						gradientFilter, binningFilter, hogFilter, cellColumnCount, cellRowCount, minWidth, maxWidth, octaveLayerCount);
//			}
//			featureExtractor = positiveFeatureExtractor;
//		}

		initialized = true;
	}

	Rect targetBounds = target.getBounds();
	positiveFeatureExtractor->update(image);
	if (useSlidingWindow)
		featureExtractor->update(image);
	shared_ptr<Patch> patch = positiveFeatureExtractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
	if (!patch) {
		reset();
		return false;
	}
	initialFeatures = patch->getData();
	vector<Mat> positiveExamples;
	positiveExamples.push_back(initialFeatures);
	vector<Mat> negativeTrainingExamples = createRandomNegativeExamples(initialNegativeExampleCount, image->getData(), targetBounds);
	usable = trainable->retrain(positiveExamples, negativeTrainingExamples);
	if (usable) {
		if (classifier->getSvm()->getSupportVectors().size() != 1)
			throw runtime_error("ExtendedHogBasedMeasurementModel: the amount of support vectors has to be one (w)");
		if (useSlidingWindow) {
			convolutionFilter->setKernel(classifier->getSvm()->getSupportVectors()[0]);
			convolutionFilter->setDelta(-classifier->getSvm()->getBias());
			heatExtractor->update(image);
			vector<Mat> newPositiveTrainingExamples;
			vector<Mat> newNegativeTrainingExamples = createGoodNegativeExamples(targetBounds);
			if (newNegativeTrainingExamples.size() > negativeExampleCount)
				newNegativeTrainingExamples.resize(negativeExampleCount);
			usable = trainable->retrain(newPositiveTrainingExamples, newNegativeTrainingExamples);
			if (usable) {
				if (classifier->getSvm()->getSupportVectors().size() != 1)
					throw runtime_error("ExtendedHogBasedMeasurementModel: the amount of support vectors has to be one (w)");
				convolutionFilter->setKernel(classifier->getSvm()->getSupportVectors()[0]);
				convolutionFilter->setDelta(-classifier->getSvm()->getBias());
			}
		} else {
			vector<Mat> candidates = createRandomNegativeExamples(randomExampleCount, image->getData(), targetBounds);
			vector<pair<float, Mat>> classifiedCandidates(candidates.size());
			std::transform(candidates.begin(), candidates.end(), classifiedCandidates.begin(), [this](const Mat& example) {
				return std::make_pair(classifier->getSvm()->computeHyperplaneDistance(example), example);
			});
			std::partial_sort(classifiedCandidates.begin(), classifiedCandidates.begin() + negativeExampleCount, classifiedCandidates.end(), [](const pair<float, Mat>& a, const pair<float, Mat>& b) {
				return a.first > b.first;
			});
			size_t count = negativeExampleCount;
			while (count > 0 && classifiedCandidates[count - 1].first <= negativeScoreThreshold)
				count--;
			if (count > 0) {
				classifiedCandidates.resize(count);
				vector<Mat> newNegativeTrainingExamples;
				newNegativeTrainingExamples.reserve(classifiedCandidates.size());
				for (const pair<float, Mat>& pair : classifiedCandidates)
					newNegativeTrainingExamples.push_back(pair.second);
				vector<Mat> newPositiveTrainingExamples;
				usable = trainable->retrain(newPositiveTrainingExamples, newNegativeTrainingExamples);
				if (usable) {
					if (classifier->getSvm()->getSupportVectors().size() != 1)
						throw runtime_error("ExtendedHogBasedMeasurementModel: the amount of support vectors has to be one (w)");
				}
			}
		}
	}
	targetLost = false;
	learned.emplace(frameIndex, targetBounds);
	frameIndex++;
	return usable;
}

bool ExtendedHogBasedMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<shared_ptr<Sample>>& samples, const Sample& target) {
	if (!usable)
		throw runtime_error("ExtendedHogBasedMeasurementModel: model is not yet usable (was not initialized)");

	targetLost = false;
	vector<Mat> positiveTrainingExamples = createPositiveTrainingExamples(samples, target);
	frameIndex++; // only necessary for output of learned positive patches
	if (positiveTrainingExamples.empty())
		return false;
	vector<Mat> negativeTrainingExamples = createNegativeTrainingExamples(image->getData(), target);
	usable = trainable->retrain(positiveTrainingExamples, negativeTrainingExamples);
	if (usable) {
		if (classifier->getSvm()->getSupportVectors().size() != 1)
			throw runtime_error("ExtendedHogBasedMeasurementModel: the amount of support vectors has to be one (w)");
		convolutionFilter->setKernel(classifier->getSvm()->getSupportVectors()[0]);
		convolutionFilter->setDelta(-classifier->getSvm()->getBias());
	}
	return true;
}

bool ExtendedHogBasedMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<shared_ptr<Sample>>& samples) {
	const vector<Mat> empty;
	usable = trainable->retrain(empty, empty);
	if (usable) {
		if (classifier->getSvm()->getSupportVectors().size() != 1)
			throw runtime_error("ExtendedHogBasedMeasurementModel: the amount of support vectors has to be one (w)");
		convolutionFilter->setKernel(classifier->getSvm()->getSupportVectors()[0]);
		convolutionFilter->setDelta(-classifier->getSvm()->getBias());
	}
	pastFeatureExtractors.clear();
	targetLost = true;
	frameIndex++; // only necessary for output of learned positive patches
	return false;
}

void ExtendedHogBasedMeasurementModel::reset() {
	trainable->reset();
	learned.clear(); // only necessary for output of learned positive patches
	frameIndex = 0; // only necessary for output of learned positive patches
	initialized = false;
	usable = false;
	targetLost = false;
	initialFeatures = Mat();
	trajectoryFeatures.clear();
	trajectoryToLearn.clear();
	pastFeatureExtractors.clear();
}

pair<double, Rect> ExtendedHogBasedMeasurementModel::getHeatPeak() const {
	double bestScore = std::numeric_limits<double>::lowest();
	Rect bestBounds;
	for (const shared_ptr<ImagePyramidLayer>& layer : heatPyramid->getLayers()) {
		int width = layer->getOriginal(cellColumnCount * cellSize);
		int height = layer->getOriginal(cellRowCount * cellSize);
		const Mat& heatMap = layer->getScaledImage();
		for (int rowIndex = cellRowCount / 2, lastRowIndex = heatMap.rows + cellRowCount / 2 - cellRowCount; rowIndex < lastRowIndex; ++rowIndex) {
			const float* rowValues = heatMap.ptr<float>(rowIndex);
			for (int colIndex = cellColumnCount / 2, lastColIndex = heatMap.cols + cellColumnCount / 2 - cellColumnCount; colIndex < lastColIndex; ++colIndex) {
				float score = rowValues[colIndex];
				if (score > bestScore) {
					bestScore = score;
					bestBounds.x = layer->getOriginal((colIndex - cellColumnCount / 2) * cellSize);
					bestBounds.y = layer->getOriginal((rowIndex - cellRowCount / 2) * cellSize);
					bestBounds.width = width;
					bestBounds.height = height;
				}
			}
		}
	}
	return make_pair(bestScore, bestBounds);
}

shared_ptr<Sample> ExtendedHogBasedMeasurementModel::getMean(vector<shared_ptr<Sample>> samples, vector<double> weights) const {
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightedSumVx = 0;
	double weightedSumVy = 0;
	double weightedSumVSize = 0;
	double weightSum = 0;
	for (size_t i = 0; i < samples.size(); ++i) {
		shared_ptr<Sample> sample = samples[i];
		if (sample) {
			double weight = weights[i];
			weightedSumX += weight * sample->getX();
			weightedSumY += weight * sample->getY();
			weightedSumSize += weight * sample->getSize();
			weightedSumVx += weight * sample->getVx();
			weightedSumVy += weight * sample->getVy();
			weightedSumVSize += weight * sample->getVSize();
			weightSum += weight;
		}
	}
	if (weightSum == 0)
		return shared_ptr<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	double weightedMeanVx = weightedSumVx / weightSum;
	double weightedMeanVy = weightedSumVy / weightSum;
	double weightedMeanVSize = weightedSumVSize / weightSum;
	return make_shared<Sample>(
			static_cast<int>(round(weightedMeanX)), static_cast<int>(round(weightedMeanY)), static_cast<int>(round(weightedMeanSize)),
			static_cast<int>(round(weightedMeanVx)), static_cast<int>(round(weightedMeanVy)), static_cast<int>(round(weightedMeanVSize)));
}

vector<Mat> ExtendedHogBasedMeasurementModel::createPositiveTrainingExamples(const vector<shared_ptr<Sample>>& samples, const Sample& target) {
	if (adaptation == Adaptation::NONE)
		return vector<Mat>();

	if (adaptation == Adaptation::POSITION) {
		shared_ptr<Patch> patch = positiveFeatureExtractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
		if (!patch)
			return vector<Mat>();

		double score = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
		if (score <= adaptationThreshold)
			return vector<Mat>();

		learned.emplace(frameIndex, target.getBounds()); // only necessary for output of learned positive patches
		vector<Mat> positiveTrainingExamples;
		positiveTrainingExamples.push_back(patch->getData());
		return positiveTrainingExamples;
	}

	if (adaptation == Adaptation::TRAJECTORY) {
		shared_ptr<Patch> patch = positiveFeatureExtractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
		if (!patch)
			return vector<Mat>();

		double score = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
		if (score > exclusionThreshold) {
			trajectoryFeatures.push_back(patch->getData());
			trajectoryToLearn.emplace_back(frameIndex, target.getBounds()); // only necessary for output of learned positive patches
		}

		if (score <= adaptationThreshold)
			return vector<Mat>();

		vector<Mat> positiveTrainingExamples;
		positiveTrainingExamples.reserve(trajectoryFeatures.size());
		for (const Mat& features : trajectoryFeatures)
			positiveTrainingExamples.push_back(features);
		for (const pair<int, Rect>& elem : trajectoryToLearn) // only necessary for output of learned positive patches
			learned.insert(elem);
		trajectoryFeatures.clear();
		return positiveTrainingExamples;
	}

	if (adaptation == Adaptation::CORRECTED_TRAJECTORY) {

		unordered_map<int, vector<shared_ptr<Sample>>> clusters;
		for (const shared_ptr<Sample>& sample : samples)
			clusters[sample->getClusterId()].push_back(sample);
		auto it = std::max_element(clusters.begin(), clusters.end(),
				[](const pair<int, vector<shared_ptr<Sample>>>& a, const pair<int, vector<shared_ptr<Sample>>>& b) {
						return a.second.size() < b.second.size();
		});
		if (it == clusters.end())
			return vector<Mat>();
		vector<shared_ptr<Sample>>& cluster = it->second;
		vector<double> weights(cluster.size());
		std::transform(cluster.begin(), cluster.end(), weights.begin(), [](const shared_ptr<Sample>& sample) {
			if (sample->isTarget())
				return sample->getWeight();
			return 0.0;
		});
		shared_ptr<Sample> mean = getMean(cluster, weights);
		shared_ptr<Patch> patch = positiveFeatureExtractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
		if (!patch)
			return vector<Mat>();

		double score = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
		if (score <= adaptationThreshold) {
			pastFeatureExtractors.push_front(make_shared<ExtendedHogFeatureExtractor>(*positiveFeatureExtractor));
			return vector<Mat>();
		}

		vector<Mat> positiveTrainingExamples;
		learned.emplace(frameIndex, mean->getBounds()); // only necessary for output of learned positive patches
		positiveTrainingExamples.push_back(patch->getData());
		vector<shared_ptr<Sample>> ancestors = cluster;
		size_t index = frameIndex - 1; // only necessary for output of learned positive patches
		for (const shared_ptr<FeatureExtractor>& extractor : pastFeatureExtractors) {
			std::transform(ancestors.begin(), ancestors.end(), ancestors.begin(), [](const shared_ptr<Sample>& sample) {
				if (sample)
					return sample->getAncestor();
				return sample;
			});
			shared_ptr<Sample> mean = getMean(ancestors, weights);
			if (!mean)
				break;
			shared_ptr<Patch> patch = extractor->extract(target.getX(), target.getY(), target.getWidth(), target.getHeight());
			if (!patch)
				continue;
			double score = classifier->getSvm()->computeHyperplaneDistance(patch->getData());
			if (score > exclusionThreshold) {
				positiveTrainingExamples.push_back(patch->getData());
				learned.emplace(index, mean->getBounds()); // only necessary for output of learned positive patches
			}
			--index; // only necessary for output of learned positive patches
		}
		pastFeatureExtractors.clear();
		return positiveTrainingExamples;
	}

	throw std::runtime_error("ExtendedHogBasedMeasurementModel: adaptation has unsupported value");
}

vector<Mat> ExtendedHogBasedMeasurementModel::createNegativeTrainingExamples(const cv::Mat& image, const Sample& target) const {
	vector<Mat> negativeTrainingExamples;
	if (useSlidingWindow) {
		negativeTrainingExamples = createGoodNegativeExamples(target.getBounds());
		if (negativeTrainingExamples.size() > negativeExampleCount)
			negativeTrainingExamples.resize(negativeExampleCount);
	} else {
		vector<Mat> candidates = createRandomNegativeExamples(randomExampleCount, image, target.getBounds());
		vector<pair<float, Mat>> classifiedCandidates(candidates.size());
		std::transform(candidates.begin(), candidates.end(), classifiedCandidates.begin(), [this](const Mat& example) {
			return std::make_pair(classifier->getSvm()->computeHyperplaneDistance(example), example);
		});
		std::partial_sort(classifiedCandidates.begin(), classifiedCandidates.begin() + negativeExampleCount, classifiedCandidates.end(), [](const pair<float, Mat>& a, const pair<float, Mat>& b) {
			return a.first > b.first;
		});
		size_t count = negativeExampleCount;
		while (count > 0 && classifiedCandidates[count - 1].first <= negativeScoreThreshold)
			count--;
		classifiedCandidates.resize(count);
		negativeTrainingExamples.reserve(classifiedCandidates.size());
		for (const pair<float, Mat>& pair : classifiedCandidates)
			negativeTrainingExamples.push_back(pair.second);
	}
	return negativeTrainingExamples;
}

vector<Mat> ExtendedHogBasedMeasurementModel::createGoodNegativeExamples(Rect targetBounds) const {
	// find local maxima in each pyramid layer
	vector<pair<float, Rect>> candidates;
	for (const shared_ptr<ImagePyramidLayer>& layer : heatPyramid->getLayers()) {
		int width = layer->getOriginal(cellColumnCount * cellSize);
		int height = layer->getOriginal(cellRowCount * cellSize);
		const Mat& heatMap = layer->getScaledImage();
		for (int rowIndex = cellRowCount / 2, lastRowIndex = heatMap.rows + cellRowCount / 2 - cellRowCount; rowIndex < lastRowIndex; ++rowIndex) {
			const float* rowValues = heatMap.ptr<float>(rowIndex);
			for (int colIndex = cellColumnCount / 2, lastColIndex = heatMap.cols + cellColumnCount / 2 - cellColumnCount; colIndex < lastColIndex; ++colIndex) {
				float score = rowValues[colIndex];
				if (score > negativeScoreThreshold) {
					const float* prevRowValues = heatMap.ptr<float>(rowIndex - 1);
					const float* nextRowValues = heatMap.ptr<float>(rowIndex + 1);
					if (score >= rowValues[colIndex - 1]
							&& score >= rowValues[colIndex + 1]
							&& score >= prevRowValues[colIndex - 1]
							&& score >= prevRowValues[colIndex]
							&& score >= prevRowValues[colIndex + 1]
							&& score >= nextRowValues[colIndex - 1]
							&& score >= nextRowValues[colIndex]
							&& score >= nextRowValues[colIndex + 1]) { // local maximum within this layer
						int x = layer->getOriginal((colIndex - cellColumnCount / 2) * cellSize);
						int y = layer->getOriginal((rowIndex - cellRowCount / 2) * cellSize);
						Rect bounds(x, y, width, height);
						if (computeOverlap(targetBounds, bounds) < positiveOverlapThreshold)
							candidates.emplace_back(score, bounds);
					}
				}
			}
		}
	}
	// reduce overlapping candidates using non-maximum suppression
	std::sort(candidates.begin(), candidates.end(), [](const pair<float, Rect>& a, const pair<float, Rect>& b) {
		return a.first < b.first;
	});
	vector<Mat> examples;
	examples.reserve(std::min(static_cast<size_t>(50), candidates.size()));
	while (!candidates.empty()) {
		const Rect box = candidates.back().second;
		candidates.pop_back();
		shared_ptr<Patch> patch = featureExtractor->extract(box.x + box.width / 2, box.y + box.height / 2, box.width, box.height);
		examples.push_back(patch->getData());
		candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const pair<float, Rect>& elem) {
			return computeOverlap(box, elem.second) > negativeOverlapThreshold;
		}), candidates.end());
	}
	return examples;
}

vector<Mat> ExtendedHogBasedMeasurementModel::createRandomNegativeExamples(size_t count, const Mat& image, Rect targetBounds) const {
	vector<Mat> examples;
	examples.reserve(count);
	while (examples.size() < count) {
		Rect bounds = createRandomBounds(image);
		if (computeOverlap(targetBounds, bounds) < positiveOverlapThreshold) {
			shared_ptr<Patch> patch = featureExtractor->extract(
					bounds.x + bounds.width / 2, bounds.y + bounds.height / 2, bounds.width, bounds.height);
			if (patch)
				examples.push_back(patch->getData());
		}
	}
	return examples;
}

Rect ExtendedHogBasedMeasurementModel::createRandomBounds(const Mat& image) const {
	int width = uniformIntDistribution(generator, maxWidth - minWidth) + minWidth;
	int height = width * cellRowCount / cellColumnCount;
	int x = uniformIntDistribution(generator, image.cols - width);
	int y = uniformIntDistribution(generator, image.rows - height);
	return Rect(x, y, width, height);
}

double ExtendedHogBasedMeasurementModel::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

const unordered_map<size_t, Rect>& ExtendedHogBasedMeasurementModel::getLearned() const {
	return learned;
}

void ExtendedHogBasedMeasurementModel::setHogParams(size_t cellSize, size_t cellCount,
		bool signedAndUnsigned, bool interpolateBins, bool interpolateCells, int octaveLayerCount) {
	this->cellSize = cellSize;
	this->cellCount = cellCount;
	this->signedAndUnsigned = signedAndUnsigned;
	this->interpolateBins = interpolateBins;
	this->interpolateCells = interpolateCells;
	this->octaveLayerCount = octaveLayerCount;
}

void ExtendedHogBasedMeasurementModel::setRejectionThreshold(double rejectionThreshold) {
	this->rejectionThreshold = rejectionThreshold;
}

void ExtendedHogBasedMeasurementModel::setUseSlidingWindow(bool useSlidingWindow, bool conservativeReInit) {
	this->useSlidingWindow = useSlidingWindow;
	this->conservativeReInit = conservativeReInit;
}

void ExtendedHogBasedMeasurementModel::setNegativeExampleParams(size_t negativeExampleCount,
		size_t initialNegativeExampleCount, size_t randomExampleCount, float negativeScoreThreshold) {
	this->negativeExampleCount = negativeExampleCount;
	this->initialNegativeExampleCount = initialNegativeExampleCount;
	this->randomExampleCount = randomExampleCount;
	this->negativeScoreThreshold = negativeScoreThreshold;
}

void ExtendedHogBasedMeasurementModel::setOverlapThresholds(double positiveOverlapThreshold, double negativeOverlapThreshold) {
	this->positiveOverlapThreshold = positiveOverlapThreshold;
	this->negativeOverlapThreshold = negativeOverlapThreshold;
}

void ExtendedHogBasedMeasurementModel::setAdaptation(Adaptation adaptation, double adaptationThreshold, double exclusionThreshold) {
	this->adaptation = adaptation;
	this->adaptationThreshold = adaptationThreshold;
	this->exclusionThreshold = exclusionThreshold;
}

} /* namespace condensation */
