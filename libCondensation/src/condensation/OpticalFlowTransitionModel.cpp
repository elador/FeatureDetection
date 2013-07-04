/*
 * OpticalFlowTransitionModel.cpp
 *
 *  Created on: 14.06.2013
 *      Author: poschmann
 */

#include "condensation/OpticalFlowTransitionModel.hpp"
#include "condensation/Sample.hpp"
#include "opencv2/video/video.hpp"
#include <ctime>
#include <cmath>
#include <stdexcept>

using cv::BORDER_REPLICATE;
using std::make_pair;
using std::sort;
using std::runtime_error;

namespace condensation {

OpticalFlowTransitionModel::OpticalFlowTransitionModel(shared_ptr<TransitionModel> fallback, double scatter,
		Size gridSize, Size windowSize, int maxLevel) :
				fallback(fallback),
				templatePoints(),
				gridSize(gridSize),
				windowSize(windowSize),
				maxLevel(maxLevel),
				scatter(scatter),
				generator(boost::mt19937(time(0)), boost::normal_distribution<>()) {
	float gridY = 1 / static_cast<float>(gridSize.height);
	float gridX = 1 / static_cast<float>(gridSize.width);
	Point2f gridPoint(-0.5f + 0.5 * gridX, -0.5f + 0.5 * gridY);
	for (int y = 0; y < gridSize.height; ++y) {
		for (int x = 0; x < gridSize.width; ++x) {

			// TODO circle test
			float r = 1.f / 2.f;
			if (gridPoint.x * gridPoint.x + gridPoint.y * gridPoint.y < r * r)
				templatePoints.push_back(gridPoint);
			gridPoint.x += gridX;

//				templatePoints.push_back(gridPoint);
//				gridPoint.x += gridX;
		}
		gridPoint.x = -0.5f + gridX / 2;
		gridPoint.y += gridY;
	}
}

OpticalFlowTransitionModel::~OpticalFlowTransitionModel() {}

const Mat OpticalFlowTransitionModel::makeGrayscale(const Mat& image) const {
	if (image.channels() == 1)
		return image;
	Mat grayscale;
	cvtColor(image, grayscale, CV_BGR2GRAY);
	return grayscale;
}

void OpticalFlowTransitionModel::init(const Mat& image) {
	cv::buildOpticalFlowPyramid(makeGrayscale(image), previousPyramid, windowSize, maxLevel, true, BORDER_REPLICATE, BORDER_REPLICATE);
}

void OpticalFlowTransitionModel::predict(vector<Sample>& samples, const Mat& image, const optional<Sample>& target) {
	points.clear();
	forwardPoints.clear();
	backwardPoints.clear();
	forwardStatus.clear();
	error.clear();
	squaredDistances.clear();
	correctFlowCount = 0;
//	if (previousPyramid.empty())
//		throw runtime_error("OpticalFlowTransitionModel: no previous pyramid (not properly initialized?)");
	if (previousPyramid.empty()) { // TODO temporary or not? maybe remove exception stuff
		cv::buildOpticalFlowPyramid(makeGrayscale(image), previousPyramid, windowSize, maxLevel, true, BORDER_REPLICATE, BORDER_REPLICATE);
		fallback->predict(samples, image, target);
		return;
	}
	cv::buildOpticalFlowPyramid(makeGrayscale(image), currentPyramid, windowSize, maxLevel, true, BORDER_REPLICATE, BORDER_REPLICATE);

	// TODO compute optical flow
	if (target) {
		// TODO compute grid of points at target location
		for (auto point = templatePoints.begin(); point != templatePoints.end(); ++point)
			points.push_back(Point2f(target->getWidth() * point->x + target->getX(), target->getHeight() * point->y + target->getY()));
		// TODO compute forward and backward optical flow
		cv::calcOpticalFlowPyrLK(previousPyramid, currentPyramid, points, forwardPoints, forwardStatus, error, windowSize, maxLevel);
		cv::calcOpticalFlowPyrLK(currentPyramid, previousPyramid, forwardPoints, backwardPoints, backwardStatus, error, windowSize, maxLevel);
		swap(previousPyramid, currentPyramid);
		// TODO compute forward-backward-error
		vector<Point2f> flows;
		flows.reserve(points.size());
		squaredDistances.reserve(points.size());
		for (unsigned int i = 0; i < points.size(); ++i) {
			flows.push_back(forwardPoints[i] - points[i]);
			if (forwardStatus[i] && backwardStatus[i]) {
				Point2f difference = backwardPoints[i] - points[i];
				squaredDistances.push_back(make_pair(difference.dot(difference), i));
			}
		}
		// TODO fallback
		if (squaredDistances.size() < points.size() / 2) {
			fallback->predict(samples, image, target);
			return;
		}
		sort(squaredDistances.begin(), squaredDistances.end(), [](pair<float, int> lhs, pair<float, int> rhs) { return lhs.first < rhs.first; });
		// TODO compute median move with first half of stuffs
		correctFlowCount = points.size() / 2;
		vector<float> xs, ys;
//		xs.reserve(correctFlowCount);
//		ys.reserve(correctFlowCount);
//		for (unsigned int i = 0; i < correctFlowCount; ++i) {
//			int index = squaredDistances[i].second;
//			xs.push_back(flows[index].x);
//			ys.push_back(flows[index].y);
//		}

		// TODO test with distance-threshold
		xs.reserve(squaredDistances.size());
		ys.reserve(squaredDistances.size());
		for (correctFlowCount = 0; correctFlowCount < squaredDistances.size(); ++correctFlowCount) {
			if (squaredDistances[correctFlowCount].first > 1 * 1)
				break;
			int index = squaredDistances[correctFlowCount].second;
			xs.push_back(flows[index].x);
			ys.push_back(flows[index].y);
		}
		if (correctFlowCount < points.size() / 2) { // to little correct correspondences
			// TODO fallback
			fallback->predict(samples, image, target);
			return;
		}

		sort(xs.begin(), xs.end());
		sort(ys.begin(), ys.end());
		float medianX = xs[xs.size() / 2];
		float medianY = ys[ys.size() / 2];
		// TODO compute distance to median move
		Point2f medianFlow(medianX, medianY);
		vector<float> squaredDistancesToMedianFlow;
		squaredDistancesToMedianFlow.reserve(flows.size());
		for (auto flow = flows.begin(); flow != flows.end(); ++flow) {
			Point2f difference = *flow - medianFlow;
			squaredDistancesToMedianFlow.push_back(difference.dot(difference));
		}
		sort(squaredDistancesToMedianFlow.begin(), squaredDistancesToMedianFlow.end());
		// TODO compute median of distance to median move
		float squaredMedianDistance = squaredDistancesToMedianFlow[squaredDistancesToMedianFlow.size() / 2];
		if (squaredMedianDistance > 10 * 10) {
			// TODO fallback
			fallback->predict(samples, image, target);
			return;
		}
		// TODO compute ratios of point distances before and after
		vector<float> squaredRatios;
		squaredRatios.reserve(correctFlowCount * (correctFlowCount - 1) / 2);
		for (unsigned int i = 0; i < correctFlowCount; ++i) {
			for (unsigned int j = i + 1; j < correctFlowCount; ++j) {
				Point2f point1 = points[squaredDistances[i].second];
				Point2f forwardPoint1 = forwardPoints[squaredDistances[i].second];
				Point2f point2 = points[squaredDistances[j].second];
				Point2f forwardPoint2 = forwardPoints[squaredDistances[j].second];
				Point2f differenceBefore = point1 - point2;
				Point2f differenceAfter = forwardPoint1 - forwardPoint2;
				float squaredDistanceBefore = differenceBefore.dot(differenceBefore);
				float squaredDistanceAfter = differenceAfter.dot(differenceAfter);
				squaredRatios.push_back(squaredDistanceAfter / squaredDistanceBefore);
			}
		}
		sort(squaredRatios.begin(), squaredRatios.end());
		float medianRatio = sqrt(squaredRatios[squaredRatios.size() / 2]);
		// TODO predict samples according to median flow
		for (auto sample = samples.begin(); sample != samples.end(); ++sample) {
			int oldX = sample->getX();
			int oldY = sample->getY();
			float oldSize = sample->getSize();
			// change position according to median flow
			double newX = oldX + medianX;
			double newY = oldY + medianY;
			double newSize = oldSize * medianRatio;
			// add noise to position
			double positionDeviation = scatter * sample->getSize();
			newX += positionDeviation * generator();
			newY += positionDeviation * generator();
			newSize *= pow(2, scatter * generator());
			// round to integer
			sample->setX((int)(newX + 0.5));
			sample->setY((int)(newY + 0.5));
			sample->setSize((int)(newSize + 0.5));
			// compute change
			sample->setVx(sample->getX() - oldX);
			sample->setVy(sample->getY() - oldY);
			sample->setVSize(sample->getSize() / oldSize);
		}
	} else {
		// TODO fallback
		fallback->predict(samples, image, target);
	}
}

void OpticalFlowTransitionModel::drawFlow(Mat& image, Scalar color, int thickness) const {
	for (unsigned int i = 0; i < correctFlowCount; ++i) {
		int index = squaredDistances[i].second;
		cv::line(image, points[index], forwardPoints[index], Scalar(0, 255, 0), thickness);
	}
	for (unsigned int i = correctFlowCount; i < squaredDistances.size(); ++i) {
		int index = squaredDistances[i].second;
		cv::line(image, points[index], forwardPoints[index], Scalar(0, 0, 255), thickness);
	}

//	float half = 2 * 2;
//	float full = 4 * 4;
//	for (auto distance = squaredDistances.begin(); distance != squaredDistances.end(); ++distance) {
//		int index = distance->second;
//		const Scalar& color = distance->first <= half
//				? Scalar(0, 255, cvRound(255 * distance->first / half))
//						: Scalar(0, std::max(0, cvRound((distance->first - half) / (full - half)) * 255), 255);
//		cv::line(image, points[index], forwardPoints[index], color, thickness);
//	}

//	for (auto distance = squaredDistances.begin(); distance != squaredDistances.end(); ++distance) {
//		int index = distance->second;
//		cv::line(image, points[index], forwardPoints[index], color, thickness);
//	}
}

} /* namespace condensation */
