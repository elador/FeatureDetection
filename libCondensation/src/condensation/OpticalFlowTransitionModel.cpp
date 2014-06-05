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

using cv::Mat;
using cv::Size;
using cv::Scalar;
using cv::Point2f;
using cv::BORDER_REPLICATE;
using std::sort;
using std::pair;
using std::vector;
using std::make_pair;
using std::shared_ptr;

namespace condensation {

OpticalFlowTransitionModel::OpticalFlowTransitionModel(shared_ptr<TransitionModel> fallback,
		double positionDeviation, double sizeDeviation, Size gridSize, bool circle, Size windowSize, int maxLevel) :
				fallback(fallback),
				templatePoints(),
				gridSize(gridSize),
				windowSize(windowSize),
				maxLevel(maxLevel),
				previousPyramid(),
				currentPyramid(),
				points(),
				forwardPoints(),
				backwardPoints(),
				forwardStatus(),
				backwardStatus(),
				error(),
				squaredDistances(),
				correctFlowCount(0),
				positionDeviation(positionDeviation),
				sizeDeviation(sizeDeviation),
				generator(boost::mt19937(time(0)), boost::normal_distribution<>()) {
//	float gridY = 1 / static_cast<float>(gridSize.height);
//	float gridX = 1 / static_cast<float>(gridSize.width);
//	Point2f gridPoint(-0.5f + 0.5 * gridX, -0.5f + 0.5 * gridY);
//	for (int y = 0; y < gridSize.height; ++y) {
//		for (int x = 0; x < gridSize.width; ++x) {
//			float r = 1.f / 2.f;
//			if (!circle || gridPoint.x * gridPoint.x + gridPoint.y * gridPoint.y < r * r)
//				templatePoints.push_back(gridPoint);
//			gridPoint.x += gridX;
//		}
//		gridPoint.x = -0.5f + gridX / 2;
//		gridPoint.y += gridY;
//	}

	// TODO die variante erscheint bissl besser (größerer abstand der punkte zum rand)
	// braucht aber evtl. bissl mehr scatter nötig
	float gridY = 1.f / static_cast<float>(gridSize.height + 2);
	float gridX = 1.f / static_cast<float>(gridSize.width + 2);
	float r = 0.5f - 0.5f * (gridX + gridY);
	Point2f gridPoint(-0.5f + 0.5 * gridX + gridX, -0.5f + 0.5 * gridY + gridY);
	for (int y = 0; y < gridSize.height; ++y) {
		for (int x = 0; x < gridSize.width; ++x) {
			if (!circle || gridPoint.x * gridPoint.x + gridPoint.y * gridPoint.y < r * r)
				templatePoints.push_back(gridPoint);
			gridPoint.x += gridX;
		}
		gridPoint.x = -0.5f + gridX / 2 + gridX;
		gridPoint.y += gridY;
	}
}

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

void OpticalFlowTransitionModel::predict(vector<shared_ptr<Sample>>& samples, const Mat& image, const shared_ptr<Sample> target) {
	points.clear();
	forwardPoints.clear();
	backwardPoints.clear();
	forwardStatus.clear();
	error.clear();
	squaredDistances.clear();
	correctFlowCount = 0;

	// build pyramid of the current image
	cv::buildOpticalFlowPyramid(makeGrayscale(image), currentPyramid, windowSize, maxLevel, true, BORDER_REPLICATE, BORDER_REPLICATE);
	if (previousPyramid.empty() || !target) { // optical flow cannot be computed if there is no previous pyramid or no current target
		swap(previousPyramid, currentPyramid);
		return fallback->predict(samples, image, target);
	}

	// compute grid of points at target location
	for (auto point = templatePoints.begin(); point != templatePoints.end(); ++point)
		points.push_back(Point2f(target->getWidth() * point->x + target->getX(), target->getHeight() * point->y + target->getY()));
	// compute forward and backward optical flow
	cv::calcOpticalFlowPyrLK(previousPyramid, currentPyramid, points, forwardPoints, forwardStatus, error, windowSize, maxLevel);
	cv::calcOpticalFlowPyrLK(currentPyramid, previousPyramid, forwardPoints, backwardPoints, backwardStatus, error, windowSize, maxLevel);
	swap(previousPyramid, currentPyramid);

	// compute flow and forward-backward-error
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
	if (squaredDistances.size() < points.size() / 2) // the flow for more than half of the points could not be computed
		return fallback->predict(samples, image, target);

	// find the flows with the least forward-backward-error (not more than 1 pixel)
	float maxError = 1 * 0.5f;
	vector<float> xs, ys;
	sort(squaredDistances.begin(), squaredDistances.end(), [](pair<float, int> lhs, pair<float, int> rhs) { return lhs.first < rhs.first; });
	xs.reserve(squaredDistances.size());
	ys.reserve(squaredDistances.size());
	for (correctFlowCount = 0; correctFlowCount < squaredDistances.size(); ++correctFlowCount) {
		if (squaredDistances[correctFlowCount].first > maxError)
			break;
		int index = squaredDistances[correctFlowCount].second;
		xs.push_back(flows[index].x);
		ys.push_back(flows[index].y);
	}
	if (correctFlowCount < points.size() / 2) // too few correct correspondences
		return fallback->predict(samples, image, target);

	// compute median flow (only change in position for now)
	sort(xs.begin(), xs.end());
	sort(ys.begin(), ys.end());
	float medianX = xs[xs.size() / 2];
	float medianY = ys[ys.size() / 2];
	Point2f medianFlow(medianX, medianY);

	// compute ratios of point distances in previous and current image
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

	// compute median ratio to complete the median flow
	sort(squaredRatios.begin(), squaredRatios.end());
	float medianRatio = sqrt(squaredRatios[squaredRatios.size() / 2]);

	// predict samples according to median flow and random noise
	for (shared_ptr<Sample> sample : samples) {
		// add noise to velocity
		double vx = medianX;
		double vy = medianY;
		double vs = medianRatio;
		// diffuse
		vx += positionDeviation * generator();
		vy += positionDeviation * generator();
		vs *= pow(2, sizeDeviation * generator());
		// round to integer
		sample->setVx(static_cast<int>(std::round(vx)));
		sample->setVy(static_cast<int>(std::round(vy)));
		sample->setVSize(vs);
		// change position according to velocity
		sample->setX(sample->getX() + sample->getVx());
		sample->setY(sample->getY() + sample->getVy());
		sample->setSize(static_cast<int>(std::round(sample->getSize() * sample->getVSize())));
	}
}

void OpticalFlowTransitionModel::drawFlow(Mat& image, int thickness, Scalar color, Scalar badColor) const {
	if (thickness < 0) {
		for (unsigned int i = 0; i < correctFlowCount; ++i) {
			int index = squaredDistances[i].second;
			cv::circle(image, forwardPoints[index], -thickness, color, -1);
		}
	} else {
		for (unsigned int i = 0; i < correctFlowCount; ++i) {
			int index = squaredDistances[i].second;
			cv::line(image, points[index], forwardPoints[index], color, thickness);
		}
		if (badColor[0] >= 0 && badColor[1] >= 0 && badColor[2] >= 0) {
			for (unsigned int i = correctFlowCount; i < squaredDistances.size(); ++i) {
				int index = squaredDistances[i].second;
				cv::line(image, points[index], forwardPoints[index], badColor, thickness);
			}
		}
	}
}

} /* namespace condensation */
