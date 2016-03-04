/*
 * RectLandmark.cpp
 *
 *  Created on: 24.05.2013
 *      Author: poschmann
 */

#include "imageio/RectLandmark.hpp"

using cv::Rect_;
using cv::Rect;
using cv::Point;
using cv::Point2f;
using cv::Size2f;
using cv::Vec2f;
using cv::Mat;
using cv::Scalar;
using std::string;

namespace imageio {

RectLandmark::RectLandmark(const Landmark& landmark) :
		Landmark(LandmarkType::RECT, landmark.getName(), landmark.isVisible()), position(landmark.getPosition2D()), size(landmark.getSize()) {}

RectLandmark::RectLandmark(const string& name) :
		Landmark(LandmarkType::RECT, name, false), position(0, 0), size(0, 0) {}

RectLandmark::RectLandmark(const string& name, float x, float y, float width, float height) :
		Landmark(LandmarkType::RECT, name, true), position(x, y), size(width, height) {}

RectLandmark::RectLandmark(const string& name, const Rect& rect) :
		Landmark(LandmarkType::RECT, name, true), position(rect.x + 0.5f * rect.width, rect.y + 0.5f * rect.height), size(rect.width, rect.height) {}

RectLandmark::RectLandmark(const string& name, const Rect_<float>& rect) :
		Landmark(LandmarkType::RECT, name, true), position(rect.x + 0.5f * rect.width, rect.y + 0.5f * rect.height), size(rect.width, rect.height) {}

RectLandmark::RectLandmark(const string& name, const Point2f& position, const Size2f& size) :
		Landmark(LandmarkType::RECT, name, true), position(position.x, position.y), size(size) {}

RectLandmark::RectLandmark(const string& name, const Vec2f& position, const Size2f& size) :
		Landmark(LandmarkType::RECT, name, true), position(position), size(size) {}

RectLandmark::RectLandmark(const string& name, const Vec2f& position, const Size2f& size, bool visible) :
		Landmark(LandmarkType::RECT, name, visible), position(position), size(size) {}

bool RectLandmark::isEqual(const Landmark& landmark) const {
	if (landmark.getType() != LandmarkType::RECT)
		return false;
	float eps = 1e-6;
	return std::abs(getX() - landmark.getX()) < eps
			&& std::abs(getY() - landmark.getY()) < eps
			&& std::abs(getWidth() - landmark.getWidth()) < eps
			&& std::abs(getHeight() - landmark.getHeight()) < eps;
}

bool RectLandmark::isClose(const Landmark& landmark, const float similarity) const {
	if (landmark.getType() != LandmarkType::RECT)
		return false;
	Rect_<float> thisRect = getRect();
	Rect_<float> thatRect = landmark.getRect();
	float intersectionArea = (thisRect & thatRect).area();
	float unionArea = thisRect.area() + thatRect.area() - intersectionArea;
	return intersectionArea / unionArea >= similarity;
}

void RectLandmark::draw(Mat& image, const Scalar& color, float width) const {
	if (isVisible()) {
		Rect_<float> floatRect = getRect();
		Rect intRect(
				Point(cvRound(floatRect.tl().x), cvRound(floatRect.tl().y)),
				Point(cvRound(floatRect.br().x), cvRound(floatRect.br().y)));
		cv::rectangle(image, intRect, color, width);
	}
}

} /* namespace imageio */
