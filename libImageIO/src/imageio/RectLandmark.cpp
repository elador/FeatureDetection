/*
 * RectLandmark.cpp
 *
 *  Created on: 24.05.2013
 *      Author: poschmann
 */

#include "imageio/RectLandmark.hpp"

using cv::Point_;
using std::max;
using std::min;

namespace imageio {

RectLandmark::RectLandmark(const Landmark& landmark) :
		Landmark(RECT, landmark.getName(), landmark.isVisible()), position(landmark.getPosition2D()), size(landmark.getSize()) {}

RectLandmark::RectLandmark(const string& name) :
		Landmark(RECT, name, false), position(0, 0), size(0, 0) {}

RectLandmark::RectLandmark(const string& name, float x, float y, float width, float height) :
		Landmark(RECT, name, true), position(x, y), size(width, height) {}

RectLandmark::RectLandmark(const string& name, const Rect& rect) :
		Landmark(RECT, name, true), position(rect.x + 0.5f * rect.width, rect.y + 0.5f * rect.height), size(rect.width, rect.height) {}

RectLandmark::RectLandmark(const string& name, const Rect_<float>& rect) :
		Landmark(RECT, name, true), position(rect.x + 0.5f * rect.width, rect.y + 0.5f * rect.height), size(rect.width, rect.height) {}

RectLandmark::RectLandmark(const string& name, const Point2f& position, const Size2f& size) :
		Landmark(RECT, name, true), position(position.x, position.y), size(size) {}

RectLandmark::RectLandmark(const string& name, const Vec2f& position, const Size2f& size) :
		Landmark(RECT, name, true), position(position), size(size) {}

RectLandmark::RectLandmark(const string& name, const Vec2f& position, const Size2f& size, bool visible) :
		Landmark(RECT, name, visible), position(position), size(size) {}

RectLandmark::~RectLandmark() {}

bool RectLandmark::isEqual(const Landmark& landmark) const {
	if (landmark.getType() != RECT)
		return false;
	float eps = 1e-6;
	return abs(getX() - landmark.getX()) < eps && abs(getY() - landmark.getY()) < eps
			&& abs(getWidth() - landmark.getWidth()) < eps && abs(getHeight() - landmark.getHeight()) < eps;
}

bool RectLandmark::isClose(const Landmark& landmark, const float similarity) const {
	if (landmark.getType() != RECT)
		return false;
	Rect_<float> thisRect = getRect();
	Rect_<float> thatRect = landmark.getRect();
	float intersectionArea = (thisRect & thatRect).area();
	float unionArea = thisRect.area() + thatRect.area() - intersectionArea;
	return intersectionArea / unionArea >= similarity;
}

void RectLandmark::draw(Mat& image, const Scalar& color, float width) const {
	if (isVisible())
		cv::rectangle(image, getRect(), color, width);
}

} /* namespace imageio */
