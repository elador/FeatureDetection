/*
 * ModelLandmark.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/ModelLandmark.hpp"

using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Scalar;
using std::make_pair;
using std::string;
using std::array;
using std::map;

namespace imageio {

ModelLandmark::ModelLandmark(const string& name) :
		Landmark(LandmarkType::MODEL, name, false), position(0, 0, 0) {}

ModelLandmark::ModelLandmark(const string& name, float x, float y, float z) :
		Landmark(LandmarkType::MODEL, name, true), position(x, y, z) {}

ModelLandmark::ModelLandmark(const string& name, const Vec2f& position) :
		Landmark(LandmarkType::MODEL, name, true), position(position[0], position[1], 0) {}

ModelLandmark::ModelLandmark(const string& name, const Vec3f& position) :
		Landmark(LandmarkType::MODEL, name, true), position(position) {}

ModelLandmark::ModelLandmark(const string& name, const Vec3f& position, bool visible) :
		Landmark(LandmarkType::MODEL, name, visible), position(position) {}

bool ModelLandmark::isEqual(const Landmark& landmark) const
{
	if (landmark.getType() != LandmarkType::MODEL)
		return false;
	// TODO implement this function
	return false;
}

bool ModelLandmark::isClose(const Landmark& landmark, const float similarity) const
{
	if (landmark.getType() != LandmarkType::MODEL)
		return false;
	// TODO implement this function
	return false;
}

void ModelLandmark::draw(Mat& image, const Scalar& color, float width) const
{
	if (isVisible()) {
		//cv::Point2i realFfpCenter(patch->c.x+patch->w_inFullImg*thisLm.displacementFactorW, patch->c.y+patch->h_inFullImg*thisLm.displacementFactorH);
		array<bool, 9> symbol = LandmarkSymbols::get(getName());
		Scalar color = LandmarkSymbols::getColor(getName());
		unsigned int pos = 0;
		for (int currRow = cvRound(position[1])-1; currRow<=cvRound(position[1])+1; ++currRow) {
			for (int currCol = cvRound(position[0])-1; currCol<=cvRound(position[0])+1; ++currCol) {
				if (symbol[pos]==true) {
					if (image.channels()==4) {
						image.at<cv::Vec4b>(currRow, currCol)[0] = (uchar)cvRound(255.0f * color.val[0]);
						image.at<cv::Vec4b>(currRow, currCol)[1] = (uchar)cvRound(255.0f * color.val[1]);
						image.at<cv::Vec4b>(currRow, currCol)[2] = (uchar)cvRound(255.0f * color.val[2]);
					} else if (image.channels()==3) {
						image.at<cv::Vec3b>(currRow, currCol)[0] = (uchar)cvRound(255.0f * color.val[0]);
						image.at<cv::Vec3b>(currRow, currCol)[1] = (uchar)cvRound(255.0f * color.val[1]);
						image.at<cv::Vec3b>(currRow, currCol)[2] = (uchar)cvRound(255.0f * color.val[2]);
					} // TODO could add chans=1?
				}
				++pos;
			}
		}
	}
}

map<string, array<bool, 9>> LandmarkSymbols::symbolMap;
map<string, cv::Scalar> LandmarkSymbols::colorMap;

array<bool, 9> LandmarkSymbols::get(string landmarkName)
{
	if (symbolMap.empty()) {
		array<bool, 9> reye_c		= {	false, true, false,
										false, true, true,
										false, false, false };
		symbolMap.insert(make_pair("right.eye.pupil.center", reye_c));	// Use an initializer list as soon as msvc supports it...

		array<bool, 9> leye_c	= {	false, true, false,
									true, true, false,
									false, false, false };
		symbolMap.insert(make_pair("left.eye.pupil.center", leye_c));

		array<bool, 9> nose_tip	= {	false, false, false,
									false, true, false,
									true, false, true };
		symbolMap.insert(make_pair("center.nose.tip", nose_tip));

		array<bool, 9> mouth_rc	= {	false, false, true,
									false, true, false,
									false, false, true };
		symbolMap.insert(make_pair("right.lips.corner", mouth_rc));

		array<bool, 9> mouth_lc	= {	true, false, false,
									false, true, false,
									true, false, false };
		symbolMap.insert(make_pair("left.lips.corner", mouth_lc));

		array<bool, 9> reye_oc	= {	false, true, false,
									false, true, true,
									false, true, false };
		symbolMap.insert(make_pair("right.eye.corner_outer", reye_oc));

		array<bool, 9> leye_oc	= {	false, true, false,
									true, true, false,
									false, true, false };
		symbolMap.insert(make_pair("left.eye.corner_outer", leye_oc));

		array<bool, 9> mouth_ulb	= {	false, false, false,
										true, true, true,
										false, true, false };
		symbolMap.insert(make_pair("center.lips.upper.outer", mouth_ulb));

		array<bool, 9> nosetrill_r	= {	true, false, false,
										true, true, true,
										false, false, false };
		symbolMap.insert(make_pair("right.nose.wing.tip", nosetrill_r));

		array<bool, 9> nosetrill_l	= {	false, false, true,
										true, true, true,
										false, false, false };
		symbolMap.insert(make_pair("left.nose.wing.tip", nosetrill_l));

		array<bool, 9> rear_DONTKNOW	= {	false, true, true,
											false, true, false,
											false, true, true };
		symbolMap.insert(make_pair("right.ear.DONTKNOW", rear_DONTKNOW)); // right.ear.(antihelix.tip | lobule.center | lobule.attachement)

		array<bool, 9> lear_DONTKNOW	= {	true, true, false,
											false, true, false,
											true, true, false };
		symbolMap.insert(make_pair("left.ear.DONTKNOW", lear_DONTKNOW));

	}
	const auto symbol = symbolMap.find(landmarkName);
	if (symbol == symbolMap.end()) {
		array<bool, 9> unknownLmSymbol	= {	true, false, true,
											false, true, false,
											true, false, true };
		return unknownLmSymbol;
	}
	return symbol->second;
}

Scalar LandmarkSymbols::getColor(string landmarkName)
{
	if (colorMap.empty()) {
		Scalar reye_c(0.0f, 0.0f, 1.0f);
		colorMap.insert(make_pair("right.eye.pupil.center", reye_c));	// Use an initializer list as soon as msvc supports it...

		Scalar leye_c(1.0f, 0.0f, 0.0f);
		colorMap.insert(make_pair("left.eye.pupil.center", leye_c));

		Scalar nose_tip(0.0f, 1.0f, 0.0f);
		colorMap.insert(make_pair("center.nose.tip", nose_tip));

		Scalar mouth_rc(0.0f, 1.0f, 1.0f);
		colorMap.insert(make_pair("right.lips.corner", mouth_rc));

		Scalar mouth_lc(1.0f, 0.0f, 1.0f);
		colorMap.insert(make_pair("left.lips.corner", mouth_lc));

		Scalar reye_oc(0.0f, 0.0f, 0.48f);
		colorMap.insert(make_pair("right.eye.corner_outer", reye_oc));

		Scalar leye_oc(1.0f, 1.0f, 0.0f);
		colorMap.insert(make_pair("left.eye.corner_outer", leye_oc));

		Scalar mouth_ulb(0.63f, 0.75f, 0.9f);
		colorMap.insert(make_pair("center.lips.upper.outer", mouth_ulb));

		Scalar nosetrill_r(0.27f, 0.27f, 0.67f);
		colorMap.insert(make_pair("right.nose.wing.tip", nosetrill_r));

		Scalar nosetrill_l(0.04f, 0.78f, 0.69f);
		colorMap.insert(make_pair("left.nose.wing.tip", nosetrill_l));

		Scalar rear_DONTKNOW(1.0f, 0.0f, 0.52f);
		colorMap.insert(make_pair("right.ear.DONTKNOW", rear_DONTKNOW)); // right.ear.(antihelix.tip | lobule.center | lobule.attachement)

		Scalar lear_DONTKNOW(0.0f, 0.6f, 0.0f);
		colorMap.insert(make_pair("left.ear.DONTKNOW", lear_DONTKNOW));

	}
	const auto symbol = colorMap.find(landmarkName);
	if (symbol == colorMap.end()) {
		Scalar unknownLmColor(0.35f, 0.35f, 0.35f);
		return unknownLmColor;
	}
	return symbol->second;
}

} /* namespace imageio */
