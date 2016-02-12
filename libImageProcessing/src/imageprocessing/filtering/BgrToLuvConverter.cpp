/*
 * BgrToLuvConverter.cpp
 *
 *  Created on: 16.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/BgrToLuvConverter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec3b;
using cv::Vec3f;
using std::invalid_argument;

namespace imageprocessing {
namespace filtering {

BgrToLuvConverter::BgrToLuvConverter(bool maintainPerceptualUniformity) :
		maintainPerceptualUniformity(maintainPerceptualUniformity), luvLut() {
	createLuvLut();
}

void BgrToLuvConverter::createLuvLut() {
	createBgrFloat(luvLut);
	convertToLuvFloat(luvLut, luvLut);
	normalizeLuv(luvLut);
}

void BgrToLuvConverter::createBgrFloat(Mat& bgrImage) {
	bgrImage.create(1, 256 * 256 * 256, CV_32FC3);
	Vec3f* bgrValues = bgrImage.ptr<Vec3f>();
	for (int blue = 0; blue < 256; ++blue) {
		for (int green = 0; green < 256; ++green) {
			for (int red = 0; red < 256; ++red) {
				Vec3f& bgr = *bgrValues;
				bgr[0] = blue / 255.0;
				bgr[1] = green / 255.0;
				bgr[2] = red / 255.0;
				++bgrValues;
			}
		}
	}
}

Mat BgrToLuvConverter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() == CV_8UC3) {
		convertToNormalizedLuvUchar(image, filtered);
	} else if (image.type() == CV_32FC3) {
		convertToLuvFloat(image, filtered);
		normalizeLuv(filtered);
	} else {
		throw invalid_argument("BgrToLuvConverter: image must be of type CV_8UC3 or CV_32FC3, but was " + std::to_string(image.type()));
	}
	return filtered;
}

void BgrToLuvConverter::convertToNormalizedLuvUchar(const Mat& image, Mat& filtered) const {
	filtered.create(image.rows, image.cols, CV_32FC3);
	for (int row = 0; row < image.rows; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			convertToNormalizedLuv(image.at<Vec3b>(row, col), filtered.at<Vec3f>(row, col));
		}
	}
}

void BgrToLuvConverter::convertToNormalizedLuv(const Vec3b& bgr, Vec3f& luv) const {
	int blue = bgr[0];
	int green = bgr[1];
	int red = bgr[2];
	int index = (blue << 16) | (green << 8) | red;
	luv = luvLut.ptr<Vec3f>()[index];
}

void BgrToLuvConverter::convertToLuvFloat(const Mat& image, Mat& filtered) const {
	cv::cvtColor(image, filtered, CV_BGR2Luv);
}

void BgrToLuvConverter::normalizeLuv(Mat& image) const {
	for (int rowIndex = 0; rowIndex < image.rows; ++rowIndex) {
		Vec3f* row = image.ptr<Vec3f>(rowIndex);
		for (int colIndex = 0; colIndex < image.cols; ++colIndex) {
			normalize(row[colIndex]);
		}
	}
}

void BgrToLuvConverter::normalize(cv::Vec3f& luv) const {
	if (maintainPerceptualUniformity) {
		luv[0] =  luv[0]        / 354;
		luv[1] = (luv[1] + 134) / 354;
		luv[2] = (luv[2] + 140) / 354;
	} else {
		luv[0] =  luv[0]        / 100;
		luv[1] = (luv[1] + 134) / 354;
		luv[2] = (luv[2] + 140) / 262;
	}
}

} /* namespace filtering */
} /* namespace imageprocessing */
