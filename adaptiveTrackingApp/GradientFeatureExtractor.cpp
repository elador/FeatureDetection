/*
 * GradientFeatureExtractor.cpp
 *
 *  Created on: 23.05.2013
 *      Author: poschmann
 */

#include "GradientFeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/Patch.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/lexical_cast.hpp"
#include <string>
#include <stdexcept>
#include <iostream> // TODO
using boost::lexical_cast;
using std::make_shared;
using std::string;

namespace imageprocessing {

GradientFeatureExtractor::GradientFeatureExtractor() : imageFilter(), version(-1), grayscaleImage(), integralImage(), areas() {
	for (float featureSize = 0.1; featureSize < 0.5; featureSize *= 2) {
		for (float featureX = 0; featureX + featureSize <= 1; featureX += 0.5 * featureSize) {
			for (float featureY = 0; featureY + featureSize <= 1; featureY += 0.5 * featureSize) {
				areas.push_back(Rect_<float>(featureX, featureY, featureSize, featureSize));
			}
		}
	}
}

GradientFeatureExtractor::~GradientFeatureExtractor() {}

void GradientFeatureExtractor::update(const Mat& image) {
	version = -1;
	imageFilter.applyTo(image, grayscaleImage);
	cv::integral(grayscaleImage, integralImage);
}

void GradientFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	if (version != image->getVersion()) {
		version = image->getVersion();
		imageFilter.applyTo(image->getData(), grayscaleImage);
		cv::integral(grayscaleImage, integralImage);
//		std::cout << "image " << image->getData().cols << " x " << image->getData().rows << std::endl;
//		std::cout << "type: " << grayscaleImage.type() << " " << integralImage.type() << std::endl;
//		for (int x = 0; x < integralImage.cols; ++x) {
//			for (int y = 0; y < integralImage.rows; ++y) {
//				double value = integralImage.at<double>(y, x);
//				if (value != value)
//					std::cout << x << ", " << y << " -> " << integralImage.at<double>(y - 1, x) << " + " << integralImage.at<double>(y, x - 1) << " - " << integralImage.at<double>(y - 1, x - 1) << std::endl;
//			}
//		}
//		for (int x = 0; x < integralImage.cols; ++x) {
//			for (int y = 0; y < integralImage.rows; ++y) {
//				int value = integralImage.at<int>(y, x);
//				if (value < 0) {
//					std::cout << x << ", " << y << " -> " << value << " = " << integralImage.at<int>(y - 1, x) << " + " << integralImage.at<int>(y, x - 1) << " - " << integralImage.at<int>(y - 1, x - 1) << std::endl;
//					throw std::runtime_error("stop - hammertime!");
//				}
//			}
//		}
//		for (int x = 1; x < integralImage.cols; ++x) {
//			for (int y = 1; y < integralImage.rows; ++y) {
//				int value = integralImage.at<int>(y, x);
//				if (value <= 0) {
//					std::cout << x << ", " << y << " -> " << value << " = " << integralImage.at<int>(y - 1, x) << " + " << integralImage.at<int>(y, x - 1) << " - " << integralImage.at<int>(y - 1, x - 1) << std::endl;
//					throw std::runtime_error("stop - hammertime!");
//				}
//			}
//		}
//		throw std::runtime_error("stop - hammertime!");
	}
}

shared_ptr<Patch> GradientFeatureExtractor::extract(int x, int y, int width, int height) const {
	if (x < 0 || x + width >= integralImage.cols || y < 0 || y + height >= integralImage.rows)
		return shared_ptr<Patch>();
//	double sum = 0;
	Mat data(1, 2 * areas.size(), CV_32F);
	float* dataValues = data.ptr<float>(0);
	for (unsigned int i = 0; i < areas.size(); ++i) {
		const Rect_<float>& area = areas[i];
		float featureX = x + area.x * width;
		float featureY = y + area.y * height;
		float featureWidth = area.width * width;
		float featureHeight = area.height * height;
		double leftArea = getAreaSum(featureX, featureY, featureX + 0.5f * featureWidth, featureY + featureHeight);
		double rightArea = getAreaSum(featureX + 0.5f * featureWidth, featureY, featureX + featureWidth, featureY + featureHeight);
		double topArea = getAreaSum(featureX, featureY, featureX + featureWidth, featureY + 0.5f * featureHeight);
		double bottomArea = getAreaSum(featureX, featureY + 0.5f * featureHeight, featureX + featureWidth, featureY + featureHeight);
		double gradientX = (rightArea - leftArea) / (0.5 * featureWidth * featureHeight);
		double gradientY = (bottomArea - topArea) / (0.5 * featureWidth * featureHeight);
		dataValues[2 * i] = static_cast<float>(gradientX);
		dataValues[2 * i + 1] = static_cast<float>(gradientY);
//		std::cout << gradientX << " " << gradientY;
//		sum += gradientX;
//		sum += gradientY;
//		if (gradientX != gradientX)
//			std::cout << "right " << rightArea << " left " << leftArea << " denom " << (0.5f * featureWidth * featureHeight) << std::endl;
//		if (gradientY != gradientY)
//			std::cout << "top " << topArea << " bottom " << bottomArea << " denom " << (0.5f * featureWidth * featureHeight) << std::endl;
	}
//	std::cout << std::endl;
//	std::cout << (sum / (2 * areas.size())) << std::endl;
	return make_shared<Patch>(x, y, width, height, data);
}

//static void check(const Mat& ii, int x1, int x2) {
//	float value = ii.at<double>(x1, x2);
//	if (value != value)
//		std::cout << x1 << "," << x2 << " -> " << value << std::endl;
//}

int GradientFeatureExtractor::getSum(int x, int y) const {
	return integralImage.at<int>(y, x);
}

double GradientFeatureExtractor::getAreaSum(float x1, float y1, float x2, float y2) const {
	int x1d = static_cast<int>(floor(x1));
	int x1u = static_cast<int>(ceil(x1));
	int y1d = static_cast<int>(floor(y1));
	int y1u = static_cast<int>(ceil(y1));
	int x2d = static_cast<int>(floor(x2));
	int x2u = static_cast<int>(ceil(x2));
	int y2d = static_cast<int>(floor(y2));
	int y2u = static_cast<int>(ceil(y2));
	if (integralImage.type() == CV_32S) {
//		const Mat& ii = integralImage;
//		check(ii, x1d, y1d); check(ii, x1d, y1u); check(ii, x1u, y1d); check(ii, x1u, y1u);
//		check(ii, x1d, y2d); check(ii, x1d, y2u); check(ii, x1u, y2d); check(ii, x1u, y2u);
//		check(ii, x2d, y1d); check(ii, x2d, y1u); check(ii, x2u, y1d); check(ii, x2u, y1u);
//		check(ii, x2d, y2d); check(ii, x2d, y2u); check(ii, x2u, y2d); check(ii, x2u, y2u);
		double a = interpolate(x1d, x1u, y1d, y1u, getSum(x1d, y1d), getSum(x1d, y1u), getSum(x1u, y1d), getSum(x1u, y1u), x1, y1);
		double b = interpolate(x1d, x1u, y2d, y2u, getSum(x1d, y2d), getSum(x1d, y2u), getSum(x1u, y2d), getSum(x1u, y2u), x1, y2);
		double c = interpolate(x2d, x2u, y1d, y1u, getSum(x2d, y1d), getSum(x2d, y1u), getSum(x2u, y1d), getSum(x2u, y1u), x2, y1);
		double d = interpolate(x2d, x2u, y2d, y2u, getSum(x2d, y2d), getSum(x2d, y2u), getSum(x2u, y2d), getSum(x2u, y2u), x2, y2);
		return d - b - c + a;
	} else {
		throw std::runtime_error("GradientFeatureExtractor: unsupported image type " + lexical_cast<string>(integralImage.type()));
	}
}

double GradientFeatureExtractor::interpolate(float x1, float x2, float y1, float y2, double f11, double f12, double f21, double f22, float xa, float ya) const {
	double fa1 = interpolate(x1, x2, f11, f21, xa);
	double fa2 = interpolate(x1, x2, f12, f22, xa);
	return interpolate(y1, y2, fa1, fa2, ya);
}

double GradientFeatureExtractor::interpolate(float x1, float x2, double fx1, double fx2, float xa) const {
	if (x1 == x2)
		return fx1;
	double m = (fx1 - fx2) / (x1 - x2);
	double n = fx1 - m * x1;
//	double r = m * xa + n;
//	if (r != r) {
//		std::cout << "m " << m << " n " << n << " x1 " << x1 << " x2 " << x2 << " fx1 " << fx1 << " fx2 " << fx2 << std::endl;
//	}
	return m * xa + n;
}

} /* namespace imageprocessing */
