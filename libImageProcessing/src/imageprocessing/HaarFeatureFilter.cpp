/*
 * HaarFeatureFilter.cpp
 *
 *  Created on: 16.07.2013
 *      Author: poschmann
 */

#include "imageprocessing/HaarFeatureFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect_;
using std::vector;
using std::invalid_argument;

namespace imageprocessing {

HaarFeatureFilter::HaarFeatureFilter() : features() {
	vector<float> sizes;
	sizes.push_back(0.2f);
	sizes.push_back(0.4f);
	buildFeatures(sizes, 5, 5, TYPES_ALL);
}

HaarFeatureFilter::HaarFeatureFilter(vector<float> sizes, unsigned int count, int types) : features() {
	buildFeatures(sizes, count, count, types);
}

HaarFeatureFilter::HaarFeatureFilter(vector<float> sizes, unsigned int xCount, unsigned int yCount, int types) : features() {
	buildFeatures(sizes, xCount, yCount, types);
}

HaarFeatureFilter::HaarFeatureFilter(vector<float> sizes, vector<float> coords, int types) : features() {
	buildFeatures(sizes, coords, coords, types);
}

HaarFeatureFilter::HaarFeatureFilter(vector<float> sizes, vector<float> xs, vector<float> ys, int types) : features() {
	buildFeatures(sizes, xs, ys, types);
}

void HaarFeatureFilter::buildFeatures(vector<float> sizes, unsigned int xCount, unsigned int yCount, int types) {
	vector<float> xs(xCount);
	vector<float> ys(yCount);
	float xStep = 1.f / (xCount + 1);
	float yStep = 1.f / (yCount + 1);
	for (unsigned int i = 0; i < xCount; ++i)
		xs[i] = (i + 1) * xStep;
	for (unsigned int i = 0; i < yCount; ++i)
		ys[i] = (i + 1) * yStep;
	buildFeatures(sizes, xs, ys, types);
}

void HaarFeatureFilter::buildFeatures(vector<float> sizes, vector<float> xs, vector<float> ys, int types) {
	HaarFeature feature;
	for (float size : sizes) {
		for (float y : ys) {
			for (float x : xs) {
				Rect_<float> base(x - size / 2, y - size / 2, size, size);
				if (base.x < 0.f || base.x + base.width > 1.f
						|| base.y < 0.f || base.y + base.height > 1.f)
					continue;
				feature.area = base.area();

				if (types & TYPE_2RECTANGLE) {
					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, 0.5f * base.width, base.height));
					feature.rects.push_back(Rect_<float>(base.x + 0.5f * base.width, base.y, 0.5f * base.width, base.height));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(-1.f);
					feature.factor = 255 * 1.f / 2;
					features.push_back(feature);

					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, base.width, 0.5f * base.height));
					feature.rects.push_back(Rect_<float>(base.x, base.y + 0.5f * base.height, base.width, 0.5f * base.height));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(-1.f);
					feature.factor = 255 * 1.f / 2;
					features.push_back(feature);
				}

				if (types & TYPE_3RECTANGLE) {
					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, base.width / 3, base.height));
					feature.rects.push_back(Rect_<float>(base.x + base.width / 3, base.y, base.width / 3, base.height));
					feature.rects.push_back(Rect_<float>(base.x + 2 * base.width / 3, base.y, base.width / 3, base.height));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(-2.f);
					feature.weights.push_back(1.f);
					feature.factor = 255 * 2.f / 3;
					features.push_back(feature);

					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, base.width, base.height / 3));
					feature.rects.push_back(Rect_<float>(base.x, base.y + base.height / 3, base.width, base.height / 3));
					feature.rects.push_back(Rect_<float>(base.x, base.y + 2 * base.height / 3, base.width, base.height / 3));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(-2.f);
					feature.weights.push_back(1.f);
					feature.factor = 255 * 2.f / 3;
					features.push_back(feature);
				}

				if (types & TYPE_4RECTANGLE) {
					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, base.width / 2, base.height / 2));
					feature.rects.push_back(Rect_<float>(base.x + base.width / 2, base.y + base.height / 2, base.width / 2, base.height / 2));
					feature.rects.push_back(Rect_<float>(base.x + base.width / 2, base.y, base.width / 2, base.height / 2));
					feature.rects.push_back(Rect_<float>(base.x, base.y + base.height / 2, base.width / 2, base.height / 2));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(1.f);
					feature.weights.push_back(-1.f);
					feature.weights.push_back(-1.f);
					feature.factor = 255 * 1.f / 2;
					features.push_back(feature);
				}

				if (types & TYPE_CENTER_SURROUND) {
					feature.rects.clear();
					feature.rects.push_back(Rect_<float>(base.x, base.y, base.width, base.height));
					feature.rects.push_back(Rect_<float>(base.x + base.width / 4, base.y + base.height / 4, base.width / 2, base.height / 2));
					feature.weights.clear();
					feature.weights.push_back(1.f);
					feature.weights.push_back(-4.f);
					feature.factor = 255 * 3.f / 4;
					features.push_back(feature);
				}
			}
		}
	}
}

Mat HaarFeatureFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_32SC1)
		throw invalid_argument("HaarFeatureFilter: the image must be of type CV_32SC1");
	filtered.create(1, features.size(), CV_32F);
	float* data = filtered.ptr<float>(0);
	for (unsigned int i = 0; i < features.size(); ++i) {
		float value = 0;
		const HaarFeature& feature = features[i];
		for (unsigned int j = 0; j < feature.rects.size(); ++j) {
			const Rect_<float>& rect = feature.rects[j];
			int x1 = cvRound(rect.x * image.cols);
			int x2 = cvRound((rect.x + rect.width) * image.cols);
			int y1 = cvRound(rect.y * image.rows);
			int y2 = cvRound((rect.y + rect.height) * image.rows);
			int areaSum = image.at<int>(y1, x1) + image.at<int>(y2, x2) - image.at<int>(y1, x2) - image.at<int>(y2, x1);
			value += feature.weights[j] * areaSum;
		}
		data[i] = value / (feature.factor * feature.area * image.cols * image.rows);
	}
	return filtered;
}

} /* namespace imageprocessing */
