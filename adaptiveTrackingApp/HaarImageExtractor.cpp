/*
 * DirectImageExtractor.cpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#include "HaarImageExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Rect;
using cv::Size;
using cv::resize;

namespace imageprocessing {

HaarImageExtractor::HaarImageExtractor() : version(-1), image(), filter(), integralFilter(), features() {
//	float s[] = { 0.2f, 0.4f };
//	float x[] = { 0.2f, 0.4f, 0.6f, 0.8f };
//	float y[] = { 0.2f, 0.4f, 0.6f, 0.8f };
	float s[] = { 0.2f, 0.4f };
	float x[] = { 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
	float y[] = { 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
//	float s[] = { 0.2f, 0.4f };
//	float x[] = { 0.2f, 0.35f, 0.5f, 0.65f, 0.8f };
//	float y[] = { 0.2f, 0.35f, 0.5f, 0.65f, 0.8f };
	HaarFeature feature;
//	for (int is = 0; is < 2; ++is) {
//		for (int iy = 0; iy < 4; ++iy) {
//			for (int ix = 0; ix < 4; ++ix) {
	for (int is = 0; is < 2; ++is) {
		for (int iy = 0; iy < 7; ++iy) {
			for (int ix = 0; ix < 7; ++ix) {
//	for (int is = 0; is < 2; ++is) {
//		for (int iy = 0; iy < 5; ++iy) {
//			for (int ix = 0; ix < 5; ++ix) {
				Rect_<float> base(x[ix] - s[is] / 2, y[iy] - s[is] / 2, s[is], s[is]);
				feature.area = base.area();

				feature.rects.clear();
				feature.rects.push_back(Rect_<float>(base.x, base.y, 0.5f * base.width, base.height));
				feature.rects.push_back(Rect_<float>(base.x + 0.5f * base.width, base.y, 0.5f * base.width, base.height));
				feature.weights.clear();
				feature.weights.push_back(1.f);
				feature.weights.push_back(-1.f);
				feature.factor = 0.5f * 255;
				features.push_back(feature);

				feature.rects.clear();
				feature.rects.push_back(Rect_<float>(base.x, base.y, base.width, 0.5f * base.height));
				feature.rects.push_back(Rect_<float>(base.x, base.y + 0.5f * base.height, base.width, 0.5f * base.height));
				feature.weights.clear();
				feature.weights.push_back(1.f);
				feature.weights.push_back(-1.f);
				feature.factor = 0.5f * 255;
				features.push_back(feature);

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

//				feature.rects.clear();
//				feature.rects.push_back(Rect_<float>(base.x, base.y, base.width / 2, base.height / 2));
//				feature.rects.push_back(Rect_<float>(base.x + base.width / 2, base.y + base.height / 2, base.width / 2, base.height / 2));
//				feature.rects.push_back(Rect_<float>(base.x + base.width / 2, base.y, base.width / 2, base.height / 2));
//				feature.rects.push_back(Rect_<float>(base.x, base.y + base.height / 2, base.width / 2, base.height / 2));
//				feature.weights.clear();
//				feature.weights.push_back(1.f);
//				feature.weights.push_back(1.f);
//				feature.weights.push_back(-1.f);
//				feature.weights.push_back(-1.f);
//				feature.factor = 255 * 1.f / 2;
//				features.push_back(feature);
//
//				feature.rects.clear();
//				feature.rects.push_back(Rect_<float>(base.x, base.y, base.width, base.height));
//				feature.rects.push_back(Rect_<float>(base.x + base.width / 4, base.y + base.height / 4, base.width / 2, base.height / 2));
//				feature.weights.clear();
//				feature.weights.push_back(1.f);
//				feature.weights.push_back(-4.f);
//				feature.factor = 255 * 3.f / 4;
//				features.push_back(feature);
			}
		}
	}
}

HaarImageExtractor::~HaarImageExtractor() {}

shared_ptr<Patch> HaarImageExtractor::extract(int x, int y, int width, int height) const {
	int px = x - width / 2;
	int py = y - height / 2;
	if (px < 0 || py < 0 || px + width >= image.cols || py + height >= image.rows)
		return shared_ptr<Patch>();

	Mat patchData(1, features.size(), CV_32F);
	float* data = patchData.ptr<float>(0);
	for (int i = 0; i < features.size(); ++i) {
		float value = 0;
		const HaarFeature& feature = features[i];
		for (int j = 0; j < feature.rects.size(); ++j) {
			const Rect_<float>& rect = feature.rects[j];
			int x1 = px + cvRound(rect.x * width);
			int x2 = px + cvRound((rect.x + rect.width) * width);
			int y1 = py + cvRound(rect.y * height);
			int y2 = py + cvRound((rect.y + rect.height) * height);
			int areaSum = image.at<int>(y1, x1) + image.at<int>(y2, x2) - image.at<int>(y1, x2) - image.at<int>(y2, x1);
			value += feature.weights[j] * areaSum;
		}
		data[i] = value / (feature.factor * feature.area * width * height);
	}
	return make_shared<Patch>(x, y, width, height, patchData);
}

} /* namespace imageprocessing */
