/*
 * DirectImageExtractor.cpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#include "DirectImageExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Rect;
using cv::Size;
using cv::resize;

namespace imageprocessing {

DirectImageExtractor::DirectImageExtractor() : version(-1), image(), patchWidth(20), patchHeight(20), filter() {}

DirectImageExtractor::~DirectImageExtractor() {}

shared_ptr<Patch> DirectImageExtractor::extract(int x, int y, int width, int height) const {
	int px = x - width / 2;
	int py = y - height / 2;
	if (px < 0 || py < 0 || px + width >= image.cols || py + height >= image.rows)
		return shared_ptr<Patch>();
	Mat patchData(image, Rect(px, py, width, height));
	Mat resizedPatchData;
	resize(patchData, resizedPatchData, Size(patchWidth, patchHeight), 0, 0, CV_INTER_AREA);
	return make_shared<Patch>(x, y, width, height, resizedPatchData);
}

} /* namespace imageprocessing */
