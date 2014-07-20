/*
 * VlHogFilter.cpp
 *
 *  Created on: 03.07.2014
 *      Author: Patrik Huber
 */

#include "superviseddescent/VlHogFilter.hpp"

using cv::Mat;

namespace superviseddescent {

cv::Mat VlHogFilter::applyTo(const cv::Mat& image, cv::Mat& filtered) const
{
	/*
	cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) {
		Mat grayImage;
		if (image.channels() == 3) {
			cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
		}
		else {
			grayImage = image;
		}
		VlHogVariant vlHogVariant;
		switch (hogType)
		{
		case VlHogDescriptorExtractor::VlHogType::DalalTriggs:
			vlHogVariant = VlHogVariant::VlHogVariantDalalTriggs;
			break;
		case VlHogDescriptorExtractor::VlHogType::Uoctti:
			vlHogVariant = VlHogVariant::VlHogVariantUoctti;
			break;
		default:
			break;
		}

		int patchWidthHalf = numCells * (cellSize / 2); // patchWidthHalf: Zhenhua's 'numNeighbours'. cellSize: has nothing to do with HOG. It's rather the number of HOG cells we want.

		//int hogDim1 = (numNeighbours * 2) / hogCellSize; // i.e. how many times does the hogCellSize fit into our patch
		//int hogDim2 = hogDim1; // as our patch is quadratic, those two are the same
		//int hogDim3 = 16; // VlHogVariantUoctti: Creates 4+3*numOrientations dimensions. DT: 4*numOri dimensions.
		//int hogDims = hogDim1 * hogDim2 * hogDim3;
		//Mat hogDescriptors(locations.size(), hogDims, CV_32FC1); // better allocate later, when we know ww/hh/dd?
		Mat hogDescriptors; // We'll get the dimensions later from vl_hog_get_*

		for (int i = 0; i < locations.size(); ++i) {
			// get the (x, y) location and w/h of the current patch
			int x = cvRound(locations[i].x);
			int y = cvRound(locations[i].y);

			Mat roiImg;
			if (x - patchWidthHalf < 0 || y - patchWidthHalf < 0 || x + patchWidthHalf >= image.cols || y + patchWidthHalf >= image.rows) {
				// The feature extraction location is too far near a border. We extend the image (add a black canvas)
				// and then extract from this larger image.
				int borderLeft = (x - patchWidthHalf) < 0 ? std::abs(x - patchWidthHalf) : 0; // Our x and y are center.
				int borderTop = (y - patchWidthHalf) < 0 ? std::abs(y - patchWidthHalf) : 0;
				int borderRight = (x + patchWidthHalf) >= image.cols ? std::abs(image.cols - (x + patchWidthHalf)) : 0;
				int borderBottom = (y + patchWidthHalf) >= image.rows ? std::abs(image.rows - (y + patchWidthHalf)) : 0;
				Mat extendedImage = image.clone();
				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
				cv::Rect roi((x - patchWidthHalf) + borderLeft, (y - patchWidthHalf) + borderRight, patchWidthHalf * 2, patchWidthHalf * 2); // Rect: x y w h. x and y are top-left corner.
				roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
			}
			else {
				cv::Rect roi(x - patchWidthHalf, y - patchWidthHalf, patchWidthHalf * 2, patchWidthHalf * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				// we have exactly the same window as the matlab code.
				// extract the patch and supply it to vl_hog
				roiImg = image(roi).clone(); // clone because we need a continuous memory block
			}
			roiImg.convertTo(roiImg, CV_32FC1); // because vl_hog_put_image expects a float* (values 0.f-255.f)
			// vl_hog_new: numOrientations=hogParameter.numBins, transposed (=col-major):false)
			VlHog* hog = vl_hog_new(vlHogVariant, numBins, false); // VlHogVariantUoctti seems to be default in Matlab.
			vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, 1, cellSize); // (the '1' is numChannels)
			vl_size ww = vl_hog_get_width(hog); // we could assert that ww == hh == numCells
			vl_size hh = vl_hog_get_height(hog);
			vl_size dd = vl_hog_get_dimension(hog); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
			//float* hogArray = (float*)malloc(ww*hh*dd*sizeof(float));
			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
			//vl_hog_extract(hog, hogArray); // just interpret hogArray in col-major order to get the same n x 1 vector as in matlab. (w * h * d)
			vl_hog_extract(hog, hogArray.ptr<float>(0));
			vl_hog_delete(hog);
			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
			// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector
			for (int j = 0; j < dd; ++j) {
				//Mat hogFeatures(hh, ww, CV_32FC1, hogArray + j*ww*hh);
				Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
				hogFeatures = hogFeatures.t(); // Necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
				hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
				Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
				hogFeatures.copyTo(currentDimSubMat);
			}
			//free(hogArray); // not necessary - we use a Mat.
			//features = [features; double(reshape(tmp, [], 1))];
			// B = reshape(A,m,n) returns the m-by-n matrix B whose elements are taken column-wise from A
			// Matlab (& Eigen, OpenGL): Column-major.
			// OpenCV: Row-major.
			// (access is always (r, c).)
			//Mat currentFeaturesSubrange = hogDescriptors.rowRange(i*hogDims, i*hogDims + hogDims);
			//hogDescriptor.copyTo(currentFeaturesSubrange);
			hogDescriptor = hogDescriptor.t(); // now a row-vector
			hogDescriptors.push_back(hogDescriptor);
		}
		// hogDescriptors needs to have dimensions numLandmarks x hogFeaturesDimension, where hogFeaturesDimension is e.g. 3*3*16=144
		return hogDescriptors;
	};
	*/
	filtered = Mat();
	return filtered;
}

} /* namespace superviseddescent */
