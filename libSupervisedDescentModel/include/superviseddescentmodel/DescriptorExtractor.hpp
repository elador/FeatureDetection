/*
 * DescriptorExtractor.hpp
 *
 *  Created on: 21.03.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef DESCRIPTOREXTRACTOR_HPP_
#define DESCRIPTOREXTRACTOR_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

extern "C" {
	#include "superviseddescentmodel/hog.h"
}

using cv::Mat;
using std::vector;


namespace superviseddescentmodel {

/**
 * Base-class for extracting descriptors at given points
 * in an image.
 * Todo: Check how this integrates with our FeatureExtractors
 * in libImageProcessing.
 */
class DescriptorExtractor
{
public:
	// returns a Matrix, as many rows as points, 1 descriptor = 1 row
	virtual cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) = 0;
};

class SiftDescriptorExtractor : public DescriptorExtractor
{
public:
	// c'tor with param diameter & orientation? (0.0f = right?)
	// and store them as private vars.
	// However, it might be better to store the parameters separately, to be able to share a FeatureDescriptorExtractor over multiple Sdm cascade levels

	cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) {
		Mat grayImage;
		if (image.channels() == 3) {
			cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
		} else {
			grayImage = image;
		}
		cv::SIFT sift; // init only once, could reuse
		vector<cv::KeyPoint> keypoints;
		for (const auto& loc : locations) {
			keypoints.emplace_back(cv::KeyPoint(loc, 32.0f, 0.0f)); // Angle is set to 0. If it's -1, SIFT will be calculated for 361degrees. But Paper (email) says upwards.
		}
		Mat siftDescriptors;
		sift(grayImage, Mat(), keypoints, siftDescriptors, true);
		//cv::drawKeypoints(img, keypoints, img, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		return siftDescriptors;
	};
};

class VlHogDescriptorExtractor : public DescriptorExtractor
{
public:
	// Store the params as private vars? However, it might be better to store the parameters separately, to be able to share a FeatureDescriptorExtractor over multiple Sdm cascade levels

	enum class VlHogType {
		DalalTriggs,
		Uoctti
	};

	VlHogDescriptorExtractor(VlHogType vlhogType, int cellSize, int numBins) : hogType(vlhogType), cellSize(cellSize), numBins(numBins) {

	};

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
		
		// TODO: HOG code not yet updated for the new model, col/rows etc!
		
		int numNeighbours = cellSize * 6; // this cellSize has nothing to do with HOG. It's the number of "cells", i.e. image-windows/patches.
		// if cellSize=1, our window is 12x12, and because our HOG-cellsize is 12, it means we will have 1 cell (the minimum).
		int hogCellSize = 12;
		int hogDim1 = (numNeighbours * 2) / hogCellSize; // i.e. how many times does the hogCellSize fit into our patch
		int hogDim2 = hogDim1; // as our patch is quadratic, those two are the same
		int hogDim3 = 16; // VlHogVariantUoctti: Creates 4+3*numOrientations dimensions
		int hogDims = hogDim1 * hogDim2 * hogDim3;
		Mat currentFeatures(locations.size() * hogDims, 1, CV_32FC1);

		for (int i = 0; i < locations.size(); ++i) {
			// get the (x, y) location and w/h of the current patch
			int x = cvRound(locations[i].x);
			int y = cvRound(locations[i].y);
			cv::Rect roi(x - numNeighbours, y - numNeighbours, numNeighbours * 2, numNeighbours * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
			// we have exactly the same window as the matlab code.
			// extract the patch and supply it to vl_hog
			Mat roiImg = image(roi).clone(); // clone because we need a continuous memory block
			roiImg.convertTo(roiImg, CV_32FC1); // because vl_hog_put_image expects a float* (values 0.f-255.f)
			// vl_hog_new: numOrientations=hogParameter.numBins, transposed (=col-major):false)
			VlHog* hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, numBins, false); // VlHogVariantUoctti seems to be default in Matlab.
			vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, 1, hogCellSize); // (the '1' is numChannels)
			vl_size ww = vl_hog_get_width(hog);
			vl_size hh = vl_hog_get_height(hog);
			vl_size dd = vl_hog_get_dimension(hog); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
			//float* hogArray = (float*)malloc(ww*hh*dd*sizeof(float));
			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
			//vl_hog_extract(hog, hogArray); // just interpret hogArray in col-major order to get the same n x 1 vector as in matlab. (w * h * d)
			vl_hog_extract(hog, hogArray.ptr<float>(0));
			vl_hog_delete(hog);
			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
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
			Mat currentFeaturesSubrange = currentFeatures.rowRange(i*hogDims, i*hogDims + hogDims);
			hogDescriptor.copyTo(currentFeaturesSubrange);
			// currentFeatures needs to have dimensions n x 1, where n = numLandmarks * hogFeaturesDimension, e.g. n = 22 * (3*3*16=144) = 3168 (for the first hog Scale)
		}



		Mat hogDescriptors;
		return hogDescriptors;
	};

private:
	VlHogType hogType;
	int cellSize;
	int numBins;
};


} /* namespace superviseddescentmodel */
#endif /* DESCRIPTOREXTRACTOR_HPP_ */
