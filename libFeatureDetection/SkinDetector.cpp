#include "stdafx.h"
#include "SkinDetector.h"

#include "FdImage.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

SkinDetector::SkinDetector(void)
{
	this->identifier = "SkinDetector";
}


SkinDetector::~SkinDetector(void)
{
}


cv::Mat SkinDetector::detectOnImage(FdImage* img)
{
	cv::Mat hsvImg;
	cv::cvtColor(img->data_matbgr, hsvImg, CV_BGR2HSV);
	cv::Mat binarySkinMap(hsvImg.rows, hsvImg.cols, CV_32F);
	//std::cout << img->data_matbgr << std::endl;
	//std::cout << hsvImg << std::endl;
	//Hue component as an 8-bit integer in the range 0 to 179
	// Hue 6-38 = skin 0-179?
	// Hue 9-54 0-255?
	// Hue12-76 0-360?	PS: Um 42 gut

	// PS:	0-360, 100, 100
	// Bush: H5-25, S25-60, B50-90
	
	for (int i=0; i<img->data_matbgr.rows; i++)
	{
		for (int j=0; j<img->data_matbgr.cols; j++)
		{
			//cv::Vec3b test = hsvImg.at<cv::Vec3b>(i, j);
			//std::cout << "H\t" << (int)hsvImg.at<cv::Vec3b>(i, j)[0] << "S\t" << (int)hsvImg.at<cv::Vec3b>(i, j)[1] << "V\t" << (int)hsvImg.at<cv::Vec3b>(i, j)[2] << std::endl;
			//if((int)hsvImg.at<cv::Vec3b>(i, j)[0]>=3 && (int)hsvImg.at<cv::Vec3b>(i, j)[0]<=18 && (int)hsvImg.at<cv::Vec3b>(i, j)[1]>=25 && (int)hsvImg.at<cv::Vec3b>(i, j)[1]<=60 && (int)hsvImg.at<cv::Vec3b>(i, j)[2]>=50 && (int)hsvImg.at<cv::Vec3b>(i, j)[2]<=90) { // (y, x) !!! i=row, j=column (matrix)
			if((int)hsvImg.at<cv::Vec3b>(i, j)[0]>=4 && (int)hsvImg.at<cv::Vec3b>(i, j)[0]<=35) { // (y, x) !!! i=row, j=column (matrix)
				binarySkinMap.at<float>(i, j) = 1.0f;
			} else {
				binarySkinMap.at<float>(i, j) = 0.0f;
			}
		}
	}
	
	//LOST Logger->LogImgSkinDetector(&binarySkinMap, img->filename, this->getIdentifier());

	return binarySkinMap;
}
