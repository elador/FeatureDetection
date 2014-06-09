/*
 * Texture.cpp
 *
 *  Created on: 07.12.2012
 *      Author: Patrik Huber
 */

#include "render/Texture.hpp"
#include "render/MatrixUtils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

namespace render {

void Texture::createFromFile(const std::string& fileName, unsigned int mipmapsNum)
{
	cv::Mat image;
	try {
		image = cv::imread(fileName);
	} catch(cv::Exception& e) {
		std::cout << "OpenCV Exception caught while loading the image: " << e.what() << std::endl;
	}

	if (image.empty()) {
		std::cout << "Error: cv::imread: Loading image " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}

	this->mipmapsNum = (mipmapsNum == 0 ? render::utils::getMaxPossibleMipmapsNum(image.cols, image.rows) : mipmapsNum);
	/*if (mipmapsNum == 0)
	{
		uchar mmn = render::utils::MatrixUtils::getMaxPossibleMipmapsNum(image.cols, image.rows);
		this->mipmapsNum = mmn;
	} else
	{
		this->mipmapsNum = mipmapsNum;
	}*/

	if (this->mipmapsNum > 1)
	{
		if (!isPowerOfTwo(image.cols) || !isPowerOfTwo(image.rows))
		{
			std::cout << "Error: Couldn't generate mipmaps for image: " << fileName << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	image.convertTo(image, CV_8UC4);	// Most often, the input img is CV_8UC3. Img is BGR. Add an alpha channel
	cv::cvtColor(image, image, CV_BGR2BGRA);

	int currWidth = image.cols;
	int currHeight = image.rows;
	for (int i = 0; i < this->mipmapsNum; i++)
	{
		if( i==0 ) {
			mipmaps.push_back(image);
		} else {
			cv::Mat currMipMap(currHeight, currWidth, CV_8UC4);
			cv::resize(mipmaps[i-1], currMipMap, currMipMap.size());
			mipmaps.push_back(currMipMap);
		}

		if (currWidth > 1)
			currWidth >>= 1;
		if (currHeight > 1)
			currHeight >>= 1;
	}
	this->fileName = fileName;
	this->widthLog = (uchar)(std::log(mipmaps[0].cols)/CV_LOG2 + 0.0001f); // std::epsilon or something? or why 0.0001f here?
	this->heightLog = (uchar)(std::log(mipmaps[0].rows)/CV_LOG2 + 0.0001f); // Changed std::logf to std::log because it doesnt compile in linux (gcc 4.8). CHECK THAT
}

} /* namespace render */
