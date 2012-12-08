/*!
 * \file Texture.cpp
 *
 * \author Patrik Huber
 * \date December 7, 2012
 *
 * [comment here]
 */

#include "render/Texture.hpp"
#include "render/MatrixUtils.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

namespace render {

Texture::Texture(void)
{
}

Texture::~Texture(void)
{
}

void Texture::create(ushort width, ushort height, uchar mipmapsNum)
{
	this->mipmapsNum = (mipmapsNum == 0 ? render::utils::MatrixUtils::getMaxPossibleMipmapsNum(width, height) : mipmapsNum);
	//this->mipmaps = new Mipmap<uchar>[this->mipmapsNum];

	for (int i = 0; i < this->mipmapsNum; i++)
	{
		/*mipmaps[i].width = width;
		mipmaps[i].height = height;
		mipmaps[i].data = new uchar[4*width*height]; */	// 4, because of the 4 color channels
		cv::Mat currMipMap(height, width, CV_8UC4);
		mipmaps.push_back(currMipMap);

		if (width > 1)
			width >>= 1;
		if (height > 1)
			height >>= 1;
	}

	this->widthLog = (uchar)(std::logf(mipmaps[0].cols)/std::logf(2.0f) + 0.0001f); // _log2(mm.w) // epsilon
	this->heightLog = (uchar)(std::logf(mipmaps[0].rows)/std::logf(2.0f) + 0.0001f); // log2(mm.h) // epsilon. Use CV_LOG2?
}

void Texture::createFromFile(const std::string& fileName, uchar mipmapsNum)
{
	//CImageByte image;	// convert to byte
	cv::Mat image;
	try {
		image = cv::imread(fileName);
	} catch( cv::Exception& e ) {
		const char* err_msg = e.what();
		std::cout << "OpenCV Exception caught while loading the image: " << err_msg << std::endl;
	}

	if (image.empty()) {
		std::cout << "Error: cv::imread: Loading image " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}

	this->mipmapsNum = (mipmapsNum == 0 ? render::utils::MatrixUtils::getMaxPossibleMipmapsNum(image.cols, image.rows) : mipmapsNum);

	if (this->mipmapsNum > 1)
	{
		if (!isPowerOfTwo(image.cols) || !isPowerOfTwo(image.rows))
		{
			std::cout << "Error: Couldn't generate mipmaps for image: " << fileName << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	image.convertTo(image, CV_8UC4);	// ... check this
	cv::cvtColor(image, image, CV_BGR2BGRA);

	int currWidth = image.cols;
	int currHeight = image.rows;
	for (int i = 0; i < this->mipmapsNum; i++)
	{
		if( i==0 ) {
			mipmaps.push_back(image);
		} else {
			cv::Mat currMipMap(currHeight, currWidth, CV_8UC4);
			// image.swapChannels(0, 2);	// we already have BGR. Ok?
			// scale the texture here & pushback
			cv::resize(mipmaps[i-1], currMipMap, currMipMap.size());
			mipmaps.push_back(currMipMap);
		}

		if (currWidth > 1)
			currWidth >>= 1;
		if (currHeight > 1)
			currHeight >>= 1;
	}
	this->fileName = fileName;
}

}
