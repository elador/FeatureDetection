/*
 * frsdk.hpp
 *
 * Todo: Rename to frsdkHelpers or frsdkUtils
 *
 *  Created on: 11.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FRSDK_HPP_
#define FRSDK_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "frsdk/image.h"
#include "frsdk/jpeg.h"

#include <vector>
#include <string>
//#include <sstream>

using cv::Mat;
using std::string;
using std::vector; // Remove after finalising these headers

namespace facerecognition {

/**
 * Decodes a JPEG byte array into a FRsdk::Image.
 *
 * @param[in] bytearray In-memory representation of a JPEG image
 * @param[in] length length of the bytearray
 * @return A FRsdk::Image.
 * (Todo proper doxygen) Throws if not given a jpeg stream.
 */
FRsdk::Image decodeMemory(const char* bytearray, size_t length) // Note: We could put this in an anonymous namespace
{
	// if a std:string (used for memory management) is created from a byte array
	// and with the additional input of the length of the array the complete
	// array is used as a string and will not end at the first '\0'
	string strfromarray(bytearray, length);

	// create a std::stream from std::string
	std::stringstream memstream(strfromarray);

	// use FRsdk::Jpeg::load( std::istream&, ...) in order to read from
	// the stringstream
	FRsdk::Image img = FRsdk::Jpeg::load(memstream);
	
	return img;
}

/**
 * Converts a FRsdk::Image into an OpenCV cv::Mat.
 *
 * It does not copy the data of the underlying FRsdk::Image,
 * i.e. it will die when the FRsdk::Image goes out of scope.
 * Use the clone() member from cv::Mat if you want a copy of the memory.
 *
 * @param[in] image A FRsdk::Image
 * @return Image in cv::Mat format.
 */
cv::Mat frsdkImageToMat(FRsdk::Image image)
{
	Mat outImage;
	// FRsdk::Image uses uchar
	if (image.isColor()) {
		// FRsdk::Image data is BGRA
		// Danger: We're casting away the const-ness of the FRsdk::Image data here. It might be preferable to copy the data first, but we leave this to the user.
		outImage = Mat(image.height(), image.width(), CV_8UC4, reinterpret_cast<void*>(const_cast<FRsdk::Rgb*>(image.colorRepresentation())));
	}
	else {
		outImage = Mat(image.height(), image.width(), CV_8UC1, reinterpret_cast<void*>(const_cast<FRsdk::Byte*>(image.grayScaleRepresentation())));
	}
	return outImage;
}

/**
 * Converts an OpenCV cv::Mat into an FRsdk::Image.
 *
 * The image will be converted into JPEG first, therefore,
 * small differences in pixel values will arise. 
 * FRsdk::Image will own its own memory.
 *
 * @param[in] image An OpenCV image
 * @return Image in FRsdk::Image format.
 */
FRsdk::Image matToFRsdkImage(cv::Mat image)
{
	vector<uchar> jpegBuffer;
	vector<int> jpegParams = { cv::IMWRITE_JPEG_QUALITY, 100 }; // default is 95
	cv::imencode(".jpg", image, jpegBuffer, jpegParams);

	FRsdk::Image frsdkImage = decodeMemory(reinterpret_cast<char*>(&jpegBuffer[0]), jpegBuffer.size());

	return frsdkImage;
}

} /* namespace facerecognition */

#endif /* FRSDK_HPP_ */
