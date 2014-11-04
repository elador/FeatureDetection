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

namespace FRsdk {
	class Image;
}

namespace facerecognition {

/**
 * Decodes a JPEG byte array into a FRsdk::Image.
 *
 * @param[in] bytearray In-memory representation of a JPEG image
 * @param[in] length length of the bytearray
 * @return A FRsdk::Image.
 * (Todo proper doxygen) Throws if not given a jpeg stream.
 */
FRsdk::Image decodeMemory(const char* bytearray, size_t length); // Note: We could put this in an anonymous namespace

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
cv::Mat frsdkImageToMat(FRsdk::Image image);

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
FRsdk::Image matToFRsdkImage(cv::Mat image);

} /* namespace facerecognition */

#endif /* FRSDK_HPP_ */
