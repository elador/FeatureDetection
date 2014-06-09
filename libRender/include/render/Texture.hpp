/*
 * Texture.hpp
 *
 *  Created on: 07.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include "opencv2/core/core.hpp"

namespace render {

/**
 * Desc
 */
class Texture
{
public:

	void createFromFile(const std::string& fileName, unsigned int mipmapsNum = 0);

	std::vector<cv::Mat> mipmaps;	// make Texture a friend class of renderer, then move this to private?
	unsigned char widthLog, heightLog; // log2 of width and height of the base mip-level

private:
	std::string fileName;
	unsigned int mipmapsNum;

	inline bool isPowerOfTwo(int x)
	{
		return !(x & (x-1));
	};	// goto cpp/helpers
};

} /* namespace render */

#endif /* TEXTURE_HPP_ */
