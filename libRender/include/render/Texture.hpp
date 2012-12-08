/*!
 * \file Texture.h
 *
 * \author Patrik Huber
 * \date December 7, 2012
 *
 * [comment here]
 */
#pragma once

#include <opencv2/core/core.hpp>

namespace render {

class Texture
{
public:
	Texture(void);
	~Texture(void);

	void create(ushort width, ushort height, uchar mipmapsNum = 0);	// maybe this is not needed? purpose = ?
	void createFromFile(const std::string& fileName, uchar mipmapsNum = 0);

	//Mipmap<unsigned char> *mipmaps;
	std::vector<cv::Mat> mipmaps;	// make Texture a friend class of renderer, then move this to private?
	unsigned char widthLog, heightLog; // log2 of width and height of the base mip-level

private:
	std::string fileName;
	unsigned char mipmapsNum;

	inline bool isPowerOfTwo(int x)
	{
		return !(x & (x-1));
	};	// goto cpp/helpers
};

}
