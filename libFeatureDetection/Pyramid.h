#pragma once
#include "StdImage.h"
#include "utility.h"
#include "FdPatch.h"


class Pyramid : public StdImage
{
public:
	Pyramid(void);
	~Pyramid(void);
	
	FdPatchSet patches;

	/*cv::Mat pyramid;
	cv::Mat this_pyramid;

	cv::Mat iimg_x_pyramid;
	cv::Mat iimg_xx_pyramid;

	cv::Mat iimg_x, iimg_xx;*/
};

