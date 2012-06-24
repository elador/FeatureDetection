#pragma once
#include "StdImage.h"
#include "utility.h"
#include "FdPatch.h"
#include <set>
#include <string>


class Pyramid : public StdImage
{
public:
	Pyramid(void);
	~Pyramid(void);
	
	FdPatchSet patches;					// All patches that belong to this pyramid (resolution level)
	std::set<std::string> detectorIds;	// All the detectors that use this pyramid add themselves here


	/*cv::Mat pyramid;		// For further extensions, eg pyramid-II (see lib MR)
	cv::Mat this_pyramid;

	cv::Mat iimg_x_pyramid;
	cv::Mat iimg_xx_pyramid;

	cv::Mat iimg_x, iimg_xx;*/
};

