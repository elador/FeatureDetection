#pragma once
#include "VDetector.h"

#include "opencv2/core/core.hpp"

class CircleDetector : public VDetector
{
public:
	CircleDetector(void);
	~CircleDetector(void);

	cv::vector<cv::Vec3f> detectOnImage(FdImage*);
	cv::Mat getProbabilityMap(FdImage*);

	cv::vector<cv::Vec3f> circles;
};

