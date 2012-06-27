#pragma once
#include "VDetector.h"
class CircleDetector : public VDetector
{
public:
	CircleDetector(void);
	~CircleDetector(void);

	cv::vector<cv::Vec3f> detectOnImage(FdImage*);
	cv::Mat getProbabilityMap(FdImage*);

	cv::vector<cv::Vec3f> circles;
};

