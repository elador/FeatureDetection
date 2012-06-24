#pragma once
#include "vdetector.h"
class SkinDetector : public VDetector
{
public:
	SkinDetector(void);
	~SkinDetector(void);

	cv::Mat detectOnImage(FdImage*);
};

