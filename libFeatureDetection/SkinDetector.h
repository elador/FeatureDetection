#pragma once
#include "VDetector.h"

#include "opencv2/core/core.hpp"

class SkinDetector : public VDetector
{
public:
	SkinDetector(void);
	~SkinDetector(void);

	cv::Mat detectOnImage(FdImage*);
};

