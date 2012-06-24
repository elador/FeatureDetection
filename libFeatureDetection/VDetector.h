#pragma once
#include "Rect.h"
#include "FdImage.h"

class VDetector
{
public:
	VDetector(void);
	virtual ~VDetector(void);

	int initROI(FdImage*);

	std::string getIdentifier();
	void setIdentifier(std::string);

	char outputPath[_MAX_PATH];
	int expected_num_faces[2];

	Rect roi;
	Rect roi_inImg;

private:
	bool canOutputProbabilistic;

protected:
	std::string identifier;
};
