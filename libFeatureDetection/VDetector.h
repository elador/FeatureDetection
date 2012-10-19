#pragma once

#include <string>
#include "Rect.h"

//class Rect;
class FdImage;

class VDetector
{
public:
	VDetector(void);
	virtual ~VDetector(void);

	std::string getIdentifier();
	void setIdentifier(std::string);

	//char outputPath[255];	// TODO: Delete this / use std::string
	int expected_num_faces[2];

	Rect roiDistFromBorder;
	Rect roiInImg;

	int initROI(FdImage*);	// Convert roiDistFromBorder to roiInImg, given the FdImage.

private:
	bool canOutputProbabilistic;

protected:
	std::string identifier;
};
