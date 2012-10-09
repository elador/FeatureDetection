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

	int initROI(FdImage*);

	std::string getIdentifier();
	void setIdentifier(std::string);

	//char outputPath[255];	// TODO: Delete this / use std::string
	int expected_num_faces[2];

	Rect roi;
	Rect roi_inImg;

private:
	bool canOutputProbabilistic;

protected:
	std::string identifier;
};
