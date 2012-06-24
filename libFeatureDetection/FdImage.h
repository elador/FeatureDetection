#pragma once
#include "StdImage.h"
#include "Pyramid.h"

typedef std::map<int, Pyramid*> PyramidMap;

class FdImage : public StdImage
{
public:
	FdImage(void);
	~FdImage(void);
	FdImage(unsigned char*, int, int, int=8);
	int load(const std::string);
	int load(const cv::Mat*);

	int createPyramid(int, int*, std::string);

	cv::Mat data_matbgr;
	cv::Mat data_matgray;

	PyramidMap pyramids;	// a map of all the pyramids this image has

protected:
	int rescale(StdImage&, StdImage&, float);

};

