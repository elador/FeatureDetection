#include "stdafx.h"
#include "StdImage.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


StdImage::StdImage(void)
{
	w = 0;
	h = 0;
	colordepth = 0;
	data = NULL;
}


StdImage::~StdImage(void)
{
	if(data!=NULL) {
		delete[] data;
	}
	data = NULL;
}

StdImage::StdImage(unsigned char* data, int w, int h, int colordepth)
{
	this->w = w;
	this->h = h;
	this->colordepth = colordepth;
	this->data = data;

}

void StdImage::writePNG(void)
{
	cv::Mat test(this->h, this->w, CV_8UC1, this->data);
	char* fn = new char[255];
	sprintf(fn, "img_out/img_%d_%d.png", this->h, this->w);
	cv::imwrite(fn, test);
	delete[] fn;
	// TODO: If needed sometime: This should also work when colordepth is not 8 (=grayscale).
}

void StdImage::writePNG(const std::string filename) const
{
	cv::Mat test(this->h, this->w, CV_8UC1, this->data);
	cv::imwrite(filename, test);
}