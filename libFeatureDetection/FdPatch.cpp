#include "stdafx.h"
#include "FdPatch.h"

#include "IImg.h"

FdPatch::FdPatch(void) : w_inFullImg(0), h_inFullImg(0), iimg_x(NULL), iimg_xx(NULL), sampleID(0)
{
}


FdPatch::~FdPatch(void)
{
	if(iimg_x != NULL)
		delete iimg_x;
	if(iimg_xx != NULL)
		delete iimg_xx;
}

FdPatch::FdPatch(int c_x_py, int c_y_py, int w, int h) : w_inFullImg(0), h_inFullImg(0), iimg_x(NULL), iimg_xx(NULL)
{
	this->c.x_py = c_x_py;
	this->c.y_py = c_y_py;
	this->w = w;
	this->h = h;
	this->data = NULL;
}

