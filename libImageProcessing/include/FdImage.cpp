#include "stdafx.h"
#include "FdImage.h"

#include "SLogger.h"
#include "Pyramid.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

FdImage::FdImage(void)
{

}


FdImage::~FdImage(void)
{
	PyramidMap::iterator it = pyramids.begin();
	for(; it != pyramids.end(); it++) {
		delete it->second;
		it->second = NULL;
	}

}


FdImage::FdImage(unsigned char* data, int w, int h, int colordepth) : StdImage(data, w, h, colordepth)
{

	// maybe it would be wise to initialize mat_bgr and mat_gray?
}

int FdImage::load(const std::string filename)
{
	try {
		this->data_matbgr = cv::imread(filename);
	} catch( cv::Exception& e ) {
		const char* err_msg = e.what();
		std::cout << "OpenCV Exception caught while loading the image: " << err_msg << std::endl;
	}

	if (this->data_matbgr.empty()) {
		std::cout << "Error: cv::imread: Loading image " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	this->filename = filename;

	Logger->LogImgInputRGB(&this->data_matbgr, this->filename);
	cv::cvtColor(this->data_matbgr, this->data_matgray, CV_BGR2GRAY);
	// Mat.convertTo(dst, CV_8U); not working (stays bgr)... maybe doesn't copy?
	Logger->LogImgInputGray(&this->data_matgray, this->filename);

	this->w = this->data_matgray.cols;
	this->h = this->data_matgray.rows;

	this->data = new unsigned char[w*h];
	if(Logger->getVerboseLevelText()>=2) {		// Todo: Check if this is consistent with the other log-levels (see SLogger.h)
		std::cout << "[FdImage] Allocating space for image. Cols(=width): " << w << ", Rows(=height): " << h << ", Size: " << w*h << std::endl;
	}

	for (int i=0; i<this->data_matgray.rows; i++)
	{
		for (int j=0; j<this->data_matgray.cols; j++)
		{
			this->data[i*w+j] = this->data_matgray.at<uchar>(i, j); // (y, x) !!! i=row, j=column (matrix)
		}
	}
	//double* row = mat.ptr<double>(i);
	//to get pointer to i'th row and row[j] to get j'th value.

	/*cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
	//cv::imshow("image", this->data_matgray);
	//cv::waitKey();

	cv::Mat test = cv::imread(filename, 0);
	cv::imshow("image", test);

	bool loop = true;
	while(loop) {
		loop = !(bool)cv::waitKey();
	}*/

	return 1;
}

int FdImage::load(const cv::Mat* mat)
{
	
	if (mat->empty()) {
		std::cout << "Error: cv::imread: Loading image " << filename << std::endl;
		return 0;
	}
	cv::cvtColor(*mat, this->data_matgray, CV_BGR2GRAY);
	// Mat.convertTo(dst, CV_8U); not working (stays bgr)... maybe doesn't copy?

	this->w = this->data_matgray.cols;
	this->h = this->data_matgray.rows;

	this->data = new unsigned char[w*h];
	if(Logger->getVerboseLevelText()>=2) {
		std::cout << "[FdImage] Allocating space for image. Cols(=width): " << w << ", Rows(=height): " << h << ", Size: " << w*h << std::endl;
	}

	for (int i=0; i<this->data_matgray.rows; i++)
	{
		for (int j=0; j<this->data_matgray.cols; j++)
		{
			this->data[i*w+j] = this->data_matgray.at<uchar>(i, j); // (y, x) !!! i=row, j=column (matrix)
		}
	}
	//double* row = mat.ptr<double>(i);
	//to get pointer to i'th row and row[j] to get j'th value.

	//cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
	//cv::imshow("image", this->data_matgray);
	//cv::waitKey();
	
	//cv::Mat test = cv::imread(filename, 0);
	//cv::imshow("image", test);

	/*bool loop = true;
	while(loop) {
		loop = !(bool)cv::waitKey();
	}*/

	this->filename = filename;

	return 1;
}


//int FdImage::createPyramid(float factorFromOrig, float subsampFacForOneDown)
int FdImage::createPyramid(int pyrIdx, std::vector<int> pyrWidthList, std::string detectorId)
{
	if(Logger->getVerboseLevelText()>=3) {
		std::cout << "[FdImage] Creating pyramid (w=" << pyrWidthList[pyrIdx] << ")";
	}
	//calc width
	//int pyr_width = (int)(this->w*factorFromOrig+0.5);
	PyramidMap::iterator it = this->pyramids.find(pyrWidthList[pyrIdx]);
	if(it != this->pyramids.end()) {
		// already in map
		if(Logger->getVerboseLevelText()>=3) {
			std::cout << "... skipping, already created" << std::endl;
		}
		it->second->detectorIds.insert(detectorId);
		return 1;
	}
	if(Logger->getVerboseLevelText()>=3) {
		std::cout << std::endl;
	}
	// else: create pyramid
	// check if one-step-larger already exists. If yes, use that.
	//pyr_width = (int)(pyr_width/subsampFacForOneDown+0.5);
	if(pyrIdx!=0) {		// TODO I think there's room for improvement here. If pyrIdx is 0, there could nevertheless be a pyramid already created that we could use (i think)
		PyramidMap::iterator it = this->pyramids.find(pyrWidthList[pyrIdx-1]);
		if(it != this->pyramids.end()) {
			// one size bigger is already in map
			Pyramid *pyr = new Pyramid();
			float tmp = (float)pyrWidthList[pyrIdx]/(float)pyrWidthList[pyrIdx-1];
			this->rescale(*it->second, *pyr, tmp);
			pyr->detectorIds.insert(detectorId);
			pyramids.insert(PyramidMap::value_type(pyr->w, pyr));

			Logger->LogImgPyramid(pyr, this->filename, pyrIdx);
		}
	} else { // First pyramid and not already in map: create from orig.
		// create from orig-img
		Pyramid *pyr = new Pyramid();
		float tmp = (float)pyrWidthList[pyrIdx]/(float)this->w;
		this->rescale(*this, *pyr, tmp);
		pyr->detectorIds.insert(detectorId);
		pyramids.insert(PyramidMap::value_type(pyr->w, pyr));

		Logger->LogImgPyramid(pyr, this->filename, pyrIdx);
	}

	return 1;
}

int FdImage::rescale(StdImage& in, StdImage& out, float factor) // better: make this all pointer *
{
	if(factor<=0.0) {
		std::cout << "Error rescale, factor <=0!" << std::endl;
		return 0;
	}
	if (factor == 1.0) {
		if (out.data==NULL)	{
			out.w = in.w;
			out.h = in.h;
			out.data = new unsigned char[out.w*out.h];
		} else {
			std::cout << "You want to rescale the image to another image with factor==1, and the out image already has the data allocated. Something might have gone wrong?" << std::endl;
		}
  		memcpy(out.data, in.data, in.w*in.h); // TODO: unbenennen in data
  		return 1;
  	}
	out.h = (int)(in.h*factor+0.5); // 0.5 now not necessary with this method, but doesnt hurt
	out.w = (int)(in.w*factor+0.5);
	out.data = new unsigned char[out.w*out.h];

	/*for(int i=0; i<=2100; i++)
		std::cout << (int)in.data[i] << " "; // TODO remove (debug) */

  	int oil = 0;
	int oir = in.w;
  	int oit = 0;
	int oib = in.h;

  	// each pixel in the output image (x,y) maps into a region of pixels
  	// in the original image: ((int)xl + (float)xlf, (int)yt + (float)ytf,
  	//						   (int)xr + (float)xrf, (int)yb + (float)ybf)
  	// The width of this region is (int)x_int + (float)x_frac,
  	//			 and the height is (int)y_int + (float)y_frac,
  	// where the (float)s all satisfy 0.0 <= float <= 1.0

  	// pixel is used to grab pixels from the original image
  	// sumr, sumg, sumb add up all the pixel rgb values.

  	float sum;
  	float x_frac = 1.0f/factor;
  	float y_frac = 1.0f/factor;
  	int x_int = (int)x_frac;
  	int y_int = (int)y_frac;
  	x_frac -= x_int;
  	y_frac -= y_int;

  	int xl, xr, yt, yb,y,x,y_pix,x_pix;
  	float xlf, xrf, ytf, ybf;

  	yt = oit; ytf = 0.0f;

  	for (y = 0; y < out.h; y++) {
  		xl = oil; xlf = 0.0f;
  		yb = yt + y_int; ybf = ytf + y_frac;
  		if (ybf > 1.0f) {
  			ybf--; yb++;
  		}
  		for (x = 0; x < out.w; x++) {
  			sum = 0.0;
  			xr = xl + x_int; xrf = xlf + x_frac;
  			if (xrf > 1.0f) {
  				xrf--; xr++;
  			}
  			for (y_pix = std::max(oit, yt); y_pix < std::min(oib, yb); y_pix++) {
  				if (xl >= oil)
  					sum -= in.pixelAt(xl, y_pix) * xlf;
  				for (x_pix = std::max(oil, xl); x_pix < std::min(oir, xr); x_pix++)
  					  sum += in.pixelAt(x_pix, y_pix); 
//  					{ sum += in.pixelAt(x_pix, y_pix); if (x_pix>700) break; }
//printf("x_pix:%d, oir:%d, xr:%d, min:%d, x_pix < min(oir, xr):%d\n",x_pix,oir,xr,min(oir, xr),x_pix < (min(oir, xr)));
  				if (xr < oir)
  					sum += in.pixelAt(xr, y_pix) * xrf;
  			}
  			for (x_pix = std::max(oil, xl); x_pix < std::min(oir, xr); x_pix++) {
  				if (yt >= oit)
  					sum -= in.pixelAt(x_pix, yt) * ytf;
  				if (yb < oib)
  					sum += in.pixelAt(x_pix, yb) * ybf;
  			}
  			if (yt >= oit) {
  				if (xl >= oil)
  					sum += in.pixelAt(xl, yt) * xlf * ytf;
  				if (xr < oir)
  					sum -= in.pixelAt(xr, yt) * xrf * ytf;
  			}
  			if (yb < oib) {
  				if (xl >= oil)
  					sum -= in.pixelAt(xl, yb) * xlf * ybf;
  				if (xr < oir)
  					sum += in.pixelAt(xr, yb) * xrf * ybf;
  			}

  			double area_recip = factor*factor; //(out_img.Width() * out_img.Height());
  			sum *= (float)area_recip;
  			out.data[y*out.w+x] = (unsigned char)sum;
  			xl = xr; xlf = xrf;
  		}
  		yt = yb; ytf = ybf;
  	}
	/*for(int i=0; i<=1300; i++)
		std::cout << (int)out.data[i] << " "; // TODO remove (debug)
	std::cout << std::endl;*/
	return 1;

}

/*
	* Patrik: Bilinear interpolation/resampling method.
	* Sources: Numerical recipes (3rd edition) p. 132ff, http://www.alglib.net/interpolation/spline2d.php
	*/
/*int FdImage::rescaleBilinear(StdImage& in, StdImage& out, float factor) {

	assert(factor > 0.0);
	strcpy(out_img.filename,filename);

	const int newheight=(int)(h*factor+0.5);
	const int newwidth=(int)(w*factor+0.5);

	out_img.rowsize=(int)(rowsize * factor+0.5);
	out_img.colordepth=colordepth;
	out_img.Allocate((int)(w*factor+0.5), (int)(h*factor+0.5));

	if (factor == 1.0) {
		memcpy(out_img.data,data,rowsize*h);
		return;
	}
		
	int l, c;
	double t, u;

	for(int i = 0; i <= newheight-1; i++) {
		for(int j = 0; j <= newwidth-1; j++)
		{
			l = i*(h-1)/(newheight-1);
			if( l==h-1 ) {
				l = h-2;
			}

			u = double(i)/double(newheight-1)*(h-1)-l;
			c = j*(w-1)/(newwidth-1);
			if( c==w-1 ) {
				c = w-2;
			}
			t = double(j*(w-1))/double(newwidth-1)-c;
			out_img.data[i*newwidth+j] = (1-t)*(1-u)*data[l*w+c]+t*(1-u)*data[l*w+c+1]+t*u*data[(l+1)*w+c+1]+(1-t)*u*data[(l+1)*w+c];
		}
	}
}*/
