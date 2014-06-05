/*
 * HistogramEqualizationFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistEq64Filter.hpp"

using cv::Mat;

namespace imageprocessing {

HistEq64Filter::HistEq64Filter() {
	// initialization for the histogram equalization:
	LUTbin = new unsigned char[256];
	// init LUT_bin for 64 bins:
	// LUT_bin should look like: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 ... 63 63 63 63
	for(int i=0; i<64; i++) {
		LUTbin[4*i]=i;
		LUTbin[4*i+1]=i;
		LUTbin[4*i+2]=i;
		LUTbin[4*i+3]=i;
	}
}

HistEq64Filter::~HistEq64Filter() {
	delete[] LUTbin;
	LUTbin = NULL;
}

Mat HistEq64Filter::applyTo(const Mat& image, Mat& filtered) const {
	//cv::equalizeHist(image, filtered);
	float stretchFactor = 255.0f/(float)(image.cols*image.rows);
	filtered.create(image.rows, image.cols, CV_8U);
	int rows = image.rows;
	int cols = image.cols;
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	//fp->data = new unsigned char[filter_size_x*filter_size_y];
	// DO THIS ONCE PER PATCH: //

	float pdf_bins[64]; // can contain up to value '400', so uchar is not enough (255).
							// need float anyway because we multiply by stretch_fac later.
	// pdf_bins initialized with zeros.
	for(int i=0; i<64; i++) {
		pdf_bins[i]=0.0f;
	}
	
	// fill our histogram with the values from the IMAGE PYRAMID
										// patch-height/2                         patch-width/2
	//int start_orig = (patch_cy_py-(10))*this_pyramid->w + (patch_cx_py-(10)); // w=rowsize // 10 would be logical (or 9), but I only get the same results as matthias when I use 11 ???? (some image-errors then, but quite logical, considering 10-11=-1)
	//int start_orig = (fp->c.y_py-(filter_size_y/2))*pyr->w + (fp->c.x_py-(filter_size_x/2));
	//int start_orig = 0;
	//int startcoord = start_orig;
	//int index=0;    // patch-height, det_filtersize_y
	for (int z=0; z<rows; z++) {		// could be made with modulo and 1 for loop 0-399 but this is supposed to be faster
					 // patch-width, det_filtersize_x
		const uchar* originalValues = image.ptr<uchar>(z);
		for(int i=0; i<cols; i++) {
			pdf_bins[LUTbin[originalValues[i]]]=pdf_bins[LUTbin[originalValues[i]]]+1;	// TODO I should not use image.data here! Use .at<>(i,j) !
			//startcoord++;
		}									// patch-width
		//startcoord=startcoord+(pyr->w-filter_size_x);
	}

	// stretch the probability density function (PDF)
	for(int i=0; i<64; i++) {
		if(pdf_bins[i]!=0) {
			pdf_bins[i]=pdf_bins[i]*stretchFactor;
		}
	}
	// TODO: maybe round pdf_bins to 4 decimals here to prevent multiplying of the inaccuracy
	// cdf_BINS=cumsum(pdf_bins):
	float cdf_BINS[64];
	cdf_BINS[0]=pdf_bins[0];
	for(unsigned int i=1; i<64; i++) {
		cdf_BINS[i]=cdf_BINS[i-1]+pdf_bins[i];
	}

	// fill the equalized look-up table
	float LUTeq[256];
	for(int i=0; i<256; i++) {
		LUTeq[i]=cdf_BINS[LUTbin[i]];
	}
	//printf("HistEq64 - Fill the patch with data.\n");
	// equalize the patch with the help of the LUT:
	//index=0;    // patch-height, det_filtersize_y
	for (int z=0; z<rows; z++) {		// could be made with modulo and 1 for loop 0-399 but this is supposed to be faster
					 // patch-width, det_filtersize_x
		const uchar* originalValues = image.ptr<uchar>(z);
		uchar* filteredValues = filtered.ptr<uchar>(z);
		for(int i=0; i<cols; i++) {
			//patch_to_equalize[index] = this_pyramid->data[startcoord]; // original
			filteredValues[i] = (uchar)floor(LUTeq[originalValues[i]]+0.5);
			/*if(LUTeq[this_pyramid->data[start_orig]] > 255) {
				printf("hq error 1...\n");
			}*/
			//if(floor(LUTeq[pyr->data[start_orig]]+0.5) > 255) { // deleted because of speed :-)
				//printf("\nhq overflow error: %f -> %d. Check the stretch factor!\n", floor(LUTeq[pyr->data[start_orig]]+0.5), (unsigned char)floor(LUTeq[pyr->data[start_orig]]+0.5));

				/* for(int tt=0; tt<256; tt++) {
					printf("%d, ", LUT_bin[tt]);
				} LUT_bin is OK */

			//}// deleted because of speed :-)
			/*if((unsigned char)floor(LUTeq[this_pyramid->data[start_orig]]+0.5) > 255) {
				printf("hq error 3...\n");
			}*/
			//start_orig++;
			//index++;
		}									// patch-width
		//start_orig=start_orig+(pyr->w-filter_size_x);
	}

	// we are finished! result is equal to the result from the matlab algorithm.
	// (double checked. A few values are off by 1 but this is supposed to be a small rounding/precision issue that does not really make a difference)




	return filtered;
}

void HistEq64Filter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
