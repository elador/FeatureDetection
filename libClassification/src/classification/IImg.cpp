#include "classification/IImg.hpp"

namespace classification {

IImg::IImg(void) : data(0), w(0), h(0), colordepth(0)
{
	rowsize=0;
	size=0;
}

IImg::IImg(int _w, int _h, int _colordepth=32) : data(0), w(_w), h(_h), colordepth(_colordepth) {
	rowsize=(int)colordepth/8*w;
	if (w && h)
		data = new float[rowsize*h];
	size=w*h;
}


IImg::~IImg(void)
{
	delete[] data;
	data = 0;
}


void IImg::calIImgPatch(const unsigned char* in_img, bool sqr=false)
{
			
	int c,r;
	long z,zb;
	float rowsum;

	if (sqr) {
		rowsum=0;
		for (c=0;c<w;c++) {
			rowsum+=in_img[c]*in_img[c];
			data[c]=rowsum;
		}
		z=w; zb=0;
		for (r=1;r<h;r++) {
			rowsum=0;
			for (c=0;c<w;c++) {
				rowsum+=in_img[z+c]*in_img[z+c];
				data[z+c]=data[zb+c]+rowsum;
			}
			z+=w; zb+=w;
		}
	} else {
		rowsum=0;
		for (c=0;c<w;c++) {
			rowsum+=in_img[c];
			data[c]=rowsum;
		}

		z=w; zb=0;
		for (r=1;r<h;r++) {
			rowsum=0;
			for (c=0;c<w;c++) {
				rowsum+=in_img[z+c];
				data[z+c]=data[zb+c]+rowsum;
			}
			z+=w; zb+=w;
		}
	}
}

} /* namespace classification */


//float IImg::Max(int _x /*=0*/, int _y /*=0*/, int _w /*=0*/, int _h /*=0*/)
/*{
 	int		last_col,last_row,i,j;
	long	z;
	float	maxv;

	if (_w==0) _w=w;
	if (_h==0) _h=h;

	last_col=_x+_w;
	last_row=_y+_h;

	#ifdef _DEBUG
		if ( (_y>h) || (_y<0) || (_x>w) || (_x<0) ||
			(last_row>h) || (last_row<0) || (last_col>w) || (last_col<0) ) {
			printf("Max(): cut outside of the image\n");
			return -1;
		}
	#endif

	z=(_y)*w; maxv=data[z+_x];
	for(j = _y; j < last_row; ++j) {
		for(i = _x; i < last_col; ++i) 
			maxv= max(maxv,data[z+i]);
		z+=w;
	}

	return maxv;
}
*/
//float IImg::Min(int _x /*=0*/, int _y /*=0*/, int _w /*=0*/, int _h /*=0*/) 
/*{
 	int		last_col,last_row,i,j;
	long	z;
	float	minv;

	if (_w==0) _w=w;
	if (_h==0) _h=h;

	last_col=_x+_w;
	last_row=_y+_h;

	#ifdef _DEBUG
		if ( (_y>h) || (_y<0) || (_x>w) || (_x<0) ||
			(last_row>h) || (last_row<0) || (last_col>w) || (last_col<0) ) {
			printf("Min(): cut outside of the image\n");
			return -1;
		}
	#endif

	z=(_y)*w; minv=1e30f;
	for(j = _y; j < last_row; ++j) {
		for(i = _x; i < last_col; ++i) 
			if (data[z+i]!=NOTCOMPUTED)
				minv= min(minv,data[z+i]);
		z+=w;
	}

	if(minv==1e30f) {
		fprintf(stderr,"\n\n\tWARNING: Min(): no point != NOTCOMPUTED => min=NOTCOMPUTED(%g)\n\n",NOTCOMPUTED);
		minv=NOTCOMPUTED;
	}

	return minv;
}
*/