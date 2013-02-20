/*
 * IImg.hpp
 *
 *  Created on: 2011/2012
 *      Author: Patrik Huber
 */

#pragma once

#ifndef IIMG_HPP_
#define IIMG_HPP_

namespace classification {

/**
 * Integral Image class that represents (stores) integral images and is able to calculate them for a patch.
 */
class IImg
{
	//IImg(void);
	//~IImg(void);

public:
	float* data;
	int w, h, size, rowsize, colordepth;

	//std::string filename;

	IImg(void);
	IImg(int, int, int);
	~IImg();

	void calIImgPatch(const unsigned char*, bool); // Calculate the integral image of a patch (needed for WVM histogram equalization)

};

} /* namespace classification */
#endif /* IIMG_HPP_ */


/*
inline int Width()  const {return w;}
	inline int Height() const {return h;}
	inline CRect Rect() const {return CRect(0, 0, w, h);}
	inline int Left()   const {return 0;}
	inline int Top()    const {return 0;}
	inline int Right()  const {return w;}
	inline int Bottom() const {return h;}
	inline int RowSize() const {return rowsize;}
	inline int ColorDepth() const {return colordepth;}
	inline float  Pixel(int x, int y) const {return data[y*w+x];}
	inline float& Pixel(int x, int y)       {return data[y*w+x];}
	float Min(int _x=0, int _y=0, int _w=0, int _h=0); 
	float Max(int _x=0, int _y=0, int _w=0, int _h=0); 
	inline virtual float CalPoint(const CStdImg& in_img, long z, int c) { return (in_img.data[z+c]); };
	inline float CalPoint(const CStdImg& in_img, long z) { return ((float)in_img.data[z]); };
	inline float CalPointSqr(const CStdImg& in_img, long z) { return ((float)in_img.data[z]*(float)in_img.data[z]); };
	int Write(const char *filename="", int _x=0, int _y=0, int _w=0, int _h=0, char *colormode="");

  float Sum(int x1=0, int y1=0, int x2=-1, int y2=-1) {

  	if (x2==-1) x2=w-1;
  	if (y2==-1) y2=h-1;

  	#ifdef _DEBUG
  		if (y1>y2 || x1>x2) {
  			fprintf(stderr,"\n\nERROR: Sum(): koord. mixed: (x1,y1,x2,y2):%d,%d,%d,%d ( y1<=y2 && x1<=x2 )\n\n",
  				x1,y1,x2,y2);
  			assert(y1<=y2 && x1<=x2);
  		}
  		if (y1<0 || y2>h || x1<0 || x2>w) {
  			fprintf(stderr,"\n\nERROR: Sum(): out of image: (x1,y1,x2,y2):%d,%d,%d,%d ( x(0-%d), y(0-%d) )\n\n",
  				x1,y1,x2,y2,w,h);
  			assert(y1<=0 && y2<=h && x1>=0 && x2<=w);
  		}
  	#endif

  	long z=y1*w; float sum=0;
  	for (int y=y1; y<=y2; y++) {
  		for (int x=x1; x<=x2; x++)
  			sum+=data[z+x];
  		z+=w;
  		}

  	return sum;
  }

  inline float ISum(int x1=0, int y1=0, int x2=-1, int y2=-1) const {

  	if (x2==-1) x2=w-1;
  	if (y2==-1) y2=h-1;

  	#ifdef _DEBUG
  		if (y1>y2 || x1>x2) {
  			fprintf(stderr,"\n\nERROR: ISum(): koord. mixed: (x1,y1,x2,y2):%d,%d,%d,%d ( y1<=y2 && x1<=x2 )\n\n",
  				x1,y1,x2,y2);
  			assert(y1<=y2 && x1<=x2);
  		}
  		if (y1<0 || y2>=h || x1<0 || x2>=w) {
  			fprintf(stderr,"\n\nERROR: ISum(): out of image: (x1,y1,x2,y2):%d,%d,%d,%d ( x(0-%d), y(0-%d) )\n\n",
  				x1,y1,x2,y2,w-1,h-1);
  			assert(y1<=0 && y2<h && x1>=0 && x2<w);
  		}
  	#endif

  	return ( data[y2*w+x2] - ((y1>0)? data[(y1-1)*w+x2]:0) - ((x1>0)? data[y2*w+x1-1]:0) + ((x1>0 && y1>0)? data[(y1-1)*w+x1-1]:0) );
  }

  inline float ISumV(int uull,int uur,int dll,int dr,int x1 ,int y1 ,int x2 ,int y2) const {

  	#ifdef _DEBUG
  		if (y1>y2 || x1>x2) {
  			fprintf(stderr,"\n\nERROR: ISum(): koord. mixed: (x1,y1,x2,y2):%d,%d,%d,%d ( y1<=y2 && x1<=x2 )\n\n",
  				x1,y1,x2,y2);
  			assert(y1<=y2 && x1<=x2);
  		}
  		if (y1<0 || y2>=h || x1<0 || x2>=w) {
  			fprintf(stderr,"\n\nERROR: ISum(): out of image: (x1,y1,x2,y2):%d,%d,%d,%d ( x(0-%d), y(0-%d) )\n\n",
  				x1,y1,x2,y2,w-1,h-1);
  			assert(y1<=0 && y2<h && x1>=0 && x2<w);
  		}
  	#endif

  	return ( data[dr] - ((y1>0)? data[uur]:0) - ((x1>0)? data[dll]:0) + ((x1>0 && y1>0)? data[uull]:0) );
  }
*/











	/*inline IImg(const CStdImg& in_img) : data(0), w(in_img.w), h(in_img.h), rowsize(sizeof(float)*w), colordepth(32) {
		if (w && h)
			data = new float[rowsize*h];
		size=w*h;
		strcpy(filename,in_img.filename);    
		CalIImg(in_img);
	}

	inline IImg(const IImg& rhs) : data(0) {
		size = w*h;
		strcpy(filename,rhs.filename);
		colordepth=rhs.colordepth;
		rowsize=rhs.rowsize;
		Allocate(rhs.w, rhs.h);

		strcpy(filename,rhs.filename);
		memcpy(data,rhs.data,rowsize*h);
	}
	inline IImg& operator= (const IImg& rhs) {
		size = rhs.w*rhs.h;
		colordepth=rhs.colordepth;
		rowsize=rhs.rowsize;
		Allocate(rhs.w, rhs.h);

		strcpy(filename,rhs.filename);
		memcpy(data,rhs.data,rowsize*h);

		return *this;
	}**/

	/**
	 * \deprecated
	 * (Kommentar Lorenzo)
	 * Das ist gefährlich! wenn ich nun nacher "rhs" lösche?
	 * wenn rhs const sein soll muss data eigentlich kopiert werden
	 * Verwendung nur in fd_DetectImg Zeile 230
	 *//*
	inline IImg& operator= (const IImg* rhs) {
		w=rhs->w; h=rhs->h;
		size = rhs->size;
		colordepth=rhs->colordepth;
		rowsize=rhs->rowsize;
        data = rhs->data;
		strcpy(filename,rhs->filename);
		return *this;
	}*/

	/**
	 * \author Lorenzo
	 * Alternative für operator=(const CImg*) die in diesem Zusammenhang passt:
     *//*
	inline void construct(int _w, int _h, int _colordepth=3)
	{
		IImg tmp(_w,_h,_colordepth);
		w=tmp.w;
		h=tmp.h;
		size=tmp.size;
		colordepth=tmp.colordepth;
		rowsize=tmp.rowsize;
		data=tmp.data;
		strcpy(filename,tmp.filename);

		// PREVENT FROM DELETION, which was the problem with operator=(const IImg*)
		tmp.data=NULL;
	}


	inline IImg& operator= (const CStdImg& rhs) {
		if (w!=rhs.w || h!=rhs.h || colordepth!=32 || rowsize!=(int)sizeof(float)*rhs.w)  {
			if (!w || !h)  delete [] data;
			w=rhs.w; h=rhs.h; colordepth=32; rowsize=sizeof(float)*w; 
            data = new float[rowsize*h];
			size=w*h;
			strcpy(filename,rhs.filename); 
		}
		CalIImg(rhs);

		return *this;
	}
	inline void Allocate(int _w, int _h) {
		w = _w;
		h = _h;
		size=w*h;
		if (data!=NULL) delete [] data;
		data = new float[rowsize*h];
	}
	inline bool operator== (const IImg& rhs) const {

		if (w!=rhs.w || h!=rhs.h || colordepth!=rhs.colordepth || rowsize!=rhs.rowsize)
			return false;
		
		for (long s=0;s<size;s++)
			if (data[s]!=rhs.data[s])
				return false;

		return true;
	}
	
	inline void CalIImg(const CStdImg& in_img, bool sqr=false) {
			
		int c,r;
		long z,zb;
		float rowsum;

		if (sqr) {
			rowsum=0;
			for (c=0;c<w;c++) {
				rowsum+=CalPointSqr(in_img,c);
				data[c]=rowsum;
			}
			z=w; zb=0;
			for (r=1;r<h;r++) {
				rowsum=0;
				for (c=0;c<w;c++) {
					rowsum+=CalPointSqr(in_img,z+c);
					data[z+c]=data[zb+c]+rowsum;
				}
				z+=w; zb+=w;
			}
		} else {
			rowsum=0;
			for (c=0;c<w;c++) {
				rowsum+=CalPoint(in_img,c);
				data[c]=rowsum;
			}

			z=w; zb=0;
			for (r=1;r<h;r++) {
				rowsum=0;
				for (c=0;c<w;c++) {
					rowsum+=CalPoint(in_img,z+c);
					data[z+c]=data[zb+c]+rowsum;
				}
				z+=w; zb+=w;
			}
		}
	}*/
