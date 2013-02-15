#pragma once
#include "StdImage.h"

#include "FdPoint.h"

#include <map>
#include <set>

class IImg;

typedef std::map<std::string, double> CertaintyMap;
typedef std::map<std::string, double> FoutMap;


class FdPatch : public StdImage
{
public:
	FdPatch(void);
	~FdPatch(void);

	FdPatch(int, int, int, int);

	FdPoint c;
	//int w, h;		// patch dimension in input image
	int w_inFullImg, h_inFullImg;		// patch dimension in pyramid image

	//CFdFFp ffp;
	//bool hasData;
	//unsigned char* data;
	CertaintyMap certainty;
	//double certainty;  // > 0 for a face, < 0 for a nonface (all the instances of this struct will have a certainty > 0)
	FoutMap fout;
	//double fout;
	
	unsigned int sampleID;	// TODO remove this somehow if possible or do it some other way... (eg new class). TRACKING.

	IImg* iimg_x;
	IImg* iimg_xx;


	class SortByCertainty {
	public:
		SortByCertainty(std::string detectorType) { this->detectorType = detectorType; };
		bool operator ()(FdPatch *lhs, FdPatch *rhs) {	// greater than ("<" would be "less" operator)
			return lhs->certainty.find(detectorType)->second > rhs->certainty.find(detectorType)->second;
		};
	private:
		std::string detectorType;
	};
 

};

struct FdPatch_comp		// TRUE if a > b, FALSE if a <= b	(TODO hm my sort is the other way round?)
{
    bool operator () (const FdPatch* a, const FdPatch* b) const
    {
        // sort
		if(a->w < b->w) {
            return true;
		}
		if(a->w > b->w) {
			return false;
		}
		//else (equality)
		if(a->h < b->h) {
            return true;
		}
		if(a->h > b->h) {
			return false;
		}

		//else
		if(a->c.y_py < b->c.y_py) {
            return true;
		}
		if(a->c.y_py > b->c.y_py) {
			return false;
		}

		//else
		if(a->c.x_py < b->c.x_py) {
            return true;
		}
		if(a->c.x_py > b->c.x_py) {
			return false;
		}

		// I SHOULD NEVER REACH HERE BECAUSE THE ELEMENTS (PATCHES) SHOULD BE UNIQUE
		// (sure I land here when I try to insert a new patch that is already in the patchset)
		return false;

	}
};

typedef std::set<FdPatch*, FdPatch_comp> FdPatchSet;	// The pyramids use this for efficiently storing their patches
