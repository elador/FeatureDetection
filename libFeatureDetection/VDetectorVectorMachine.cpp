#include "stdafx.h"
#include "VDetectorVectorMachine.h"

#include "FdImage.h"
#include "Pyramid.h"

#include <iostream>

VDetectorVectorMachine::VDetectorVectorMachine(void)
{

	identifier = "DetectorVectorMachine";

	nonlin_threshold = 0;
	nonLinType = 0;
	basisParam = 0;
	polyPower = 0;
	divisor = 0;

	filter_size_x = 0;
	filter_size_y= 0;

	subsamplingMinHeight = 0;
	subsamplingFactor = 0;
	//subsampfac = NULL;
	numSubsamplingLevels = 0;
	subsamplingLevelStart = 0;
	subsamplingLevelEnd = 0;

	pyramid_widths = NULL;
	
	// initialization for the histogram equalization:
	LUT_bin = new unsigned char[256];
	// init LUT_bin for 64 bins:
	// LUT_bin should look like: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 ... 63 63 63 63
	for(int i=0; i<64; i++) {
		LUT_bin[4*i]=i;
		LUT_bin[4*i+1]=i;
		LUT_bin[4*i+2]=i;
		LUT_bin[4*i+3]=i;
	}
	stretch_fac = 0.0;
}


VDetectorVectorMachine::~VDetectorVectorMachine(void)
{
	//delete[] subsampfac;
	delete[] pyramid_widths;
	delete[] LUT_bin; LUT_bin=NULL;
}

int VDetectorVectorMachine::initPyramids( FdImage *img )
{

	if(subsamplingMinHeight<filter_size_y) {
		std::cout << "Error: subsamplingMinHeight < filter_size_y" << std::endl;
	}
	//int numSubsamplingLevels = subsamplingLevels;
	// with 2^x=b => log(b)=x, l: levels, f: factor, w: weight =>
	// w*f^l = 20 => f^l = 20/w => l*log(f) = log(20/w)  =>   l = log(20/w) / log(f)
	subsamplingLevelEnd = (int) std::min(
		floor(log(filter_size_x/(float)img->w)/log(subsamplingFactor)),
		floor(log(filter_size_y/(float)img->h)/log(subsamplingFactor)) );

	subsamplingLevelStart = (int) std::min(
		subsamplingLevelEnd,
		(int)floor(log(filter_size_y/(float)std::max(filter_size_y, subsamplingMinHeight))/log(subsamplingFactor)) );

	if (subsamplingLevelEnd != -1) // never? if not set in config, WARNING and is 0
 		subsamplingLevelEnd = std::min(subsamplingLevelEnd, subsamplingLevelStart+numSubsamplingLevels-1);
 
 	if(subsamplingLevelEnd<=0) {
		fprintf(stderr,"\n\nERROR: subsamplingLevels(%d)<=0 => Image to small (h,w:%d,%d)\n",subsamplingLevelEnd,img->h,img->w);
 		return(-1);
 	}

	printf("[VDetVecMach] subsamplingLevels:%g^%d->%g^%d (img(h,w:)%d,%d) => imgh min:%d, max:%d; => face_h min:%d, max:%d; maxpossiblelev(w,h):%g,%g, arg.maxlev:%d\n",
 		subsamplingFactor,subsamplingLevelStart,subsamplingFactor,subsamplingLevelEnd,img->h,img->w,
 		(int)(img->h*(float)pow(subsamplingFactor,subsamplingLevelEnd)), (int)(img->h*(float)pow(subsamplingFactor,subsamplingLevelStart)),
 		(int)(filter_size_y/(float)pow(subsamplingFactor,subsamplingLevelStart)), (int)(filter_size_y/(float)pow(subsamplingFactor,subsamplingLevelEnd)),
 		floor(log(filter_size_x/(float)img->w)/log(subsamplingFactor)),
 		floor(log(filter_size_y/(float)img->h)/log(subsamplingFactor)), numSubsamplingLevels
 		);

	/*if(subsampfac!=NULL) {
		delete[] subsampfac;
		subsampfac = NULL;
	}*/
	if(subsampfac.size()>0)	// If I come here and its not empty, then I need to delete it, because I'm working on a completely new image!
		subsampfac.empty();

	if(pyramid_widths!=NULL) {
		delete[] pyramid_widths;
		pyramid_widths = NULL;
	}
	//subsampfac = new float[numSubsamplingLevels];	// we only need 9!
	//subsampfac[0] = (float)pow(subsamplingFactor,subsamplingLevelStart);

	pyramid_widths = new int[numSubsamplingLevels];

	//int direct = (int)(img->w*(float)pow(subsamplingFactor,subsamplingLevelStart)+0.5);
	int curr_width=img->w;
	for(int i=1; i<=subsamplingLevelStart; i++) {
		curr_width = (int)(curr_width*subsamplingFactor+0.5);
	}
	subsampfac.insert(std::map<int, float>::value_type(curr_width, (float)curr_width/(float)img->w));
	//subsampfac[0] = (float)curr_width/(float)img->w;
	pyramid_widths[0] = curr_width;
	for(int i=1; i<numSubsamplingLevels; i++) {
		pyramid_widths[i] = (int)(pyramid_widths[i-1]*subsamplingFactor+0.5);
		//curr_width = (int)(curr_width*subsamplingFactor+0.5);
		//subsampfac[i] = (float)pyramid_widths[i]/(float)img->w;
		subsampfac.insert(std::map<int, float>::value_type(pyramid_widths[i], (float)pyramid_widths[i]/(float)img->w));
	}

	/*img->createPyramid(subsampfac[0], subsamplingFactor);
	for(int i=1; i<numSubsamplingLevels; i++) {
		subsampfac[i] = subsampfac[i-1] * subsamplingFactor;
		img->createPyramid(subsampfac[i], subsamplingFactor);
	}*/
	for(int i=0; i<numSubsamplingLevels; i++) {
		img->createPyramid(i, pyramid_widths, this->identifier);	// pyramid_widths unnecessary now? subsampfac has everything in it?
	}
	//TODO: Sanity check:
		//if ((pyramid[i].Width()<=detector.filter_size.cx) || (pyramid[i].Height()<=detector.filter_size.cy)) {
 		//printf("WARNING: scale_img too small (rounding effects) => subsamplingLevels: %d->%d\n",subsamplingLevels,j-1);

	return 1;
}

int VDetectorVectorMachine::extractToPyramids(FdImage *img)
{
	std::cout << "[VDetVecMach] Extracting..." << std::endl;
	//for(int current_scale = 0; current_scale < this->numSubsamplingLevels; current_scale++) {// TODO: Careful: numSubsamplingLevels is not necessarily the actual number of available pyramids. if the image is too small, it may be less...
	for(int current_scale = this->numSubsamplingLevels-1; current_scale >=0 ; current_scale--) {

		PyramidMap::iterator it = img->pyramids.find(this->pyramid_widths[current_scale]);
		float coef = this->subsampfac[this->pyramid_widths[current_scale]];
		//float coef = this->subsampfac[current_scale];
		int pyrw = it->second->w;
		int pyrh = it->second->h;

		Rect roi((int)(this->roi_inImg.left*coef), (int)(this->roi_inImg.top*coef), (int)(this->roi_inImg.right*coef), (int)(this->roi_inImg.bottom*coef));

		int stepsize_from_config = 1;
		int ss=std::max(1, int(coef * stepsize_from_config + 0.5f)); // stepsize, from config
		//int ss = 1;
		printf("[VDetVecMach] Scale: %d: faceh:%d, coef:%1.4f, step:%d (%1.4f)\n",current_scale,(int)(filter_size_y/coef),coef,ss,coef * stepsize_from_config);
		int h_prim = std::min(pyrh-filter_size_y/2,roi.bottom);
		int w_prim = std::min(pyrw-filter_size_x/2,roi.right);
		int y1, y2, x1, x2;
		for (int y = std::max(roi.top,filter_size_y/2),y1=y-filter_size_y/2,y2=y+filter_size_y/2-1; y < h_prim; y+=ss,y1+=ss,y2+=ss) {
			for (int x = std::max(filter_size_x/2,roi.left),x1=x-filter_size_x/2,x2=x+filter_size_x/2-1; x < w_prim; x+=ss,x1+=ss,x2+=ss) {
				
				//Check if the patch is already in the Set
				FdPatch* fp = new FdPatch();	// we dont have to delete this because the FdPatchSet takes ownership!
				fp->c.x_py = x;
				fp->c.y_py = y;	
				fp->w = filter_size_x;
				fp->h = filter_size_y;	

				std::pair<FdPatchSet::iterator, bool> patch_pair = it->second->patches.insert(fp);

				if(patch_pair.second == true) {	// insert made, fp is new
					fp->c.s = pyrw; // current_scale is not unique! Better use pyrw instead!

					fp->c.x=(int)(x/coef);
					fp->c.y=(int)(y/coef);
					fp->w_inFullImg=(int)(filter_size_x/coef);
					fp->h_inFullImg=(int)(filter_size_y/coef);
					//fp.setHasData(true);
					extractAndHistEq64(it->second, fp);	// Extract + Normalize (1 step). Data is now in fp
														// it->second is the current pyramid
				} else {
					//std::cout << "!";
					delete fp;	// because we dont use this, but use the one thats already in the patchlist!
				}
				// Its in the set for sure now: (either all along, or newly)
				// TODO. For now: If its in the set, it is already hq64 normalized.
			}
		}

	}
	return 1;
}

std::vector<FdPatch*> VDetectorVectorMachine::getPatchesROI(FdImage *img, int x_py, int y_py, int scale, int x_offset, int y_offset, int s_offset, std::string detectorId)
{
	std::cout << "[VDetVecMach] Extracting patches around a ROI..." << std::endl;

	std::vector<FdPatch*> extractedPatches;

	for(unsigned int curr_x = x_py-x_offset; curr_x <= x_py+x_offset; ++curr_x) {
		for(unsigned int curr_y = y_py-y_offset; curr_y <= y_py+y_offset; ++curr_y) {
			for(int curr_s = -s_offset; curr_s <= s_offset; ++curr_s) {	// have to find the right pyramid
				int idx; // Todo maybe a failsafe... if scale not found...
				for(int i=0; i<this->numSubsamplingLevels; ++i) {
					if(this->pyramid_widths[i]==scale) {
						idx=i;
						break;
					}
				}
				PyramidMap::iterator it = img->pyramids.find(this->pyramid_widths[idx+curr_s]);
				// IF NOT FOUND, SKIP. TODO: UNTESTED
				if(it == img->pyramids.end()) {
					std::cout << "Patch is at border. No lower/higher scale to use for ROI." << std::endl;
					continue;
				}
				float coef = this->subsampfac[this->pyramid_widths[idx+curr_s]];
				int pyrw = it->second->w;
				int pyrh = it->second->h;

				//Check if the patch is already in the Set
				FdPatch* fp = new FdPatch();	// we dont have to delete this because the FdPatchSet takes ownership!
				fp->c.x_py = curr_x;
				fp->c.y_py = curr_y;	
				fp->w = filter_size_x;
				fp->h = filter_size_y;	

				//std::pair<FdPatchSet::iterator, bool> patch_pair = it->second->patches.insert(fp);
				FdPatchSet::iterator pit = it->second->patches.find(fp);
				//if(patch_pair.second == true) {	// insert made, fp is new
				if(pit == it->second->patches.end()) {
					// TODO Sanity check if the ROI of the patch isn't out of the whole-img-ROI.
					// For now: We take for granted that the whole det-ROI is in the pyramid. So if the patch is not in the pyramid here, we can skip it as it is out of the whole-img-ROI/out of img!

					std::cout << "Patch is not in pyramid, probably at border / not in whole-img-ROI. Skipping..." << std::endl;
					/*fp->c.s = scale;
					fp->c.x=(int)(curr_x/coef);	// coef...
					fp->c.y=(int)(curr_y/coef);
					fp->w_inFullImg=(int)(filter_size_x/coef);
					fp->h_inFullImg=(int)(filter_size_y/coef);
					//fp.setHasData(true);
					extractAndHistEq64(it->second, fp);	// Extract + Normalize (1 step). Data is now in fp
														// it->second is the current pyramid
					//TODO add fp to set
					//it->second->patches.insert(fp);	// this_pyr->patches
					extractedPatches.push_back(fp);*/
				} else {
					//std::cout << "!";
					delete fp;	// because we dont use this, but use the one thats already in the patchlist!
					//extractedPatches.push_back( (*patch_pair.first) );
					extractedPatches.push_back(*pit);
				}
			}
		}
	}
	return extractedPatches;
}


// Returns a vector with the extracted, valid patches!
/*std::vector<FdPatch*> VDetectorVectorMachine::extractPatches(FdImage *img, std::vector<FdPatch*>& patchesToExtract)
{
	std::cout << "[VDetVecMach] Extracting patches..." << std::endl;

	std::vector<FdPatch*> extractedPatches;

	for(unsigned int i = 0; i < patchesToExtract.size(); ++i) {

		PyramidMap::iterator it = img->pyramids.find(this->pyramid_widths[patchesToExtract[i]->c.s]);
		float coef = this->subsampfac[patchesToExtract[i]->c.s];
		int pyrw = it->second->w;
		int pyrh = it->second->h;

		//Check if the patch is already in the Set
		FdPatch* fp = new FdPatch();	// we dont have to delete this because the FdPatchSet takes ownership!
		fp->c.x_py = patchesToExtract[i]->c.x_py;
		fp->c.y_py = patchesToExtract[i]->c.y_py;	
		fp->w = filter_size_x;
		fp->h = filter_size_y;	

		std::pair<FdPatchSet::iterator, bool> patch_pair = it->second->patches.insert(fp);
		//new FdPatch(x, y, filter_size_x, filter_size_y)
		if(patch_pair.second == true) {	// insert made, fp is new
			//*(*(patch_pair.first))->fout; // fp points to same adress, I can use fp?
			
			fp->c.s = patchesToExtract[i]->c.s; // maybe I need to/should use the pyr_width here instead?
					
			//fp->certainty = NOTCOMPUTED;
			// TODO: (maybe) Write in a set/map/vec, what ffp-detector went over it
			fp->c.x=patchesToExtract[i]->c.x; //TODO(int)(x/coef);
			fp->c.y=patchesToExtract[i]->c.y; //(int)(y/coef);
			fp->w_inFullImg=(int)(filter_size_x/coef);
			fp->h_inFullImg=(int)(filter_size_y/coef);
			//fp.setHasData(true);
			extractAndHistEq64(it->second, fp);	// Extract + Normalize (1 step). Data is now in fp
												// it->second is the current pyramid
			//TODO add fp to set
			//it->second->patches.insert(fp);	// this_pyr->patches
			extractedPatches.push_back(fp);
		} else {
			//std::cout << "!";
			delete fp;	// because we dont use this, but use the one thats already in the patchlist!
			extractedPatches.push_back( (*patch_pair.first) );
		}
		// Its in the set for sure now: (either all along, or newly)
		// TODO. For now: If its in the set, it is already hq64 normalized.

		//std::cout << "px: "<< fp->c.x_py << ", py: " << fp->c.y_py << ", fout: " << fp->fout << std::endl;
		//if(fp->fout>0.5)
		//	fp->writePNG();
				
		//FdPatchSet::iterator pit = it->second->patches.find();
		/*FdPatchSet::iterator pit = it->second->patches.begin();
		std::cout << "===Start patchlist===" << std::endl;
		for(; pit != it->second->patches.end(); pit++) {
			std::cout << (*pit)->w << ", " << (*pit)->h << ", " << (*pit)->c.x_py << ", " << (*pit)->c.y_py << std::endl;
		}
		FdPatchSet::iterator p2it = it->second->patches.find(fp);
		std::cout << fp->fout << std::endl;
		std::cout << (*p2it)->fout << std::endl;



	}
	return extractedPatches;
}*/


/*
Classifies all patches in the PatchList of the pyramids. (maybe too many... fix later).
Updates the patches in the PatchList.
Additionally, returns a vec with pointer to candidates (which are PatchList) which can be used.
*/
std::vector<FdPatch*> VDetectorVectorMachine::detect_on_image(FdImage *img)
{

	std::cout << "[VDetVecMach] Detecting on image..." << std::endl;
	std::vector<FdPatch*> candidates;

	//for(int current_scale = 0; current_scale < this->numSubsamplingLevels; current_scale++) {// TODO: Careful: numSubsamplingLevels is not necessarily the actual number of available pyramids. if the image is too small, it may be less...
	for(int current_scale = this->numSubsamplingLevels-1; current_scale >=0 ; current_scale--) { // For each pyramid, go through the patchlist and classify

		PyramidMap::iterator it = img->pyramids.find(this->pyramid_widths[current_scale]);
		float coef = this->subsampfac[this->pyramid_widths[current_scale]];

		FdPatchSet::iterator pit = it->second->patches.begin();

		bool passed = false;
		for(; pit != it->second->patches.end(); pit++) {
			passed = classify((*pit));	// We could do the whole classification in a separate loop over the pyramids now. But this is ok for now. (maybe even faster)
			if(passed) {
				candidates.push_back((*pit)); // SVM&WVM only added if feature, SVR/WVR always added
			}
		}

	}
	return candidates;
}

// classifies all in patchvec.
// Returns:
//		-WVM: All that passed last filter
//		-SVM: All >0 (respectively limit_reliability)
std::vector<FdPatch*> VDetectorVectorMachine::detect_on_patchvec(std::vector<FdPatch*> &patchvec)
{
	std::cout << "[VDetVecMach] Detecting on list of patches..." << std::endl;
	std::vector<FdPatch*> candidates;

	bool passed = false;
	bool warningPrinted = false;
	std::vector<FdPatch*>::iterator itr;
	for (itr = patchvec.begin(); itr != patchvec.end(); ++itr ) {
		if((*itr)->w != this->filter_size_x || (*itr)->h != this->filter_size_y) {
			if(!warningPrinted) {
				std::cout << "[VDetVecMach] Warning: This detector has another filter_size (w=" << this->filter_size_x << ", h=" << this->filter_size_y << ") than the input patches (w=" << (*itr)->w << ", h=" << (*itr)->w << "). I will resize the patches to match the detector size. Although this works, it is slower, and there might be a decrease in result quality." << std::endl;
				std::cout << "[VDetVecMach] Not yet implemented!" << std::endl;
				warningPrinted = true;
			}
			//resize
		}
		passed = classify((*itr));
		if(passed) {
			candidates.push_back((*itr));	// SVM&WVM only added if feature, SVR/WVR always added
		}
	}

	return candidates;
}


int VDetectorVectorMachine::extractAndHistEq64(const Pyramid* pyr, FdPatch* fp)
{

	fp->data = new unsigned char[filter_size_x*filter_size_y];
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
	int start_orig = (fp->c.y_py-(filter_size_y/2))*pyr->w + (fp->c.x_py-(filter_size_x/2));	
	int startcoord = start_orig;
	int index=0;    // patch-height, det_filtersize_y
	for (int z=0; z<filter_size_y; z++) {		// could be made with modulo and 1 for loop 0-399 but this is supposed to be faster
					 // patch-width, det_filtersize_x
		for(int i=0; i<filter_size_x; i++) { // TODO change 400 to dynamic val
			//patch_to_equalize[index] = this_pyramid->data[startcoord];
			pdf_bins[LUT_bin[pyr->data[startcoord]]]=pdf_bins[LUT_bin[pyr->data[startcoord]]]+1;
			startcoord++;
			index++;
		}									// patch-width
		startcoord=startcoord+(pyr->w-filter_size_x);
	}

	// stretch the probability density function (PDF)
	for(int i=0; i<64; i++) {
		if(pdf_bins[i]!=0) {
			pdf_bins[i]=pdf_bins[i]*stretch_fac;
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
		LUTeq[i]=cdf_BINS[LUT_bin[i]];
	}
	//printf("HistEq64 - Fill the patch with data.\n");
	// equalize the patch with the help of the LUT:
	index=0;    // patch-height, det_filtersize_y
	for (int z=0; z<filter_size_y; z++) {		// could be made with modulo and 1 for loop 0-399 but this is supposed to be faster
					 // patch-width, det_filtersize_x
		for(int i=0; i<filter_size_x; i++) { // TODO change 400 to dynamic val
			//patch_to_equalize[index] = this_pyramid->data[startcoord]; // original
			fp->data[index] = (unsigned char)floor(LUTeq[pyr->data[start_orig]]+0.5);
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
			start_orig++;
			index++;
		}									// patch-width
		start_orig=start_orig+(pyr->w-filter_size_x);
	}

	// we are finished! result is equal to the result from the matlab algorithm.
	// (double checked. A few values are off by 1 but this is supposed to be a small rounding/precision issue that does not really make a difference)

	return 1;
}