#include "stdafx.h"
#include "RegressorWVR.h"

#include "MatlabReader.h"
#include "FdPatch.h"
#include "IImg.h"
#include "SLogger.h"

#include <iostream>
#include <cmath>

RegressorWVR::RegressorWVR(void)
{

	identifier = "RegressorWVR";

	lin_filters			= 0;
	lin_thresholds		= 0;
	nLinFilters			= 0;
	lin_hierar_thresh	= 0;
	hk_weights			= 0;

	area = NULL;
	app_rsv_convol = NULL;

	filter_output = NULL;
	u_kernel_eval = NULL;

}


RegressorWVR::~RegressorWVR(void)
{
	if (lin_filters != 0) {
		for (int i = 0; i < nLinFilters; ++i)
			delete [] lin_filters[i];
	}
	delete [] lin_filters;
	if (hk_weights != 0)
		for (int i = 0; i < nLinFilters; ++i) delete [] hk_weights[i];
	delete [] hk_weights;
	delete [] lin_thresholds;
	delete [] lin_hierar_thresh;

	if (area!=NULL)	 {
		for (int i=0;i<nLinFilters;i++)
			if (area[i]!=NULL) 
				delete area[i];
		delete[] area;
	}
	if (app_rsv_convol!=NULL) delete [] app_rsv_convol;

	if (filter_output!=NULL) delete [] filter_output;
	if (u_kernel_eval!=NULL) delete [] u_kernel_eval;
}

/*
Returns a bool if the patch passes the cascade (always true for SVMs, for WVM, only true if passes last filter)
*/
bool RegressorWVR::classify(FdPatch* fp)
{
	//std::cout << "[RegrWVR] Classifying!\n";

	// Check if the patch has already been classified by this detector! If yes, don't do it again.
	FoutMap::iterator it = fp->fout.find(this->getIdentifier());
	if(it!=fp->fout.end()) {	// The fout value is already in the map
								// Assumption: When fp->fout is not found, we assume that also fp->certainty is not yet set! This assumption should always hold.
		if(Logger->getVerboseLevelText()>=4) {
			std::cout << "[RegressorWVR] An element 'fout' already exists for this detector. 'fout' not changed. Not running the same detector twice over a patch." << std::endl;
		}
		return true;
	} // else: The fout value is not already computed. Go ahead.

	// patch II of fp already calc'ed?
	if(fp->iimg_x == NULL) {
		fp->iimg_x = new IImg(this->filter_size_x, this->filter_size_y, 8);
		fp->iimg_x->calIImgPatch(fp->data, false);
	}
	if(fp->iimg_xx == NULL) {
		fp->iimg_xx = new IImg(this->filter_size_x, this->filter_size_y, 8);
		fp->iimg_xx->calIImgPatch(fp->data, true);
	}

	for (int n=0;n<this->nLinFilters_wvm;n++) {
		u_kernel_eval[n]=0.0f;
	}
	int filter_level=-1;
	float fout = 0.0;
	do {
		filter_level++;
		fout = this->linEvalWvmHisteq64(filter_level, (filter_level%this->nLinFilters_wvm), fp->c.x_py, fp->c.y_py, filter_output, u_kernel_eval, fp->iimg_x, fp->iimg_xx);
	} while (/*fout >= this->lin_hierar_thresh[filter_level] &&*/ filter_level+1 < this->nLinFilters); //280
	
	//fp->fout = fout;
	std::pair<FoutMap::iterator, bool> fout_insert = fp->fout.insert(FoutMap::value_type(this->identifier, fout));
	if(fout_insert.second == false) {
		std::cout << "[RegressorWVR] An element 'fout' already exists for this detector, you classified the same patch twice. This should never happen." << std::endl;
	}
	
	//fp->certainty = NOTCOMPUTED;//1.0f / (1.0f + exp(posterior_wrvm[0]*fout + posterior_wrvm[1]));
	// TODO: filter statistics, nDropedOutAsNonFace[filter_level]++;
	//if(filter_level+1 == this->nLinFilters) {
	//	fp->writePNG("pos.png");
	return true;
	//}
	//return false;
}

/*
 * WVM evaluation with a histogram equalized patch (64 bins) and patch integral image 
*/
float RegressorWVR::linEvalWvmHisteq64(
												int level, int n, int x, int y, //n: n-th WSV at this apprlevel
												float* hk_kernel_eval,
												float* u_kernel_eval,
												const IImg* iimg_x/*=NULL*/, 
												const IImg* iimg_xx/*=NULL*/      ) const 
{
	/* iimg_x and iimg_xx are now patch-integral images! */

	float *this_weight = hk_weights[level];
	float res = -lin_thresholds[level];
	double norm = 0.0F;
	int p;
	const int fx = 0;
	const int fy = 0;
	const int lx = filter_size_x-1;
	const int ly = filter_size_y-1;


		
	//###########################################
	// compute evaluation and return it to fout.Pixel(x,y)
	// eval = sum_{i=1,...,level} [ hk_weights[level][i] * exp(-basisParam*(lin_filters[i]-img.data)^2) ]
	//      = sum_{i=1,...,level} [ hk_weights[level][i] * exp(-basisParam*norm[i]) ]
	//      = sum_{i=1,...,level} [ hk_weights[level][i] * hk_kernel_eval[i] ]

	//===========================================
	// first, compute this kernel and save it in hk_kernel_eval[level], 
	// because the hk_kernel_eval[0,...,level-1] are the same for each new level
	// hk_kernel_eval[level] = exp( -basisParam * (lin_filters[level]-img.data)^2 )
	//                       = exp( -basisParam * norm[level] )

	//rvm_kernel_begin = clock();
	//rvm_norm_begin = clock();


	/* adjust wvm calculation to patch-II */

	//.........................................
	// calculate the norm for that kernel (hk_kernel_eval[level] = exp(-basisParam*(lin_filters[level]-img.data)^2))
	//  norm[level] = ||x-z||^2 = (lin_filters[level]-img.data)^2   approximated by     
	//  norm[level] = ||x-p||^2 = x*x - 2*x*p + p*p   with  (x: cur. patch img.data, p: appr. RSV lin_filters[level])

	double	norm_new = 0.0F,sum_xp = 0.0F,sum_xx = 0.0F,sum_pp = 0.0F;
	float	sumv = 0.0f,sumv0 = 0.0f;
	int		r,v,/*uur,uull,dll,dr,*/ax1,ax2,ay1,ay1w,ay2w;
	PRec	rec;


	//1st term: x'*x (integral image over x^2)
	//sxx_begin = clock();
	//norm_new=iimg_xx->ISumV(0,0,0,399,0,0,lx,ly);
	//norm_new=iimg_xx->ISum(fx,fy,lx,ly);
	//norm_new=iimg_xx->data[dr];
	//uur=(fy-1)*20/*img.w*/ + lx; uull=(fy-1)*20/*img.w*/ + fx-1; dll=ly*20/*img.w*/ + fx-1; dr=ly*20/*img.w*/ + lx;
	const int dr=ly*filter_size_x/*img.w*/ + lx;
	/*if (fx>0 && fy>0)  {
		norm_new= iimg_xx->data[dr] - iimg_xx->data[uur] - iimg_xx->data[dll] + iimg_xx->data[uull];
		sumv0=    iimg_x->data[dr]  - iimg_x->data[uur]  - iimg_x->data[dll]  + iimg_x->data[uull]; 
	} else if (fx>0)   {
		norm_new= iimg_xx->data[dr] - iimg_xx->data[dll]; sumv0= iimg_x->data[dr] - iimg_x->data[dll];
	} else if (fy>0)	{
	    norm_new= iimg_xx->data[dr] - iimg_xx->data[uur]; sumv0= iimg_x->data[dr] - iimg_x->data[uur];
	} else {*///if (fx==0 && fy==0)
	    norm_new= iimg_xx->data[dr]; sumv0= iimg_x->data[dr];
	//}
	sum_xx=norm_new;

	//Profiler.sxx += (double)(clock()-sxx_begin);

	//2nd term : 2x'*p 
	//    (sum 'sum_xp' over the sums 'sumv' for each gray level of the appr. RSV 
	//     over all rectangles of that gray level which are calculated by integral input image
	//     multiplied the sum_v by the gray value)
	//
	//		dh. sum_xp= sumv0*val0 - sum_{v=1}^cntval ( sum_{r=0}^cntrec_v(iimg_x->ISum(rec_{r,v})) * val_v )
	// also we can simplify 
	//		2x'*p = 2x'* ( sum_{l=0}^{lev-1}(res_{l,n}) + res_{lev,n} ) for the n-th SV at apprlevel lev
	//		      = 2x'*u_{lev-1,n} + 2x'*res_{lev,n} 
	//		      = u_kernel_eval[n] + sum_xp, with p=res_{lev,n}
	//                                        and u_kernel_eval[n]_{lev+1}=u_kernel_eval[n] + sum_xp

	//sxp_begin = clock();
	//sumv0=iimg_x->ISum(fx,fy,lx,ly);
	//sumv0=iimg_x->ISumV(0,0,0,399,0,0,lx,ly);
	//sumv0=iimg_x->data[dr];
	for (v=1;v<area[level]->cntval;v++) {
		sumv=0;
		for (r=0;r<area[level]->cntrec[v];r++)   {
			rec=&area[level]->rec[v][r];
			//sxp_iimg_begin = clock();
			//if (rec->x1==rec->x2 && rec->y1==rec->y2) 
			//	sumv+=img.data[(fy+rec->y1)*img.w+(fx+rec->x1)];
			//
			////TODO: fasten up for lines 
			////else if (rec->x1==rec->x2 || rec->y1==rec->y2)
			//
			//else //if (rec->x1!=rec->x2 && rec->y1!=rec->y2)
			//	sumv+=iimg_x->ISum(fx+rec->x1,fy+rec->y1,fx+rec->x2,fy+rec->y2);
			//	sumv+=iimg_x->ISumV(rec->uull,rec->uur,rec->dll,rec->dr,rec->x1,rec->y1,rec->x2,rec->y2);
			//	sumv+=   iimg_x->data[rec->dr]                   - ((rec->y1>0)? iimg_x->data[rec->uur]:0) 
			//		   - ((rec->x1>0)? iimg_x->data[rec->dll]:0) + ((rec->x1>0 && rec->y1>0)? iimg_x->data[rec->uull]:0);
			ax1=fx+rec->x1-1; ax2=fx+rec->x2; ay1=fy+rec->y1;
			ay1w=(ay1-1)*filter_size_x/*img.w*/; ay2w=(fy+rec->y2)*filter_size_x/*img.w*/; 
			if (ax1+1>0 && ay1>0)
				sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay1w +ax2]
					   - iimg_x->data[ay2w +ax1] + iimg_x->data[ay1w +ax1];
			else if	(ax1+1>0)
				sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay2w +ax1];
			else if	(ay1>0)
            	sumv+=   iimg_x->data[ay2w +ax2] - iimg_x->data[ay1w +ax2];
			else //if (ax1==0 && ay1==0)
            	sumv+=   iimg_x->data[ay2w +ax2];

			//Profiler.sxp_iimg += (double)(clock()-sxp_iimg_begin);
		}
		//sxp_mval_begin = clock();
		sumv0-=sumv;
		sum_xp+=sumv*area[level]->val[v];
		//Profiler.sxp_mval += (double)(clock()-sxp_mval_begin);
	}
	sum_xp+=sumv0*area[level]->val[0];
	sum_xp+=u_kernel_eval[n];
	u_kernel_eval[n]=sum_xp;  //update u_kernel_eval[n]

	norm_new-=2*sum_xp;
	//Profiler.sxp += (double)(clock()-sxp_begin);

	//3rd term: p'*p (convolution of the appr. RSV - constrant calculated at the training)
	//spp_begin = clock();
	sum_pp=app_rsv_convol[level]; // patrik: �berfl�ssig?
	norm_new+=app_rsv_convol[level];
	//Profiler.spp += (double)(clock()-spp_begin);

	norm=norm_new;

	//Profiler.rvm_norm += (double)(clock()-rvm_norm_begin);

	//.........................................
	// calculate  now this kernel and save it in hk_kernel_eval[level], 
	// hk_kernel_eval[level] = exp(-basisParam*(lin_filters[level]-img.data)^2) ]

	hk_kernel_eval[level] = (float)(exp(-basisParam*norm)); //save it, because they 0...level-1 the same for each new level 

	//===========================================
	// second, sum over all the kernels to get the output
	// eval = sum_{i=1,...,level} [ hk_weights[level][i] * exp(-basisParam*(lin_filters[i]-img.data)^2) ]
	//      = sum_{i=1,...,level} [ hk_weights[level][i] * hk_kernel_eval[i] ]

	for (p = 0; p <= level; ++p) //sum k=0...level = b_level,k * Kernel_k
		res += this_weight[p] * hk_kernel_eval[p];

	//Profiler.rvm_kernel += (double)(clock()-rvm_kernel_begin);

	return res;
}


int RegressorWVR::load(const std::string filename)
{
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] Loading " << filename << std::endl;
	}

	MatlabReader *configReader = new MatlabReader(filename);
	int id;
	char buff[255], key[255], pos[255];

	if(!configReader->getKey("FD.ffp", buff)) {	// which feature point does this detector detect?
		std::cout << "[RegrWVR] Warning: Key in Config nicht gefunden, key:'FD.ffp'" << std::endl;
	} else {
		if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
			std::cout << "[RegrWVR] ffp: " << atoi(buff) << std::endl;
		}
	}

//	if (!configReader->getKey("ALLGINFO.outputdir", this->outputPath)) // Output folder of this detector
//		std::cout << "[RegrWVR] Warning: Key in Config nicht gefunden, key:'ALLGINFO.outputdir'" << std::endl;
//	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
//		std::cout << "[RegrWVR] outputdir: " << this->outputPath << std::endl;
//	}

	//min. und max. erwartete Anzahl Gesichter im Bild (vorerst null bis eins);											  
	sprintf(pos,"FD.expected_number_faces.#%d",0);																		  
	if (!configReader->getKey(pos,buff))																						  
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'" << pos << "', nehme Default: " << this->expected_num_faces[0] << std::endl;
	else
		this->expected_num_faces[0]=atoi(buff);
	sprintf(pos,"FD.expected_number_faces.#%d",1);
	if (!configReader->getKey(pos,buff))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'" << pos << "', nehme Default: " << this->expected_num_faces[1] << std::endl;
	else
		this->expected_num_faces[1]=atoi(buff);

	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] expected_num_faces: " << this->expected_num_faces[0] << ", " << this->expected_num_faces[1] << std::endl;
	}

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr SVM-Schwelle)
	/*if (!configReader->getKey("FD.limit_reliability",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n",
		"FD.limit_reliability",args->threshold_fullsvm);
	else args->threshold_fullsvm=(float)atof(buff); */

	//ROI: left, top, right, bottom
    // 0 0 0 0 (ganze Bild), -1 -1 -1 -1 (bzw. ganze FD-ROI) 
	int v=1;
	if (!configReader->getInt("FD.roi.#0",&v))		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.roi.#0', nehme Default: " << this->roiDistFromBorder.left << std::endl;
	else										this->roiDistFromBorder.left=v;
	if (!configReader->getInt("FD.roi.#1",&v))		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.roi.#1', nehme Default: " << this->roiDistFromBorder.top << std::endl;
	else										this->roiDistFromBorder.top=v;
	if (!configReader->getInt("FD.roi.#2",&v))		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.roi.#2', nehme Default: " << this->roiDistFromBorder.right << std::endl;
	else										this->roiDistFromBorder.right=v;
	if (!configReader->getInt("FD.roi.#3",&v))		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.roi.#3', nehme Default: " << this->roiDistFromBorder.bottom << std::endl;
	else										this->roiDistFromBorder.bottom=v;
	
	//Minimale Gesichtsoehe in Pixel 
	if (!configReader->getInt("FD.face_size_min",&this->subsamplingMinHeight))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.face_size_min', nehme Default: " << this->subsamplingMinHeight << std::endl;
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] face_size_min: " << this->subsamplingMinHeight << std::endl;
	}
	//Anzahl der Skalierungen
	if (!configReader->getInt("FD.maxscales",&this->numSubsamplingLevels))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.maxscales', nehme Default: " << this->numSubsamplingLevels << std::endl;
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] maxscales: " << this->numSubsamplingLevels << std::endl;
	}
	//Scalierungsfaktor 
	if (!configReader->getKey("FD.scalefactor",buff))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.scalefactor', nehme Default: " << this->subsamplingFactor << std::endl;
	else
		this->subsamplingFactor=(float)atof(buff);
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] scalefactor: " << this->subsamplingFactor << std::endl;
	}

	//Number filters to use
	if (!configReader->getKey("FD.numUsedFilter",buff))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.numUsedFilter', nehme Default: " << this->numUsedFilter << std::endl;
	else this->numUsedFilter=std::max(0,atoi(buff));

	//Grenze der Zuverlaesigkeit ab der Gesichter aufgenommen werden (Diffwert fr W-RSV's-Schwellen)
	// zB. +0.1 => weniger patches dr�ber(mehr rejected, langsamer),    dh. mehr fn(FRR), weniger fp(FAR)  und
	// zB. -0.1 => mehr patches dr�ber(mehr nicht rejected, schneller), dh. weniger fn(FRR), mehr fp(FAR)
	//We don't need this for Regression!
	/*if (!configReader->getKey("FD.limit_reliability_filter",buff))
		fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: %g\n",
		"FD.limit_reliability_filter",this->limit_reliability_filter);
	else this->limit_reliability_filter=(float)atof(buff);*/

	//Kassifikator
	char fn_classifier[500];
	if (!configReader->getKey("FD.classificator", fn_classifier))
		std::cout << "[RegrWVR] WARNING: Key in Config nicht gefunden, key:'FD.classificator', nehme Default: " << fn_classifier << std::endl;

	//Schwellwerte
	//We don't need/have this for Regression
	//char fn_threshold[500];
	//if (!configReader->getKey("FD.threshold", fn_threshold))
	//	fprintf(stderr,"WARNING: Key in Config nicht gefunden, key:'%s', nehme Default: '%s'\n", "FD.threshold", fn_threshold);

	delete configReader;

	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] Loading " << fn_classifier << std::endl;
	}
	MATFile *pmatfile;
	mxArray *pmxarray; // =mat
	double *matdata;
	pmatfile = matOpen(fn_classifier, "r");
	if (pmatfile == NULL) {
		std::cout << "[RegrWVR] Error: Opening file" << std::endl;
		exit(EXIT_FAILURE);
	}

	pmxarray = matGetVariable(pmatfile, "num_hk");
	if (pmxarray == 0) {
		std::cout << "[RegrWVR] Error: There is a no num_hk in the file." << std::endl;
		exit(EXIT_FAILURE);
	}
	matdata = mxGetPr(pmxarray);
	int nfilter = (int)matdata[0];
	mxDestroyArray(pmxarray);
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] Found " << nfilter << " WVM filters" << std::endl;
	}

	pmxarray = matGetVariable(pmatfile, "wrvm");
	if (pmxarray != 0) { // read area
		std::cout << "[RegrWVR] Found structur 'wrvm', reading the TODO 280 non linear filters support_hk* and weight_hk* at once" << std::endl;
		std::cout << "[RegrWVR] Stopping, because this is not yet tested!" << std::endl;
		exit(EXIT_FAILURE);
		/*
				//read dim. hxw of support_hk's 
				mxArray* msup=mxGetField(pmxarray, 0, "dim");
				if (msup == 0) {
					printf("\nfd_ReadDetector(): Unable to find the matrix \'wrvm.dim\'\n");
					return 0;
				}
				matdata = mxGetPr(msup);
				int w = (int)matdata[0];
				int h = (int)matdata[1];
				if(w==0 || h==0) {
					//error, stop
					return 0;
				}
				this->filter_size_x = w;
				this->filter_size_y = h;
				this->numSV = nfilter;


				nLinFilters = nfilter;
				lin_filters = new float* [nLinFilters];
				for (i = 0; i < nLinFilters; ++i) 
					lin_filters[i] = new float[nDim];
				lin_thresholds = new float [nLinFilters];
				lin_hierar_thresh = new float [nLinFilters];
				hk_weights = new float* [nLinFilters];
				for (i = 0; i < nLinFilters; ++i) 
					hk_weights[i] = new float[nLinFilters];


				//read matrix support_hk's w*h x nfilter
				//i.e. filter are vectorized to size(w*h) x 1 (row-wise in C format not col-wise as in matlab)
				msup=mxGetField(pmxarray,0,"support_hk");
				if (msup == 0) {
					sprintf(buff, "\nfd_ReadDetector(): Unable to find the matrix \'wrvm.support_hk\' in\n'%s' \n",args->classificator);
					fprintf(stderr,buff); PUT(buff);
					throw buff;	return 4;
				}
				if (mxGetNumberOfDimensions(msup) != 2) {
					fprintf(stderr, "\nThe matrix \'wrvm.support_hk\' in the file %s should have 2 dimensions, but has %d\7\n",args->classificator, mxGetNumberOfDimensions(msup));
					return 4;
				}
				const MYMWSIZE *dim = mxGetDimensions(msup);
				int size = (int)dim[0], n = (int)dim[1]; 

				if ( n!=nfilter || w*h!=size) {
					fprintf(stderr, "\nfd_ReadDetector(): The dimensions of matrix \'wrvm.support_hk\' should be (size=w*h=%d x nfilter=%d), but is %dx%d\7\n",w*h,nfilter,size,n);
					return 4;
				}
				//fprintf(stdout, "fd_ReadDetector(): dimensions of matrix \'wrvm.support_hk\' should be (size=w*h=%d x nfilter=%d), and is %dx%d\n",w*h,nfilter,size,n);
				
				matdata = mxGetPr(msup);
				for (int i = 0, j=0; i < nfilter; ++i) {
					for (k = 0; k < size; ++k)
						detector->lin_filters[i][k] = 255.0f*(float)matdata[j++];	// because the training images grey level values were divided by 255;
				}
				
				//read matrix weight_hk (1,...,i,...,nfilter) x nfilter
				//only the first 0,...,i-1 are set, all other are set to 0 in the matrix weight_hk
				msup=mxGetField(pmxarray,0,"weight_hk");
				if (msup == 0) {
					sprintf(buff, "\nfd_ReadDetector(): Unable to find the matrix \'wrvm.weight_hk\' in\n'%s' \n",args->classificator);
					fprintf(stderr,buff); PUT(buff);
					throw buff;	return 4;
				}
				if (mxGetNumberOfDimensions(msup) != 2) {
					fprintf(stderr, "\nThe matrix \'wrvm.weight_hk\' in the file %s should have 2 dimensions, but has %d\7\n",args->classificator, mxGetNumberOfDimensions(msup));
					return 4;
				}
				dim = mxGetDimensions(msup);
				int n1 = (int)dim[0], n2 = (int)dim[1]; 

				if ( n1!=nfilter || n2!=nfilter) {
					fprintf(stderr, "\nfd_ReadDetector(): The dimensions of matrix \'wrvm.weight_hk\' should be (nfilter=%d x nfilter=%d), but is %dx%d\7\n",nfilter,nfilter,n1,n2);
					return 4;
				}
				//fprintf(stdout, "fd_ReadDetector(): dimensions of matrix \'wrvm.weight_hk\' should be (nfilter=%d x nfilter=%d), and is %dx%d\n",nfilter,nfilter,n1,n2);
				
				matdata = mxGetPr(msup);
				for (int i = 0, r=0; i < nfilter; ++i, r+=nfilter) {
					for (k = 0; k <= i; ++k)
						detector->hk_weights[i][k] = (float)matdata[r + k];	
				}

				//for (k = 0; k < 5; ++k)	fprintf(stdout, "%1.2f ",detector->hk_weights[4][k]);	
				//fprintf(stdout, "\n");	
	
				mxDestroyArray(pmxarray);
				printf("...done\n");
				*/



	} else {	// read seq.
		if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
			std::cout << "[RegrWVR] Unable to find structur 'wrvm', reading the " << nfilter << " non linear filters support_hk* and weight_hk* sequentially (slower)" << std::endl;
		}
		char str[100];
		sprintf(str, "support_hk%d", 1);
		pmxarray = matGetVariable(pmatfile, str);
		const mwSize *dim = mxGetDimensions(pmxarray);
		int h = (int)dim[0];
		int w = (int)dim[1];
		//assert(w && h);
		this->filter_size_x = w;	// TODO check if this is right with eg 24x16
		this->filter_size_y = h;
					
		nLinFilters = nfilter;
		lin_filters = new float* [nLinFilters];
		for (int i = 0; i < nLinFilters; ++i) 
			lin_filters[i] = new float[w*h];
		
		lin_hierar_thresh = new float [nLinFilters];
		hk_weights = new float* [nLinFilters];
		for (int i = 0; i < nLinFilters; ++i) 
			hk_weights[i] = new float[nLinFilters];

		if (pmxarray == 0) {
			std::cout << "[RegrWVR] Unable to find the matrix \'support_hk1\'" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (mxGetNumberOfDimensions(pmxarray) != 2) {
			std::cout << "[RegrWVR] The matrix \'filter1\' in the file should have 2 dimensions" << std::endl;
			exit(EXIT_FAILURE);
		}
		mxDestroyArray(pmxarray);

		for (int i = 0; i < this->nLinFilters; i++) {

			sprintf(str, "support_hk%d", i+1);
			pmxarray = matGetVariable(pmatfile, str);
			if (pmxarray == 0) {
				std::cout << "[RegrWVR] Unable to find the matrix \'support_hk" << i+1 << "\'" << std::endl;
				exit(EXIT_FAILURE);
			}
			if (mxGetNumberOfDimensions(pmxarray) != 2) {
				std::cout << "[RegrWVR] The matrix \'filter" << i+1 << "\' in the file should have 2 dimensions" << std::endl;
				exit(EXIT_FAILURE);
			}

			matdata = mxGetPr(pmxarray);
				
			int k = 0;
			for (int x = 0; x < this->filter_size_x; ++x)
				for (int y = 0; y < this->filter_size_y; ++y)
					this->lin_filters[i][y*this->filter_size_x+x] = 255.0f*(float)matdata[k++];	// because the training images grey level values were divided by 255;
			mxDestroyArray(pmxarray);

			sprintf(str, "weight_hk%d", i+1);
			pmxarray = matGetVariable(pmatfile, str);
			if (pmxarray != 0) {
				const mwSize *dim = mxGetDimensions(pmxarray);
				if ((dim[1] != i+1) && (dim[0] != i+1)) {
					std::cout << "[RegrWVR] The matrix " << str << " in the file should have a dimensions 1x" << i+1 << " or " << i+1 << "x1" << std::endl;
					exit(EXIT_FAILURE);
				}
				matdata = mxGetPr(pmxarray);
				for (int j = 0; j <= i; ++j) {
					this->hk_weights[i][j] = (float)matdata[j];
				}
				mxDestroyArray(pmxarray);
			}
		}	// end for over numHKs
		if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
			std::cout << "[RegrWVR] Done" << std::endl;
		}
	}// end else read vecs/weights sequentially
	
	pmxarray = matGetVariable(pmatfile, "param_nonlin1_rvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		this->nonlin_threshold = (float)matdata[0];
		this->nonLinType       = (int)matdata[1];
		this->basisParam       = (float)(matdata[2]/65025.0); // because the training images grey level values were divided by 255
		this->polyPower        = (int)matdata[3];
		this->divisor          = (float)matdata[4];
		mxDestroyArray(pmxarray);
	} else {
		pmxarray = matGetVariable(pmatfile, "param_nonlin1");
		if (pmxarray != 0) {
			matdata = mxGetPr(pmxarray);
			this->nonlin_threshold = (float)matdata[0];
			this->nonLinType       = (int)matdata[1];
			this->basisParam       = (float)(matdata[2]/65025.0); // because the training images grey level values were divided by 255
			this->polyPower        = (int)matdata[3];
			this->divisor          = (float)matdata[4];
			mxDestroyArray(pmxarray);
		}
	}
	lin_thresholds = new float [nLinFilters];
	for (int i = 0; i < nLinFilters; ++i) {			//wrvm_out=treshSVM+sum(beta*kernel) 
		lin_thresholds[i] = (float)nonlin_threshold;
	}

	// number of filters per level (eg 14)
	pmxarray = matGetVariable(pmatfile, "num_hk_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);	// TODO REMOVE
		this->nLinFilters_wvm = (int)matdata[0]; 
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[RegrWVR] 'num_hk_wvm' not found in:" << std::endl;
		exit(EXIT_FAILURE);
	}
	// number of levels with filters (eg 20)
	pmxarray = matGetVariable(pmatfile, "num_lev_wvm");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		assert(matdata != 0);
		this->nLevels_wvm = (int)matdata[0]; 
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[RegrWVR] 'num_lev_wvm' not found in:" << std::endl;
		exit(EXIT_FAILURE);
	}

  
	//read recangles in area
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=2) {
		std::cout << "[RegrWVR] Reading rectangles in area..." << std::endl;
	}
	pmxarray = matGetVariable(pmatfile, "area");
	if (pmxarray != 0 && mxIsStruct(pmxarray)) {
		int r,v,hrsv,w,h;
		int *cntrec;
		double *d;
		TRec *rec;
		const char valstr[]="val_u";
		const char cntrecstr[]="cntrec_u";

		w=this->filter_size_x; // again?
		h=this->filter_size_y;

		const mwSize *dim=mxGetDimensions(pmxarray);
		int nHK=(int)dim[1];
		if (this->nLinFilters!=nHK){
			std::cout << "[RegrWVR] 'area' not right dim:" << nHK << " (==" << this->nLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		if ((this->nLinFilters_wvm*this->nLevels_wvm)!=nHK){
			std::cout << "[RegrWVR] 'area' not right dim:" << nHK << " (==" << this->nLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}

		this->area = new Area*[nHK];

		int cntval;
		for (hrsv=0;hrsv<nHK;hrsv++)  {

			mxArray* mval=mxGetField(pmxarray,hrsv,valstr);
 			if (mval == NULL ) {
				std::cout << "[RegrWVR] '" << valstr << "' not found (WVM: 'val_u', else: 'val', right *.mat/kernel?)" << std::endl;
				exit(EXIT_FAILURE);
			}
			dim=mxGetDimensions(mval);
			cntval=(int)dim[1];

			cntrec = new int[cntval];
			mxArray* mcntrec=mxGetField(pmxarray,hrsv,cntrecstr);
			d = mxGetPr(mcntrec);
			for (v=0;v<cntval;v++) 
				cntrec[v]=(int)d[v];
			this->area[hrsv] = new Area(cntval,cntrec);

			d = mxGetPr(mval);
			for (v=0;v<cntval;v++) 
				this->area[hrsv]->val[v]=d[v]*255.0F; // because the training images grey level values were divided by 255;

			mxArray* mrec=mxGetField(pmxarray,hrsv,"crec");
			for (v=0;v<cntval;v++) 
				for (r=0;r<this->area[hrsv]->cntrec[v];r++) {
					mxArray* mk;
					rec=&this->area[hrsv]->rec[v][r];

					mk=mxGetField(mrec,r*cntval+v,"x1");
					d = mxGetPr(mk); rec->x1=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"y1");
					d = mxGetPr(mk); rec->y1=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"x2");
					d = mxGetPr(mk); rec->x2=(int)d[0];

					mk=mxGetField(mrec,r*cntval+v,"y2");
					d = mxGetPr(mk); rec->y2=(int)d[0];

					rec->uull=(rec->y1-1)*w + rec->x1-1;
					rec->uur= (rec->y1-1)*w + rec->x2;
					rec->dll= (rec->y2)*w   + rec->x1-1;
					rec->dr=  (rec->y2)*w   + rec->x2;
				}
	
			delete [] cntrec;

			//char name[255];
			//if (args->moreOutput) { 
			//sprintf(name,"%d",hrsv); this->area[hrsv]->dump(name);
			//}

		}
		mxDestroyArray(pmxarray);

	} else {
		std::cout << "[RegrWVR] 'area' not found (right *.mat/kernel?)" << std::endl;
		exit(EXIT_FAILURE);
	}
	//printf("[RegrWVR] Done\n");

	//printf("fd_ReadDetector(): read convolution of the appr. rsv (pp) in	mat file\n");
	//read convolution of the appr. rsv (pp) in	mat file
	pmxarray = matGetVariable(pmatfile, "app_rsv_convol");
	if (pmxarray != 0) {
		matdata = mxGetPr(pmxarray);
		const mwSize *dim = mxGetDimensions(pmxarray);
		int nHK=(int)dim[1];
		if (this->nLinFilters!=nHK){
			std::cout << "[RegrWVR] 'app_rsv_convol' not right dim:" << nHK << " (==" << this->nLinFilters << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		this->app_rsv_convol = new double[nHK];
		for (int hrsv=0; hrsv<nHK; ++hrsv)
			this->app_rsv_convol[hrsv]=matdata[hrsv]*65025.0; // because the training images grey level values were divided by 255;
		mxDestroyArray(pmxarray);
	} else {
		std::cout << "[RegrWVR] 'app_rsv_convol' not found" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (matClose(pmatfile) != 0) {
		std::cout << "[RegrWVR] Error closing file" << std::endl;
	}
	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=1) {
		std::cout << "[RegrWVR] Done reading WVR!" << std::endl;
	}


	
	//printf("fd_ReadDetector(): making the hierarchical thresholds\n");
	// making the hierarchical thresholds
	//MATFile *mxtFile = matOpen(args->threshold, "r");
	/*pmatfile = matOpen(fn_threshold, "r");
	if (pmatfile == 0) {
		printf("fd_ReadDetector(): Unable to open the file (wrong format?):\n'%s' \n", fn_threshold);
		return 1;
	} else {
		pmxarray = matGetVariable(pmatfile, "hierar_thresh");
		if (pmxarray == 0) {
			fprintf(stderr, "fd_ReadDetector(): Unable to find the matrix hierar_thresh in the file %s\n\7", fn_threshold);
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			for (int o=0; o<(int)dim[1]; ++o) {
				//TPairIf p(o+1, (float)matdata[o]);
				std::pair<int, float> p(o+1, (float)matdata[o]); // = std::make_pair<int, float>
				this->hierarchical_thresholds.push_back(p);
			}
			mxDestroyArray(pmxarray);
		}
			
		//printf("fd_ReadDetector(): read posterior_svm parameter for probabilistic SVM output\n");
		//read posterior_wrvm parameter for probabilistic WRVM output
		//TODO is there a case (when svm+wvm from same trainingdata) when there exists only a posterior_svm, and I should use this here?
		pmxarray = matGetVariable(pmatfile, "posterior_wrvm");
		if (pmxarray == 0) {
			fprintf(stderr, "WARNING: Unable to find the vector posterior_wrvm, disable prob. SVM output;\n");
			this->posterior_wrvm[0]=this->posterior_wrvm[1]=0.0f;
		} else {
			double* matdata = mxGetPr(pmxarray);
			const mwSize *dim = mxGetDimensions(pmxarray);
			if (dim[1] != 2) {
				fprintf(stderr, "WARNING: Size of vector posterior_wrvm !=2, disable prob. WRVM output;\n");
				this->posterior_wrvm[0]=this->posterior_wrvm[1]=0.0f;
			} else {
				this->posterior_wrvm[0]=(float)matdata[0]; this->posterior_wrvm[1]=(float)matdata[1];
			}
			mxDestroyArray(pmxarray);
		}
		matClose(pmatfile);
	}

	int i;
	for (i = 0; i < this->nLinFilters; ++i) {
		this->lin_hierar_thresh[i] = 0;
	}
	for (i = 0; i < this->hierarchical_thresholds.size(); ++i) {
		if (this->hierarchical_thresholds[i].first <= this->nLinFilters)
			this->lin_hierar_thresh[this->hierarchical_thresholds[i].first-1] = this->hierarchical_thresholds[i].second;
	}*/
	
	//Diffwert fuer W-RSV's-Schwellen 
	//if (this->limit_reliability_filter!=0.0)
	//	for (int i = 0; i < this->nLinFilters; ++i) this->lin_hierar_thresh[i]+=this->limit_reliability_filter;


	//for (i = 0; i < this->nLinFilters; ++i) printf("b%d=%g ",i+1,this->lin_hierar_thresh[i]);
	//printf("\n");

	if((Logger->global.text.outputFullStartup==true) || Logger->getVerboseLevelText()>=1) {
		std::cout << "[RegrWVR] Done reading WVR! (we don't use/have a threshold file yet)" << std::endl;
	}

	this->stretch_fac = 255.0f/(float)(filter_size_x*filter_size_y);	// HistEq64 initialization

	filter_output = new float[this->nLinFilters];
 	u_kernel_eval = new float[this->nLinFilters];

	return 1;
}

int RegressorWVR::initForImage(FdImage* img)
{
	initPyramids(img);
	// For non-hq64: construct II pyramids here
	initROI(img);
	return 1;
}



RegressorWVR::Area::Area(void)
{
	cntval=0;
	val=NULL;
	rec=NULL;
	cntrec=NULL;
	cntallrec=0;
}

RegressorWVR::Area::Area(int cv, int *cr) : cntval(cv)
{
	int r,v;

	val = new double[cntval];
	cntrec = new int[cntval];
	cntallrec=0;
	for (v=0;v<cntval;v++) {
		cntrec[v]=cr[v];
		cntallrec+=cr[v];
	}
	rec = new TRec*[cntval];
	for (v=0;v<cntval;v++) {
		rec[v] = new TRec[cntrec[v]];
		for (r=0;r<cntrec[v];r++) { rec[v][r].x1=rec[v][r].y1=rec[v][r].x2=rec[v][r].y2=0; }
		}
}

RegressorWVR::Area::~Area(void)
{
	if (rec!=NULL) for (int v=0;v<cntval;v++) delete [] rec[v];
	if (rec!=NULL) delete [] rec;
	if (cntrec!=NULL) delete [] cntrec;
	if (val!=NULL) delete [] val;
}

void RegressorWVR::Area::dump(char *name="") {
	int r,v;  

	std::cout << std::endl << "area" << name << ": cntval:" << cntval << ", cntallrec:" << cntallrec << ", val:";
	for (v=0;v<cntval;v++) std::cout << " " << val[v];	//printf(" %1.4f",val[v]);
	std::cout << std::endl;

	for (v=0;v<cntval;v++) {
		for (r=0;r<cntrec[v];r++) { 
			//printf("r[%d][%d]:(%d,%d,%d,%d) ",v,r,rec[v][r].x1,rec[v][r].y1,rec[v][r].x2,rec[v][r].y2);
			std::cout << "r[" << v << "][" << r << "]:(" << rec[v][r].x1 << "," << rec[v][r].y1 << "," << rec[v][r].x2 << "," << rec[v][r].y2 << ") ";
			if ((r%5)==4) 
				std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}
