#include "stdafx.h"
#include "CascadeWvmOeSvmOe.h"

#include "FdImage.h"
#include "SLogger.h"

CascadeWvmOeSvmOe::CascadeWvmOeSvmOe(void)
{
	wvm = new DetectorWVM();	// only init and set everything to 0
	svm = new DetectorSVM();	// only init and set everything to 0
	oe = new OverlapElimination();	// only init and set everything to 0
}


CascadeWvmOeSvmOe::~CascadeWvmOeSvmOe(void)
{
	delete oe;
	delete wvm;
	delete svm;
}

CascadeWvmOeSvmOe::CascadeWvmOeSvmOe(std::string mat_fn)
{
	wvm = new DetectorWVM();	// only init and set everything to 0
	wvm->load(mat_fn);

	svm = new DetectorSVM();	// only init and set everything to 0
	svm->load(mat_fn);
	
	oe = new OverlapElimination();	// only init and set everything to 0
	oe->load(mat_fn);
}

int CascadeWvmOeSvmOe::init_for_image(FdImage* myimg)
{
	wvm->init_for_image(myimg);
	svm->init_for_image(myimg);
	return 1;
}

int CascadeWvmOeSvmOe::detect_on_image(FdImage* myimg)
{
	wvm->extractToPyramids(myimg);
	this->candidates = wvm->detect_on_image(myimg);
	Logger->LogImgDetectorCandidates(myimg, candidates, wvm->getIdentifier(), "1RVM");
	
	std::vector<FdPatch*> afterFirstOE;
	//0: only after SVM, 1: only before SVM, n: Reduce each cluster to n before SVM and to 1 after; Default: 1
	if(oe->doOE!=0) {		// do overlapeliminiation after RVM before SVM
		if (oe->doOE==1)
			afterFirstOE = oe->eliminate(this->candidates, wvm->getIdentifier());
		else
			//oe->eliminate(faces, args.pp_oe_percent, args.doesPPOverlapElimination, false);	// reduce to "doOE"-num Clusters
			afterFirstOE = oe->eliminate(this->candidates, wvm->getIdentifier());
			
	} else {
		afterFirstOE = this->candidates;
	}
	Logger->LogImgDetectorCandidates(myimg, afterFirstOE, wvm->getIdentifier(), "2RVMoe"); // 4th argument optional: colorBy=string getID

	std::vector<FdPatch*> tmp;
	tmp = svm->detect_on_patchvec(afterFirstOE);
	Logger->LogImgDetectorCandidates(myimg, tmp, svm->getIdentifier(), "3SVM");

	this->candidates.clear();
	// OE after SVM:
	//float dist = dist;
	//float ratio = 0.0;
	this->candidates = oe->eliminate(tmp, svm->getIdentifier());	// always do it. But doesnt do anything more than already done by first OE?
	Logger->LogImgDetectorCandidates(myimg, candidates, svm->getIdentifier(), "4SVMoe");
	
	tmp.clear();
	tmp = oe->exp_num_fp_elimination(this->candidates, svm->getIdentifier());
	Logger->LogImgDetectorCandidates(myimg, tmp, svm->getIdentifier(), "5ExpNum");

	this->candidates = tmp;
	
	if(Logger->getVerboseLevelText()>=1) {
		std::cout << "[CascadeWvmOeSvmOe] Finished detecting on " << myimg->filename << "." << std::endl;
	}

	return 1;
}