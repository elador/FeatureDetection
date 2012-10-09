#include "stdafx.h"
#include "CascadeWvmSvm.h"


CascadeWvmSvm::CascadeWvmSvm(void)
{
	wvm = new DetectorWVM();
	svm = new DetectorSVM();
}


CascadeWvmSvm::~CascadeWvmSvm(void)
{
	delete wvm;
	delete svm;
}

CascadeWvmSvm::CascadeWvmSvm(const std::string mat_fn)
{
	
	wvm = new DetectorWVM();	// only init and set everything to 0
	wvm->load(mat_fn);

	svm = new DetectorSVM();	// only init and set everything to 0
	svm->load(mat_fn);

	
}

int CascadeWvmSvm::init_for_image(FdImage* myimg)
{
	wvm->init_for_image(myimg);
	svm->init_for_image(myimg);
	return 1;
}

int CascadeWvmSvm::detect_on_image(FdImage* myimg)
{

	wvm->extractToPyramids(myimg);
	this->candidates = wvm->detect_on_image(myimg);
	std::vector<FdPatch*> tmp;
	tmp = svm->detect_on_patchvec(this->candidates);
	this->candidates.clear();
	this->candidates = tmp;
	return 1;
}