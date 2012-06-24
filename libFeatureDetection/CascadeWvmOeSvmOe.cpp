#include "StdAfx.h"
#include "CascadeWvmOeSvmOe.h"


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

CascadeWvmOeSvmOe::CascadeWvmOeSvmOe(char* mat_fn)
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

	wvm->extract(myimg);
	this->candidates = wvm->detect_on_image(myimg);

	// extract the basename
	std::stringstream ss(myimg->filename);
    std::string item;
	std::vector<std::string> elems;
	char delim = '\\';
    while(std::getline(ss, item, delim)) {
        if(item!="")
			elems.push_back(item);
    }
	unsigned int elems_end = elems.size()-1;
	std::ostringstream fn_1RVM;
	fn_1RVM << "out\\" << elems[elems_end] << "_1RVM.png" << std::ends;

	Logger->drawBoxes(myimg->data_matbgr, this->candidates, wvm->getIdentifier(), true, fn_1RVM.str());
	
	std::vector<FdPatch*> afterFirstOE;
	//0: only after SVM, 1: only before SVM, n: Reduce each cluster to n before SVM and to 1 after; Default: 1
	if(oe->doOE!=0) {		// do overlapeliminiation after RVM before SVM
		if (oe->doOE==1)
			afterFirstOE = oe->eliminate(this->candidates);
		else
			//oe->eliminate(faces, args.pp_oe_percent, args.doesPPOverlapElimination, false);	// reduce to "doOE"-num Clusters
			afterFirstOE = oe->eliminate(this->candidates);
			
	} else {
		afterFirstOE = this->candidates;
	}
	std::ostringstream fn_2RVMoe;
	fn_2RVMoe << "out\\" << elems[elems_end] << "_2RVMoe.png" << std::ends;
	Logger->drawBoxes(myimg->data_matbgr, afterFirstOE, wvm->getIdentifier(), true, fn_2RVMoe.str());

	std::vector<FdPatch*> tmp;
	tmp = svm->detect_on_patchvec(afterFirstOE);
	std::ostringstream fn_3SVM;
	fn_3SVM << "out\\" << elems[elems_end] << "_3SVM.png" << std::ends;
	Logger->drawBoxes(myimg->data_matbgr, tmp, svm->getIdentifier(), true, fn_3SVM.str());

	this->candidates.clear();
	// OE after SVM:
	//float dist = dist;
	//float ratio = 0.0;
	this->candidates = oe->eliminate(tmp);	// always do it. But doesnt do anything more than already done by first OE?
	std::ostringstream fn_4SVMoe;
	fn_4SVMoe << "out\\" << elems[elems_end] << "_4SVMoe.png" << std::ends;
	Logger->drawBoxes(myimg->data_matbgr, this->candidates, svm->getIdentifier(), true, fn_4SVMoe.str());
	
	tmp.clear();
	tmp = oe->exp_num_fp_elimination(this->candidates);
	std::ostringstream fn_5ExpNum;
	fn_5ExpNum << "out\\" << elems[elems_end] << "_5ExpNum.png" << std::ends;
	Logger->drawBoxes(myimg->data_matbgr, tmp, svm->getIdentifier(), true, fn_5ExpNum.str());
	
	this->candidates = tmp;
	return 1;
}